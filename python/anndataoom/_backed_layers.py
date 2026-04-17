"""
HDF5-backed layers storage for out-of-memory AnnData.

Stores layer matrices in a sidecar HDF5 file so that layers
(counts, scaled, normlog, etc.) can be written/read lazily
without occupying main memory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from scipy.sparse import issparse, csr_matrix

from ._backed_array import BackedArray

if TYPE_CHECKING:
    pass

_LAYERS_GROUP = "layers"


class _H5BackedMatrix:
    """Thin wrapper around an HDF5 dataset that supports slicing."""

    def __init__(self, dataset: h5py.Dataset):
        self._ds = dataset

    @property
    def shape(self):
        return self._ds.shape

    @property
    def dtype(self):
        return self._ds.dtype

    def __getitem__(self, key):
        return self._ds[key]


class BackedLayers:
    """Dict-like object that stores layer matrices in an HDF5 sidecar file.

    Usage::

        layers = BackedLayers("/path/to/data.h5ad")
        layers["counts"] = adata.X  # writes to sidecar
        X = layers["counts"]        # returns BackedArray (lazy)
        X_dense = layers["counts"][:] # materializes

    The sidecar file is ``<original_path>.layers.h5``.
    """

    def __init__(
        self,
        backing_path: str | Path | None = None,
        shape: tuple[int, int] | None = None,
    ):
        if backing_path is not None:
            self._path = str(backing_path) + ".layers.h5"
        else:
            # No backing path — use a temp file
            fd, path = tempfile.mkstemp(suffix=".layers.h5")
            os.close(fd)
            self._path = path
        self._shape = shape
        self._file: h5py.File | None = None
        self._in_memory: dict[str, np.ndarray] = {}

    def _ensure_file(self, mode: str = "a") -> h5py.File:
        if self._file is None or not self._file.id.valid:
            self._file = h5py.File(self._path, mode)
            if _LAYERS_GROUP not in self._file:
                self._file.create_group(_LAYERS_GROUP)
        return self._file

    def _group(self) -> h5py.Group:
        f = self._ensure_file()
        return f[_LAYERS_GROUP]

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        if key in self._in_memory:
            return True
        try:
            return key in self._group()
        except Exception:
            return False

    def __getitem__(self, key: str) -> BackedArray:
        if key in self._in_memory:
            arr = self._in_memory[key]
            # If already a BackedArray (or subclass like ScaledBackedArray), return as-is
            if isinstance(arr, BackedArray):
                return arr
            # Return as a simple BackedArray wrapping an in-memory array
            return BackedArray(arr, shape=arr.shape)
        grp = self._group()
        if key not in grp:
            raise KeyError(f"Layer '{key}' not found")
        ds = grp[key]
        return BackedArray(_H5BackedMatrix(ds), shape=ds.shape)

    def __setitem__(self, key: str, value) -> None:
        if value is None:
            if key in self:
                del self[key]
            return

        # If value is small enough or already dense, store directly
        if isinstance(value, BackedArray):
            # Lazy array (ScaledBackedArray, TransformedBackedArray, etc.)
            # Store reference directly — don't materialise to HDF5
            self._in_memory[key] = value
            return

        if issparse(value):
            value = value.toarray()

        if isinstance(value, np.ndarray):
            if value.nbytes < 500 * 1024 * 1024:  # < 500MB: in-memory
                self._in_memory[key] = value.copy()
            else:
                self._write_dense(key, value)
        else:
            # Try to convert to numpy
            try:
                arr = np.asarray(value)
                if arr.nbytes < 500 * 1024 * 1024:
                    self._in_memory[key] = arr.copy()
                else:
                    self._write_dense(key, arr)
            except Exception:
                # Last resort: materialize via [:]
                arr = np.asarray(value[:])
                self._in_memory[key] = arr

    def __delitem__(self, key: str) -> None:
        if key in self._in_memory:
            del self._in_memory[key]
        try:
            grp = self._group()
            if key in grp:
                del grp[key]
        except Exception:
            pass

    def keys(self):
        result = set(self._in_memory.keys())
        try:
            result |= set(self._group().keys())
        except Exception:
            pass
        return list(result)

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self):
        return iter(self.keys())

    def get(self, key: str, default=None):
        if key in self:
            return self[key]
        return default

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"BackedLayers(keys={self.keys()})"

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _write_dense(self, key: str, arr: np.ndarray) -> None:
        """Write a dense array to the HDF5 sidecar."""
        grp = self._group()
        if key in grp:
            del grp[key]
        # Use chunked storage for large arrays
        chunks = (min(1000, arr.shape[0]), arr.shape[1]) if arr.ndim == 2 else None
        grp.create_dataset(
            key,
            data=arr,
            chunks=chunks,
            compression="gzip",
            compression_opts=1,
        )
        grp.file.flush()

    def _write_chunked(self, key: str, backed: BackedArray) -> None:
        """Write a BackedArray to HDF5 via chunked reads."""
        grp = self._group()
        if key in grp:
            del grp[key]
        shape = backed.shape
        # Infer dtype from first chunk
        first_chunk = backed._read_rows(0, min(1, shape[0]))
        if issparse(first_chunk):
            first_chunk = first_chunk.toarray()
        dtype = first_chunk.dtype
        ds = grp.create_dataset(
            key,
            shape=shape,
            dtype=dtype,
            chunks=(min(1000, shape[0]), shape[1]),
            compression="gzip",
            compression_opts=1,
        )
        for start, end, chunk in backed.chunked(chunk_size=1000):
            if issparse(chunk):
                chunk = chunk.toarray()
            ds[start:end] = chunk
        grp.file.flush()

    # ------------------------------------------------------------------
    # Subset support
    # ------------------------------------------------------------------

    def subset(
        self,
        obs_indices: np.ndarray | None = None,
        var_indices: np.ndarray | None = None,
    ) -> BackedLayers:
        """Create a new BackedLayers containing only the specified rows/columns.

        For layers that are small (in-memory), this is done directly.
        For backed layers, data is read in chunks and written to new sidecar.
        """
        new_layers = BackedLayers(shape=None)

        for key in self.keys():
            if key in self._in_memory:
                arr = self._in_memory[key]
                # If it's a BackedArray (lazy), wrap it in another subset layer
                if isinstance(arr, BackedArray):
                    # Import here to avoid circular import
                    from ._core import _SubsetBackedArray
                    obs_i = obs_indices if obs_indices is not None else None
                    var_i = var_indices if var_indices is not None else None
                    new_shape = (
                        len(obs_i) if obs_i is not None else arr.shape[0],
                        len(var_i) if var_i is not None else arr.shape[1],
                    )
                    new_layers._in_memory[key] = _SubsetBackedArray(arr, obs_i, var_i, new_shape)
                    continue
                if obs_indices is not None:
                    arr = arr[obs_indices]
                if var_indices is not None:
                    arr = arr[:, var_indices]
                new_layers._in_memory[key] = arr
            else:
                # Backed layer — read in chunks and subset
                src = self[key]
                chunks = []
                for start, end, chunk in src.chunked(1000):
                    if issparse(chunk):
                        chunk = chunk.toarray()
                    if obs_indices is not None:
                        # Select rows that fall within [start, end)
                        mask = (obs_indices >= start) & (obs_indices < end)
                        local_idx = obs_indices[mask] - start
                        chunk = chunk[local_idx]
                    if var_indices is not None:
                        chunk = chunk[:, var_indices]
                    if chunk.shape[0] > 0:
                        chunks.append(chunk)
                if chunks:
                    new_layers._in_memory[key] = np.vstack(chunks)

        return new_layers

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._file is not None and self._file.id.valid:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
