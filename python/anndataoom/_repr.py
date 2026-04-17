"""
AnnDataOOM representation — xarray-style plain text + HTML for Jupyter.

Provides:
- ``_format_text(adata) -> str``     : plain-text __repr__
- ``_format_html(adata) -> str``     : Jupyter _repr_html_
- ``_format_read_message(...)``      : one-off welcome banner for ov.read()
- ``_describe_storage(adata)``       : cached storage/format summary

All expensive operations (like density) are computed once and cached on the
adata instance in ``adata._repr_cache``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.sparse import issparse


# ──────────────────────────────────────────────────────────────────────
# Summaries
# ──────────────────────────────────────────────────────────────────────

def _get_repr_cache(adata) -> dict:
    if not hasattr(adata, "_repr_cache") or adata._repr_cache is None:
        adata._repr_cache = {}
    return adata._repr_cache


def _invalidate_repr_cache(adata) -> None:
    adata._repr_cache = {}


def _describe_storage(adata, sample_chunk: bool = True) -> dict:
    """Return a cached dict with file/format/density info."""
    cache = _get_repr_cache(adata)
    if "storage" in cache:
        return cache["storage"]

    info: dict[str, Any] = {}

    # File
    fname = getattr(adata, "filename", None)
    if fname is not None:
        info["filename"] = str(fname)
        try:
            info["file_size_mb"] = os.path.getsize(str(fname)) / 1024 ** 2
        except Exception:
            info["file_size_mb"] = None
    else:
        info["filename"] = None
        info["file_size_mb"] = None

    # X format — sample a tiny chunk to detect
    X = adata._X
    info["x_class"] = type(X).__name__
    info["x_dtype"] = None
    info["x_format"] = None
    info["density"] = None
    info["chunk_mb"] = None

    if sample_chunk:
        try:
            # Drill to the lowest-level BackedArray for density/format info.
            # Don't use _SubsetBackedArray.chunked() because it buffers.
            bottom = X
            while hasattr(bottom, "_parent") and bottom._parent is not None:
                bottom = bottom._parent
            target_chunk = min(1000, bottom.shape[0])
            for s, e, chunk_sample in bottom.chunked(target_chunk):
                info["x_dtype"] = str(chunk_sample.dtype)
                info["x_format"] = (
                    type(chunk_sample).__name__
                    if issparse(chunk_sample)
                    else "ndarray"
                )
                if issparse(chunk_sample):
                    total = chunk_sample.shape[0] * chunk_sample.shape[1]
                    info["density"] = chunk_sample.nnz / max(total, 1)
                    info["chunk_mb"] = (
                        chunk_sample.data.nbytes
                        + chunk_sample.indices.nbytes
                        + chunk_sample.indptr.nbytes
                    ) / 1024 ** 2
                else:
                    info["density"] = 1.0
                    info["chunk_mb"] = chunk_sample.nbytes / 1024 ** 2
                info["sample_chunk_rows"] = chunk_sample.shape[0]
                break
        except Exception:
            pass

    cache["storage"] = info
    return info


def _describe_chain(X) -> list[dict]:
    """Walk the transform chain and describe each node."""
    chain = []
    node = X
    while node is not None:
        cls = type(node).__name__
        desc: dict[str, Any] = {"class": cls, "shape": tuple(node.shape)}

        if cls == "_SubsetBackedArray":
            obs_idx = getattr(node, "_obs_idx", None)
            var_idx = getattr(node, "_var_idx", None)
            subset_parts = []
            if obs_idx is not None:
                subset_parts.append(f"obs: {len(obs_idx)}")
            if var_idx is not None:
                subset_parts.append(f"var: {len(var_idx)}")
            desc["tag"] = "subset"
            desc["detail"] = ", ".join(subset_parts) if subset_parts else "–"
        elif cls == "TransformedBackedArray":
            parts = []
            if getattr(node, "_norm_factors", None) is not None:
                parts.append("normalize")
            if getattr(node, "_apply_log1p", False):
                parts.append("log1p")
            desc["tag"] = "transform"
            desc["detail"] = " · ".join(parts) if parts else "identity"
        elif cls == "ScaledBackedArray":
            mx = getattr(node, "_max_value", None)
            desc["tag"] = "scale"
            clip = f", clip=±{mx}" if mx is not None else ""
            desc["detail"] = f"z-score (μ,σ stored{clip})"
        elif cls == "BackedArray":
            desc["tag"] = "backed"
            if getattr(node, "_is_rs", False):
                desc["detail"] = "Rust (anndata-rs)"
            else:
                desc["detail"] = "in-memory"
        else:
            desc["tag"] = "other"
            desc["detail"] = ""

        chain.append(desc)
        node = getattr(node, "_parent", None)

    return list(reversed(chain))  # bottom (raw) → top (current)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _fmt_shape(n_obs: int, n_vars: int) -> str:
    return f"{n_obs:,} × {n_vars:,}"


def _preview_keys(keys: list, max_n: int = 5, sep: str = " · ") -> str:
    if not keys:
        return "–"
    keys_str = [str(k) for k in keys[:max_n]]
    preview = sep.join(keys_str)
    if len(keys) > max_n:
        preview += f"  +{len(keys) - max_n}"
    return preview


def _fmt_mb(x: float | None) -> str:
    if x is None:
        return "?"
    return f"{x:.1f} MB"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "?"
    return f"{x*100:.1f}%"


# ──────────────────────────────────────────────────────────────────────
# Plain text formatter
# ──────────────────────────────────────────────────────────────────────

def _format_text(adata) -> str:
    storage = _describe_storage(adata)
    chain = _describe_chain(adata._X)

    # Header
    if len(chain) == 1:
        tag = "out-of-memory · backed"
    elif len(chain) > 1:
        tag = "lazy · backed"
    else:
        tag = "backed"
    header = f"AnnDataOOM{' ':>38}[Rust · {tag}]"

    dims = f"Dimensions:  n_obs: {adata.n_obs:,}    n_vars: {adata.n_vars:,}"

    lines = [header, dims, ""]

    # Storage table
    table_rows = []
    if storage.get("filename"):
        fn = os.path.basename(storage["filename"])
        sz = storage.get("file_size_mb")
        val = fn + (f"  ({sz:.1f} MB on disk)" if sz else "")
        table_rows.append(("File", val))

    x_class = storage.get("x_class", "?")
    x_format = storage.get("x_format", "?")
    x_dtype = storage.get("x_dtype", "?")
    density = storage.get("density")
    if x_class == "BackedArray":
        x_line = f"{x_format} · {x_dtype}"
        if density is not None and density < 1.0:
            x_line += f" · {_fmt_pct(density)} density"
    else:
        x_line = f"{x_class} · {x_dtype or '?'}"
    table_rows.append(("X", x_line))

    chunk_mb = storage.get("chunk_mb")
    sample_rows = storage.get("sample_chunk_rows", 1000)
    if chunk_mb is not None:
        table_rows.append(
            ("Chunk I/O", f"~{chunk_mb:.1f} MB per {sample_rows:,}-row chunk")
        )

    if table_rows:
        label_w = max(len(r[0]) for r in table_rows)
        val_w = max(len(r[1]) for r in table_rows)
        width = label_w + val_w + 5
        lines.append("┌" + "─" * (label_w + 2) + "┬" + "─" * (val_w + 2) + "┐")
        for label, val in table_rows:
            lines.append(f"│ {label:<{label_w}} │ {val:<{val_w}} │")
        lines.append("└" + "─" * (label_w + 2) + "┴" + "─" * (val_w + 2) + "┘")
        lines.append("")

    # Metadata sections
    sections = [
        ("obs", list(adata.obs.columns)),
        ("var", list(adata.var.columns)),
        ("obsm", list(adata.obsm.keys())),
        ("varm", list(adata.varm.keys())),
        ("obsp", list(adata.obsp.keys())),
        ("varp", list(adata.varp.keys())),
        ("layers", list(adata.layers.keys()) if hasattr(adata.layers, "keys") else []),
    ]
    for name, keys in sections:
        n = len(keys) if keys else 0
        count_str = f"({n})" if n else "(–)"
        preview = _preview_keys(keys) if keys else ""
        lines.append(f"▸ {name:<7s} {count_str:<5s}  {preview}")

    # raw
    if adata.raw is not None:
        raw_shape = adata.raw.shape
        lines.append(
            f"▸ {'raw':<7s}       {raw_shape[0]:,} × {raw_shape[1]:,}  (pre-subset)"
        )
    else:
        lines.append(f"▸ {'raw':<7s} (–)")

    # Transform chain (only if > 1 node)
    if len(chain) > 1:
        lines.append("")
        lines.append(f"Transform chain ({len(chain)} nodes):")
        cls_w = max(len(n["class"]) for n in chain)
        shp_w = max(len(_fmt_shape(*n["shape"])) for n in chain)
        for i, node in enumerate(chain):
            shape_str = _fmt_shape(*node["shape"])
            detail = node.get("detail", "")
            lines.append(
                f"  [{i}] {node['class']:<{cls_w}}  {shape_str:>{shp_w}}  "
                f"{node['tag']}"
                + (f" · {detail}" if detail else "")
            )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# HTML formatter for Jupyter
# ──────────────────────────────────────────────────────────────────────

_HTML_STYLE = """
<style>
.adoom {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 900px; color: var(--jp-ui-font-color0, #212121);
}
.adoom-header {
    display: flex; align-items: baseline; justify-content: space-between;
    padding: 8px 12px; background: var(--jp-layout-color1, #f5f5f5);
    border-left: 4px solid #E74C3C; margin-bottom: 8px;
}
.adoom-title { font-size: 1.1em; font-weight: 600; }
.adoom-tag {
    font-size: 0.85em; color: var(--jp-ui-font-color2, #616161);
    padding: 2px 8px; background: var(--jp-layout-color2, #eeeeee);
    border-radius: 3px;
}
.adoom-dims {
    padding: 4px 12px 8px 12px; font-size: 0.95em;
}
.adoom-dims strong { font-weight: 600; color: #E74C3C; }
.adoom-section { margin: 4px 0; }
.adoom-section summary {
    cursor: pointer; padding: 4px 12px; user-select: none;
    font-size: 0.9em;
}
.adoom-section summary:hover { background: var(--jp-layout-color1, #f5f5f5); }
.adoom-section[open] > summary { font-weight: 600; }
.adoom-section.empty > summary {
    color: var(--jp-ui-font-color3, #9e9e9e); cursor: default;
    pointer-events: none;
}
.adoom-section.empty > summary::-webkit-details-marker { display: none; }
.adoom-section.empty > summary::marker { content: ""; }
.adoom-table {
    border-collapse: collapse; margin: 4px 12px; font-size: 0.85em;
}
.adoom-table td, .adoom-table th {
    padding: 3px 10px; text-align: left;
    border-bottom: 1px solid var(--jp-border-color2, #e0e0e0);
}
.adoom-table td:first-child {
    font-weight: 500; color: var(--jp-ui-font-color1, #424242);
}
.adoom-preview {
    color: var(--jp-ui-font-color2, #616161);
    font-size: 0.85em; margin-left: 6px;
}
.adoom-count {
    color: var(--jp-ui-font-color2, #616161); margin-left: 4px;
}
.adoom-chain {
    margin: 6px 12px; font-family: Menlo, monospace; font-size: 0.8em;
}
.adoom-chain-row {
    padding: 2px 0; display: flex; gap: 12px;
}
.adoom-chain-tag {
    padding: 1px 6px; border-radius: 3px; font-weight: 600;
    min-width: 60px; text-align: center;
}
.tag-backed { background: #f5f5f5; color: #616161; }
.tag-subset { background: #fff3cd; color: #856404; }
.tag-transform { background: #d1ecf1; color: #0c5460; }
.tag-scale { background: #f8d7da; color: #721c24; }
</style>
"""


def _escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _format_html(adata) -> str:
    storage = _describe_storage(adata)
    chain = _describe_chain(adata._X)

    if len(chain) > 1:
        tag = "Rust · lazy · backed"
    else:
        tag = "Rust · out-of-memory · backed"

    parts = [_HTML_STYLE, '<div class="adoom">']
    parts.append(
        f'<div class="adoom-header">'
        f'<span class="adoom-title">AnnDataOOM</span>'
        f'<span class="adoom-tag">{_escape(tag)}</span>'
        f'</div>'
    )
    parts.append(
        f'<div class="adoom-dims">'
        f'Dimensions: <strong>{adata.n_obs:,}</strong> cells '
        f'× <strong>{adata.n_vars:,}</strong> genes</div>'
    )

    # Storage section
    rows = []
    if storage.get("filename"):
        fn = storage["filename"]
        sz = storage.get("file_size_mb")
        rows.append(
            ("File", f"<code>{_escape(os.path.basename(fn))}</code>"
             + (f" <span class='adoom-count'>({sz:.1f} MB)</span>" if sz else ""))
        )
    x_class = storage.get("x_class", "?")
    x_format = storage.get("x_format", "?")
    x_dtype = storage.get("x_dtype", "?")
    density = storage.get("density")
    x_line = f"<code>{_escape(x_format)}</code> · {_escape(x_dtype or '?')}"
    if density is not None and density < 1.0:
        x_line += f" · {_fmt_pct(density)} density"
    rows.append(("X format", x_line))
    chunk_mb = storage.get("chunk_mb")
    if chunk_mb is not None:
        rows.append(
            ("Chunk I/O",
             f"~{chunk_mb:.1f} MB per {storage.get('sample_chunk_rows', 1000):,}-row chunk")
        )

    parts.append(
        '<details class="adoom-section" open>'
        '<summary>Storage</summary>'
        '<table class="adoom-table">'
    )
    for k, v in rows:
        parts.append(f'<tr><td>{_escape(k)}</td><td>{v}</td></tr>')
    parts.append('</table></details>')

    # Metadata sections
    meta_sections = [
        ("obs", list(adata.obs.columns)),
        ("var", list(adata.var.columns)),
        ("obsm", list(adata.obsm.keys())),
        ("varm", list(adata.varm.keys())),
        ("obsp", list(adata.obsp.keys())),
        ("varp", list(adata.varp.keys())),
        ("layers", list(adata.layers.keys()) if hasattr(adata.layers, "keys") else []),
    ]
    for name, keys in meta_sections:
        n = len(keys)
        empty_cls = " empty" if n == 0 else ""
        preview = (
            f'<span class="adoom-preview">{_escape(_preview_keys(keys, 6))}</span>'
            if n > 0 else ""
        )
        count_str = f'<span class="adoom-count">({n})</span>' if n else '<span class="adoom-count">(–)</span>'
        parts.append(f'<details class="adoom-section{empty_cls}">')
        parts.append(f'<summary><strong>{name}</strong>{count_str}{preview}</summary>')
        if n > 0:
            # Table of columns (for obs/var) or list (for obsm/etc)
            if name in ("obs", "var"):
                df = getattr(adata, name)
                parts.append('<table class="adoom-table">')
                parts.append('<tr><th>name</th><th>dtype</th><th>preview</th></tr>')
                for col in keys[:20]:
                    try:
                        series = df[col]
                        dtype = str(series.dtype)
                        try:
                            uniq = series.unique()
                            if len(uniq) > 3:
                                prev = ", ".join(str(u) for u in uniq[:3]) + f", ... ({len(uniq)} unique)"
                            else:
                                prev = ", ".join(str(u) for u in uniq)
                        except Exception:
                            prev = ""
                        parts.append(
                            f'<tr><td><code>{_escape(col)}</code></td>'
                            f'<td>{_escape(dtype)}</td>'
                            f'<td><span class="adoom-preview">{_escape(prev)}</span></td></tr>'
                        )
                    except Exception:
                        pass
                if len(keys) > 20:
                    parts.append(f'<tr><td colspan="3"><em>+{len(keys)-20} more</em></td></tr>')
                parts.append('</table>')
            else:
                parts.append('<table class="adoom-table">')
                for k in keys:
                    try:
                        v = getattr(adata, name)[k]
                        shape_str = getattr(v, "shape", ("?",))
                        dtype_str = str(getattr(v, "dtype", "?"))
                        parts.append(
                            f'<tr><td><code>{_escape(k)}</code></td>'
                            f'<td>{shape_str}</td>'
                            f'<td>{_escape(dtype_str)}</td></tr>'
                        )
                    except Exception:
                        parts.append(f'<tr><td><code>{_escape(k)}</code></td><td colspan="2">–</td></tr>')
                parts.append('</table>')
        parts.append('</details>')

    # raw
    if adata.raw is not None:
        raw_shape = adata.raw.shape
        parts.append(
            '<details class="adoom-section">'
            '<summary><strong>raw</strong>'
            f'<span class="adoom-count">{raw_shape[0]:,} × {raw_shape[1]:,} (pre-subset)</span>'
            '</summary></details>'
        )
    else:
        parts.append(
            '<details class="adoom-section empty">'
            '<summary><strong>raw</strong><span class="adoom-count">(–)</span></summary>'
            '</details>'
        )

    # Transform chain
    if len(chain) > 1:
        parts.append('<details class="adoom-section" open>')
        parts.append(
            f'<summary><strong>Transform chain</strong>'
            f'<span class="adoom-count">({len(chain)} nodes)</span></summary>'
        )
        parts.append('<div class="adoom-chain">')
        for i, node in enumerate(chain):
            shape_str = _fmt_shape(*node["shape"])
            detail = node.get("detail", "")
            parts.append(
                f'<div class="adoom-chain-row">'
                f'<span>[{i}]</span>'
                f'<span class="adoom-chain-tag tag-{node["tag"]}">{node["tag"]}</span>'
                f'<span><code>{_escape(node["class"])}</code></span>'
                f'<span>{_escape(shape_str)}</span>'
                f'<span class="adoom-preview">{_escape(detail)}</span>'
                f'</div>'
            )
        parts.append('</div></details>')

    parts.append('</div>')
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────
# ov.read() welcome banner
# ──────────────────────────────────────────────────────────────────────

def _format_read_message(filename: str, size_mb: float | None, elapsed: float,
                         ram_delta_mb: float | None = None) -> str:
    """One-off welcome message for ov.read(..., backend='rust')."""
    lines = []
    lines.append(f"📂 Reading with anndata-rs (Rust · out-of-memory)")
    lines.append(f"   {filename}" + (f"  ({size_mb:.1f} MB)" if size_mb else ""))
    extra = f" · +{ram_delta_mb:.0f} MB RAM" if ram_delta_mb is not None else ""
    lines.append(f"   ✓ Loaded in {elapsed:.2f}s{extra}")
    lines.append("")
    lines.append(
        "💡 Data stays on disk. Use ov.pp.* for chunked processing."
    )
    lines.append(
        "   adata.close() when done · adata.to_adata() to materialise"
    )
    lines.append("")
    return "\n".join(lines)
