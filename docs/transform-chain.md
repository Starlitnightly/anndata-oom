# The Transform Chain

`anndataoom`'s central design idea is that **`adata.X` is not a matrix — it is
a linked list of lazy operators**. Every preprocessing step (subset, normalize,
log1p, scale, HVG filter, …) appends a new node; each node records *what*
transformation to apply, not the transformed data itself.

Actual I/O happens only when a consumer (PCA, `write`, plotting) pulls data
through the chain. One read from disk, all pending transforms applied in a
single streaming pass, nothing full-sized held in memory.

This document walks through the design in four diagrams.

---

## 1. What a single node looks like

```
┌─────────────────────────────────────────────┐
│  TransformedBackedArray                     │
│  ─────────────────────                      │
│  shape        : (1_001_288, 45_676)         │
│  _parent      : ──────────────────┐         │
│  _norm_factors: [0.83, 1.12, ...] │         │
│  _apply_log1p : True              │         │
│                                   │         │
│  def _read_rows(start, end):      │         │
│      raw = self._parent._read_rows│(s, e)   │
│      return log1p(raw / norm[s:e])│         │
└───────────────────────────────────┼─────────┘
                                    │
                                    ▼ downstream
```

Every node carries three things:

1. A `_parent` pointer to the node below it.
2. The parameters for its own transform (e.g. normalization factors, a log1p
   flag, a column index vector).
3. Implementations of `_read_rows(start, end)` / `_read_row_indices(indices)`
   / `chunked(chunk_size)` that delegate to `_parent` and then apply the
   transform to the returned chunk.

The unified read interface means every node in the chain speaks the same
protocol — layers can be added or removed freely without the downstream code
knowing or caring.

### Node catalog

| Class                    | Role                                   | Created by                    |
|--------------------------|----------------------------------------|-------------------------------|
| `BackedArray`            | Chain root; real I/O through Rust      | `oom.read()`                  |
| `_SubsetBackedArray`     | Records obs / var index selection      | `adata[mask]`, HVG subset     |
| `TransformedBackedArray` | normalize + log1p parameters           | `chunked_normalize_total`, `chunked_log1p` |
| `ScaledBackedArray`      | μ, σ, optional clip for z-score        | `chunked_scale`               |

---

## 2. A typical chain after a full pipeline

After `qc → normalize → log1p → scale → HVG`, `adata.X` looks like:

```
adata.X  ━━━━━━━━━━━━━━━━━┓
                          ┃  what the user sees
                          ▼
 ┌────────────────────────────────┐  ← [4] outermost
 │  _SubsetBackedArray            │
 │  37,715 × 2,000                │       HVG column select
 │  _var_idx = [hvg indices]      │
 │  _parent ──────────────┐       │
 └────────────────────────┼───────┘
                          ▼
 ┌────────────────────────────────┐  ← [3]
 │  ScaledBackedArray             │
 │  37,715 × 45,676               │       z-score
 │  _mean, _std, _max_value       │
 │  _parent ──────────────┐       │
 └────────────────────────┼───────┘
                          ▼
 ┌────────────────────────────────┐  ← [2]
 │  TransformedBackedArray        │
 │  37,715 × 45,676               │       normalize + log1p
 │  _norm_factors, _apply_log1p   │
 │  _parent ──────────────┐       │
 └────────────────────────┼───────┘
                          ▼
 ┌────────────────────────────────┐  ← [1]
 │  _SubsetBackedArray            │
 │  37,715 × 45,676               │       QC row filter
 │  _obs_idx = [passing cells]    │
 │  _parent ──────────────┐       │
 └────────────────────────┼───────┘
                          ▼
 ┌────────────────────────────────┐  ← [0] root
 │  BackedArray                   │
 │  1,001,288 × 45,676            │       HDF5 file handle
 │  _elem  = PyArrayElem ─────────┼──► data.h5ad on disk
 │  _parent = None                │
 └────────────────────────────────┘

Memory footprint of the chain: a few MB (indices + normalization vectors).
Bytes read from disk so far:     zero.
```

The HTML and text `__repr__` render this chain directly — `_describe_chain`
walks up the `_parent` links, collects each node, and reverses the list so
index `[0]` is the data source and `[-1]` is the current `adata.X`.

---

## 3. How data flows when something pulls a chunk

When `chunked_pca` (or any other consumer) calls `X.chunked(1000)`, requests
flow *downward* and transformed chunks flow *upward*:

```
  user call          Python stack          Rust / disk
  ─────────          ─────────             ─────────

  X.chunked(1000)
       │
       ▼
  ┏━━━━━━━━━━━━━━━┓
  ┃ [4] subset    ┃   "give me 1000 rows of HVG-filtered data"
  ┃    var_idx    ┃─────────────────────┐
  ┗━━━━━━━━━━━━━━━┛                     │
         ▲                               │
         │  4️⃣ keep HVG columns         ▼
         │                       ┏━━━━━━━━━━━━━━━┓
         │                       ┃ [3] scale     ┃   "1000 z-scored rows"
         │                       ┗━━━━━━━━━━━━━━━┛─┐
         │                              ▲           │
         │                              │  3️⃣ (x-μ)/σ
         │                              │           ▼
         │                              │   ┏━━━━━━━━━━━━━━━┓
         │                              │   ┃ [2] transform ┃
         │                              │   ┗━━━━━━━━━━━━━━━┛─┐
         │                              │        ▲             │
         │                              │        │  2️⃣ log1p   │
         │                              │        │     + norm  ▼
         │                              │        │     ┏━━━━━━━━━━━━━━━┓
         │                              │        │     ┃ [1] subset    ┃
         │                              │        │     ┗━━━━━━━━━━━━━━━┛─┐
         │                              │        │          ▲             │
         │                              │        │          │  1️⃣ pick    │
         │                              │        │          │    rows     ▼
         │                              │        │          │     ┏━━━━━━━━━━━━━━┓
         │                              │        │          │     ┃ [0] backed   ┃
         │                              │        │          │     ┗━━━━━━━━━━━━━━┛
         │                              │        │          │           │
         │                              │        │          │           │ 0️⃣ real read
         │                              │        │          │           ▼
         │                              │        │          │      ┌──────────┐
         │                              │        │          │      │ data.h5ad│
         │                              │        │          │      └──────────┘
         │                              │        │          │           │
         │                              │        │          │◄──────────┘ raw chunk
         │                              │        │          │
         │                              │        │◄─────────┘ row-filtered
         │                              │        │
         │                              │◄───────┘ normalize + log
         │                              │
         │                              │◄────── z-scored
         │                              │
         │◄─────────────────────────────┘ HVG columns only
         │
  ◄──────┘ final chunk: (1000, 2000)

One disk read  →  every transform applied in place on the chunk
               →  no full-sized intermediate ever allocated
```

Peak working-set memory is `chunk_size × n_vars`, independent of `n_obs`.

---

## 4. Chain flattening keeps it O(1)

A naive implementation would stack wrappers on top of wrappers every time
the user writes `adata[a][b][c]`. Every subsequent read would have to walk
all of them:

```
  naïve stacking                    anndataoom's flattening
  ───────────                        ───────────

  ┌──────────┐                      ┌──────────┐
  │ subset 3 │                      │ subset   │  _obs_idx =
  │ _parent ─┼─┐                    │          │    mask1[mask2][mask3]
  └──────────┘ │                    │ _parent ─┼─┐
               ▼                    └──────────┘ │
      ┌──────────┐                                │
      │ subset 2 │                                │
      │ _parent ─┼─┐                              │
      └──────────┘ │                              │
                   ▼                              │
          ┌──────────┐                            │
          │ subset 1 │                            │
          │ _parent ─┼─┐                          │
          └──────────┘ │                          │
                       ▼                          ▼
              ┌──────────┐              ┌──────────┐
              │  backed  │              │  backed  │
              └──────────┘              └──────────┘

   3 Python frames per read          1 frame, indices pre-composed
```

`_SubsetBackedArray.__init__` handles this inline: if its `parent` is already
a `_SubsetBackedArray`, it composes the obs/var indices and rewires `_parent`
to the grandparent. A chain of `N` subsets collapses to a single subset node,
so read overhead stays O(1) regardless of subsetting depth.

---

## Why this design

| Goal                             | How the chain design achieves it                        |
|----------------------------------|---------------------------------------------------------|
| Peak RAM independent of n_obs    | Nothing full-sized is ever materialized — only parameters |
| Full `anndata.AnnData` API       | Every node implements the same read protocol             |
| Zero-cost subsetting / copying   | Adding a node is a few attribute assignments              |
| `.raw` is nearly free             | `.raw` just pins a node from before HVG subsetting        |
| Single-pass fused preprocessing   | Consumers call `.chunked()` once; all transforms fuse    |
| Transparent to scanpy / omicverse | `isinstance(oom, AnnData) → True` via ABC registration   |

The chain is what lets `anndataoom` look exactly like `anndata.AnnData` to
downstream code while having fundamentally different memory semantics.

---

## Seeing the chain at runtime

The `__repr__` prints the chain whenever it has more than one node:

```text
Transform chain (5 nodes):
  [0] backed     BackedArray           1,001,288 × 45,676   Rust (anndata-rs)
  [1] subset     _SubsetBackedArray       37,715 × 45,676   obs: 37715
  [2] transform  TransformedBackedArray   37,715 × 45,676   normalize · log1p
  [3] scale      ScaledBackedArray        37,715 × 45,676   z-score (μ,σ stored, clip=±10)
  [4] subset     _SubsetBackedArray       37,715 ×  2,000   var: 2000
```

The Jupyter HTML repr renders the same information with per-node tags and a
collapsible section. If you ever want to know what operations are pending on
an `AnnDataOOM`, inspect the chain.
