"""Basic smoke tests for anndataoom."""
import tempfile
import numpy as np
import pytest


@pytest.fixture
def h5ad_file():
    """Create a small h5ad file for testing."""
    import anndata
    import os

    np.random.seed(42)
    X = np.random.poisson(2, size=(100, 50)).astype(np.float32)
    adata = anndata.AnnData(X=X)
    adata.obs.index = [f"cell_{i}" for i in range(100)]
    adata.var.index = [f"gene_{i}" for i in range(50)]
    adata.obs["batch"] = ["A"] * 50 + ["B"] * 50

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        path = f.name
    adata.write(path)
    yield path
    os.remove(path)


def test_import():
    import anndataoom as oom
    assert hasattr(oom, "read")
    assert hasattr(oom, "AnnDataOOM")
    assert hasattr(oom, "BackedArray")


def test_read(h5ad_file):
    import anndataoom as oom
    adata = oom.read(h5ad_file)
    assert adata.shape == (100, 50)
    assert list(adata.obs.columns) == ["batch"]
    adata.close()


def test_chunked_iteration(h5ad_file):
    import anndataoom as oom
    adata = oom.read(h5ad_file)
    total = 0
    for start, end, chunk in adata.X.chunked(30):
        total += chunk.shape[0]
        assert chunk.shape[1] == 50
    assert total == 100
    adata.close()


def test_subsetting(h5ad_file):
    import anndataoom as oom
    adata = oom.read(h5ad_file)
    sub = adata[0:10]
    assert sub.shape == (10, 50)
    sub_var = adata[:, 0:5]
    assert sub_var.shape == (100, 5)
    adata.close()


def test_obs_vector(h5ad_file):
    import anndataoom as oom
    adata = oom.read(h5ad_file)
    v = adata.obs_vector("gene_0")
    assert v.shape == (100,)
    adata.close()


def test_repr(h5ad_file):
    import anndataoom as oom
    adata = oom.read(h5ad_file)
    r = repr(adata)
    assert "AnnDataOOM" in r
    assert "100" in r
    assert "50" in r
    adata.close()
