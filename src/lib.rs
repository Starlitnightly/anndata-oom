// anndataoom._backend — Python extension module exposing anndata-rs bindings.
//
// This re-exports pyanndata's public API as the `_backend` submodule
// of the `anndataoom` Python package. The heavy lifting (HDF5 I/O,
// chunked iteration, subsetting, etc.) all happens in Rust.

use pyanndata::*;
use pyo3::{prelude::*, pymodule, types::PyModule, PyResult};

#[pymodule]
fn _backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<AnnData>().unwrap();
    m.add_class::<AnnDataSet>().unwrap();
    m.add_class::<PyCompression>().unwrap();

    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(read_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_default_write_config, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_default_write_config, m)?)?;

    Ok(())
}
