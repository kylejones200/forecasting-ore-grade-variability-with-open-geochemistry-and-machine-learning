use forecasting_ore_grade_variability_with_open_geochemistry_and_machine_learning_core::rolling_mean_std;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn rolling_mean_std_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (means, stds) = rolling_mean_std(values.as_slice()?, window);
    Ok((means.into_pyarray(py), stds.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (values, window, iterations=2_000))]
fn bench_kernel_py(
    values: PyReadonlyArray1<f64>,
    window: usize,
    iterations: usize,
) -> PyResult<f64> {
    let v = values.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = rolling_mean_std(&v, window);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn forecasting_ore_grade_variability_with_open_geochemistry_and_machine_learning_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_mean_std_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
