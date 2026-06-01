//! Rolling mean and std (same window semantics as pandas rolling).

pub fn rolling_mean_std(values: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = values.len();
    let w = window.max(1);
    let mut means = vec![0.0; n];
    let mut stds = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(w - 1);
        let slice = &values[start..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        means[i] = mean;
        stds[i] = var.sqrt();
    }
    (means, stds)
}
