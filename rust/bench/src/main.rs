use forecasting_ore_grade_variability_with_open_geochemistry_and_machine_learning_core::rolling_mean_std;

fn main() {
    let v: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.02).sin() * 10.0 + 0.5).collect();
    for _ in 0..2000 {
        let _ = rolling_mean_std(&v, 30);
    }
}
