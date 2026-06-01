# Forecasting Ore Grade Variability with Open Geochemistry and Machine Learning

Published: 2025-10-07
Medium: [https://medium.com/@kyle-t-jones/forecasting-ore-grade-variability-with-open-geochemistry-and-machine-learning-e9b08c45f9af](https://medium.com/@kyle-t-jones/forecasting-ore-grade-variability-with-open-geochemistry-and-machine-learning-e9b08c45f9af)

## Business context

Miners combine geochemical data with modern machine learning to predict grade, quantify risk, and optimize sampling strategies.

Drillholes give point samples. Mines need continuous grade maps. The gap between sparse measurements and dense predictions has traditionally been filled by geostatistical methods like Ordinary Kriging. But when you add machine learning to geochemical covariates, you unlock probabilistic forecasts that reveal not just where the gold is, but where your predictions are most uncertain --- critical intelligence for adaptive drilling and pit design.

This project uses gold grade predictions across Western Australia using three methods: Ordinary Kriging (traditional geostatistics), Gaussian Process Regression (probabilistic ML), and XGBoost (gradient boosting). The GPR model reveals prediction uncertainty, highlighting zones requiring additional sampling.



## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — rolling mean and std. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p forecasting_ore_grade_variability_with_open_geochemistry_and_machine_learning_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).