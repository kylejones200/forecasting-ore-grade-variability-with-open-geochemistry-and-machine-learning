#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import rolling_mean_std  # noqa: E402

def main() -> None:
    v = np.ascontiguousarray(np.sin(np.arange(5000) * 0.02) * 10.0 + 0.5)
    window = 30
    t0 = time.perf_counter()
    for _ in range(200):
        rolling_mean_std(v, window)
    py_s = time.perf_counter() - t0
    try:
        import forecasting_ore_grade_variability_with_open_geochemistry_and_machine_learning_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(v, window, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    py_m, py_s = rolling_mean_std(v, window)
    rs_m, rs_s = rs.rolling_mean_std_py(v, window)
    np.testing.assert_allclose(py_m, np.asarray(rs_m), rtol=1e-10)
    np.testing.assert_allclose(py_s, np.asarray(rs_s), rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
