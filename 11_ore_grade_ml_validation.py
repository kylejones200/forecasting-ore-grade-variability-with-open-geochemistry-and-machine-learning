#!/usr/bin/env python3
"""
Validation script for Blog 11: Ore Grade Forecasting with ML
Tests all functions and verifies outputs.
"""

import importlib.util
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Import module with numeric prefix
spec = importlib.util.spec_from_file_location(
    "ore_grade_ml_production",
    "outputs/11_ore_grade_ml_production.py",
)
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)

# Import all functions
fetch_geochemical_data = production_module.fetch_geochemical_data
prepare_spatial_features = production_module.prepare_spatial_features
create_spatial_folds = production_module.create_spatial_folds
fit_variogram = production_module.fit_variogram
ordinary_kriging_predict = production_module.ordinary_kriging_predict
train_gaussian_process = production_module.train_gaussian_process
train_xgboost = production_module.train_xgboost
create_prediction_grid = production_module.create_prediction_grid
analyze_uncertainty_calibration = production_module.analyze_uncertainty_calibration
compare_methods = production_module.compare_methods




def _validate(condition: bool, message: str) -> None:
    """Assert a validation condition, raising ValueError with context on failure."""
    if not condition:
        raise ValueError(message)


def main():
    """Run validation tests for all functions."""
    logger.info("BLOG 11 VALIDATION - ORE GRADE FORECASTING WITH ML")
    logger.info()
    try:
        # Test 1: Data fetching
        logger.info("TEST 1: Fetching geochemical data...")
        df = fetch_geochemical_data()
        _validate(len(df) == 250, "Expected 250 samples")
        if "Au" not in df.columns:
            raise ValueError("Missing Au column")
        if "lithology" not in df.columns:
            raise ValueError("Missing lithology column")
        logger.info("✓ Data fetching successful\n")
        # Test 2: Spatial feature preparation
        logger.info("TEST 2: Preparing spatial features...")
        gdf = prepare_spatial_features(df)
        if "x" not in gdf.columns:
            raise ValueError("Missing x coordinate")
        if "y" not in gdf.columns:
            raise ValueError("Missing y coordinate")
        if "log_Au" not in gdf.columns:
            raise ValueError("Missing log_Au")
        logger.info("✓ Spatial features prepared\n")
        # Test 3: Spatial folds
        logger.info("TEST 3: Creating spatial cross-validation folds...")
        groups = create_spatial_folds(gdf)
        _validate(len(groups) == len(gdf), "Groups length mismatch")
        _validate(len(np.unique(groups)) >= 4, "Expected at least 4 folds")
        logger.info("✓ Spatial folds created\n")
        # Test 4: Variogram fitting
        logger.info("TEST 4: Fitting variogram...")
        V = fit_variogram(gdf)
        _validate(V.sill > 0, "Sill should be positive")
        _validate(V.range > 0, "Range should be positive")
        logger.info("✓ Variogram fitted\n")
        # Test 5: Ordinary Kriging
        logger.info("TEST 5: Performing Ordinary Kriging...")
        gx, gy, ok_ppm, ok_var = ordinary_kriging_predict(gdf, grid_resolution=50)
        _validate(ok_ppm.shape == (50, 50), "Unexpected grid shape")
        _validate(ok_ppm.min() >= 0, "Negative predictions found")
        logger.info("✓ Ordinary Kriging completed\n")
        # Test 6: Gaussian Process
        logger.info("TEST 6: Training Gaussian Process Regressor...")
        gp_model, gp_pred, gp_std, gpr_metrics = train_gaussian_process(gdf, groups)
        _validate(len(gp_pred) == len(gdf), "Prediction length mismatch")
        _validate(len(gp_std) == len(gdf), "Std length mismatch")
        _validate(gpr_metrics["mae"] > 0, "Invalid MAE")
        _validate(0.8 <= gpr_metrics["coverage"] <= 1.0, "Coverage out of range")
        logger.info("✓ Gaussian Process trained\n")
        # Test 7: XGBoost
        logger.info("TEST 7: Training XGBoost...")
        xgb_model, xgb_pred, xgb_metrics = train_xgboost(gdf, groups)
        _validate(len(xgb_pred) == len(gdf), "Prediction length mismatch")
        _validate(xgb_metrics["mae"] > 0, "Invalid MAE")
        logger.info("✓ XGBoost trained\n")
        # Test 8: Grid predictions
        logger.info("TEST 8: Creating prediction grid...")
        grid_results = create_prediction_grid(gdf, gp_model, xgb_model, resolution=50)
        if "gp_mean" not in grid_results:
            raise ValueError("Missing GPR mean")
        if "gp_std" not in grid_results:
            raise ValueError("Missing GPR std")
        if "xgb_pred" not in grid_results:
            raise ValueError("Missing XGB predictions")
        _validate(grid_results["gp_mean"].shape == (50, 50), "Grid shape mismatch")
        logger.info("✓ Prediction grid created\n")
        # Test 9: Calibration analysis
        logger.info("TEST 9: Analyzing uncertainty calibration...")
        calib_df = analyze_uncertainty_calibration(gdf["log_Au"].values, gp_pred, gp_std, n_bins=5)
        _validate(len(calib_df) >= 4, "Expected at least 4 calibration bins")
        if "predicted_std" not in calib_df.columns:
            raise ValueError("Missing predicted_std")
        if "actual_rmse" not in calib_df.columns:
            raise ValueError("Missing actual_rmse")
        logger.info("✓ Calibration analysis completed\n")
        # Test 10: Method comparison
        logger.info("TEST 10: Comparing methods...")
        compare_methods({}, gpr_metrics, xgb_metrics)
        logger.info("✓ Method comparison completed\n")
        logger.info("ALL VALIDATION TESTS PASSED!")
        logger.info()
        # Summary statistics
        logger.info("VALIDATION SUMMARY:")
        logger.info(f"  Total samples: {len(gdf)}")
        logger.info(
            f"  Spatial extent: {gdf['x'].max() - gdf['x'].min():.1f} × {gdf['y'].max() - gdf['y'].min():.1f} km"
        )
        logger.info(f"  Au range: {gdf['Au'].min():.3f} - {gdf['Au'].max():.3f} ppm")
        logger.info(f"  GPR MAE: {gpr_metrics['mae']:.3f}")
        logger.info(f"  GPR RMSE: {gpr_metrics['rmse']:.3f}")
        logger.info(f"  GPR Coverage: {gpr_metrics['coverage']:.1%}")
        logger.info(f"  XGB MAE: {xgb_metrics['mae']:.3f}")
        logger.info(f"  XGB RMSE: {xgb_metrics['rmse']:.3f}")
        logger.info(
            f"  XGB Improvement: {(1 - xgb_metrics['mae'] / gpr_metrics['mae']) * 100:.1f}%"
        )
        return True

    except Exception as e:
        logger.error(f"\n❌ VALIDATION FAILED: {str(e, exc_info=True)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
