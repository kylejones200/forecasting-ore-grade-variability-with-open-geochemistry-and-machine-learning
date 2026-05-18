# Forecasting Ore Grade Variability with Open Geochemistry and Machine Learning

Published: 2025-10-07
Medium: [https://medium.com/@kyle-t-jones/forecasting-ore-grade-variability-with-open-geochemistry-and-machine-learning-e9b08c45f9af](https://medium.com/@kyle-t-jones/forecasting-ore-grade-variability-with-open-geochemistry-and-machine-learning-e9b08c45f9af)

## Business context

Miners combine geochemical data with modern machine learning to predict grade, quantify risk, and optimize sampling strategies.

Drillholes give point samples. Mines need continuous grade maps. The gap between sparse measurements and dense predictions has traditionally been filled by geostatistical methods like Ordinary Kriging. But when you add machine learning to geochemical covariates, you unlock probabilistic forecasts that reveal not just where the gold is, but where your predictions are most uncertain --- critical intelligence for adaptive drilling and pit design.

This project uses gold grade predictions across Western Australia using three methods: Ordinary Kriging (traditional geostatistics), Gaussian Process Regression (probabilistic ML), and XGBoost (gradient boosting). The GPR model reveals prediction uncertainty, highlighting zones requiring additional sampling.

## About

Place the code for this article in this repository.
The original article export is saved as `article.md`.

## Files

Add your `.ipynb`, `.py`, `.yaml`, `.js`, `.ts`, or other project files here.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).