# KNN — K-Nearest Neighbors notebooks

This folder contains experiments and notebooks that demonstrate K-Nearest Neighbors for both classification and regression.

Files
- `height-weight.csv` — CSV dataset containing height and weight measurements used for regression experiments. Columns typically include `Height` and `Weight` (check the file to confirm exact column names).
- `KNN.ipynb` — Classification notebook. High-level steps performed:
  - Demonstrates generating a synthetic classification dataset using `sklearn.datasets.make_classification` (used in the notebook to create a demo dataset).
  - Splits data into training and test sets with `train_test_split`.
  - Trains a `KNeighborsClassifier` (example shown with `n_neighbors=10`).
  - Evaluates predictions with `accuracy_score`, `confusion_matrix`, and `classification_report`.
  - Uses `GridSearchCV` to tune the `n_neighbors` hyperparameter (search grid uses values from 1 to 49) and applies the best estimator to the test set.

- `KNNRegressor.ipynb` — Regression notebook. High-level steps typically include:
  - Loads `height-weight.csv` and prepares features/target for predicting weight (or height) using KNN regression.
  - Preprocesses data as needed (scaling or train/test split) and trains a `KNeighborsRegressor`.
  - Evaluates model performance using regression metrics (for example, Mean Squared Error, R²) and visualizes predictions vs actual values.

Data summary
- `height-weight.csv` — Small tabular dataset of human measurements. Typical use in the notebooks:
  - Feature(s): `Height` (numeric)
  - Target: `Weight` (numeric)
  - Suitable for simple regression examples and for demonstrating KNN regression behavior and the effect of `k`.

How the code in `KNN.ipynb` works (concise)
- Creates or loads a dataset (the notebook uses a synthetic dataset via `make_classification` for the classification demonstration).
- Splits into training and test sets.
- Trains an initial KNN classifier and reports baseline metrics.
- Runs `GridSearchCV` over `n_neighbors` to find the best `k`, then re-evaluates the best model on the test set.

Dependencies (minimum)
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`

Quick run
1. Install dependencies (PowerShell):

```powershell
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

2. Start Jupyter and open the notebook:

```powershell
jupyter notebook
# or
jupyter lab
```

Notes & suggestions
- Inspect `height-weight.csv` to confirm column names before running `KNNRegressor.ipynb`. If columns are not named `Height`/`Weight`, update the notebook cell that reads the CSV.
- If you publish the repository, consider adding a `.gitignore` (exclude `.ipynb_checkpoints`, virtual env folders, large datasets) and using `requirements.txt` for reproducibility.

If you want, I can: create a `.gitignore` for this project, add a folder-level `requirements.txt`, or commit & push this README for you.
