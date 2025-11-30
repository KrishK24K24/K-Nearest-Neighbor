# K-Nearest Neighbors (KNN) — Detailed README

This repository contains two focused Jupyter notebooks that demonstrate K-Nearest Neighbors for classification and regression, plus a small dataset used by the regression notebook. The README below explains what each notebook does, the exact steps taken in the code, evaluation details, and how to run everything locally.

Files
- KNN.ipynb — Classification notebook (sklearn KNeighborsClassifier)
- KNNRegressor.ipynb — Regression notebook (sklearn KNeighborsRegressor)
- height-weight.csv — Tiny dataset (used by KNNRegressor.ipynb)

Summary
- KNN.ipynb: builds a classification example from a synthetic dataset, trains a KNN classifier, inspects baseline performance and then tunes k using GridSearchCV. Visualizations and standard classification metrics are produced to explain model behavior.
- KNNRegressor.ipynb: demonstrates KNN regression on a real CSV of human measurements (height → weight). It shows preprocessing, training, evaluation with regression metrics, and plots predictions vs actuals to illustrate the effect of k.

Detailed explanation of what the code does

KNN.ipynb (classification) — step-by-step
1. Data generation / loading
   - Uses sklearn.datasets.make_classification to generate a synthetic 2D toy dataset suitable for visualizing decision boundaries and classification performance.
   - Optionally inspects the dataset (shape, class balance).

2. Preprocessing
   - Splits into training and test sets using train_test_split (typical random state set for reproducibility).
   - Applies scaling (StandardScaler) if included — scaling is recommended for distance-based models like KNN.

3. Baseline model training
   - Creates a KNeighborsClassifier with a baseline n_neighbors (e.g., 5 or 10) and fits on the training set.
   - Predicts on the test set.

4. Evaluation
   - Computes accuracy_score to get a simple performance estimate.
   - Produces confusion_matrix and classification_report (precision, recall, f1-score) for per-class performance insight.
   - Optionally plots decision boundaries to show how neighbors determine class regions.

5. Hyperparameter tuning
   - Uses GridSearchCV to search over a range of n_neighbors values (usually a list such as 1..49 or a smaller subset).
   - Uses cross-validation to compare candidates and selects the best k based on validation performance.
   - Fits the best estimator, evaluates on the test set, and compares results to the baseline.

6. What you should see / learn
   - How changing k affects bias/variance (small k → noisy boundaries, large k → smoother boundaries).
   - How scaling affects performance.
   - How to use GridSearchCV to reliably pick k.

KNNRegressor.ipynb (regression) — step-by-step
1. Data loading
   - Reads height-weight.csv into a pandas DataFrame.
   - Confirms column names and inspects basic statistics (head(), describe()).

2. Feature selection and preprocessing
   - Defines feature(s) X (Height) and target y (Weight). If column names differ, the notebook cell is prepared to be edited accordingly.
   - Splits data into training and test sets with train_test_split.
   - Applies scaling (StandardScaler) to features — important for KNN distance calculation. For a single numeric feature scaling may be optional but is shown for completeness.

3. Model training
   - Creates a KNeighborsRegressor with a chosen n_neighbors (example: 5 or 10) and fits on training data.
   - Predicts on the test set.

4. Evaluation
   - Computes regression metrics: Mean Squared Error (MSE), Root MSE, and R² (coefficient of determination).
   - Visualizes predicted vs actual target values (scatter plot) and errors (residual plot).

5. Hyperparameter exploration
   - Optionally iterates over different k values and plots MSE/R² vs k to show how k affects predictive performance.
   - Selects a k that gives a trade-off between low error and stable predictions.

6. What you should see / learn
   - How K affects smoothness of predictions.
   - Why scaling matters when features have different ranges.
   - How to interpret MSE and R² for KNN regressors.

Dataset: height-weight.csv
- Tiny CSV of human measurements (Height, Weight). Inspect the file before running the regression notebook to confirm exact column names. The notebook includes instructions/comments for adjusting column names if they differ.

Dependencies
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter (or jupyterlab)

Quick start — run locally
1. Create a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate       # Windows PowerShell

2. Install dependencies
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter

3. Start Jupyter and open the notebooks
   jupyter notebook
   # or
   jupyter lab

4. Open KNN.ipynb and KNNRegressor.ipynb and run cells from top to bottom.
   - For KNNRegressor.ipynb: confirm the CSV column names in the first cell that loads the data; update if necessary.
