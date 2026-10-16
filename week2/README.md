# Week 2: Classification Models and Data Preprocessing

Comparing three classifiers (Decision Tree, k-Nearest Neighbors, Naive Bayes) on three datasets, using one reusable scikit-learn pipeline.

**Workflow:** split data, preprocess inside a Pipeline, train a model, cross-validate, evaluate, visualise.

## Datasets
| Dataset | Source | Task |
|---|---|---|
| Titanic | seaborn | Predict survival (missing values + categorical features) |
| Wine | scikit-learn | 3-class wine type |
| Breast Cancer | scikit-learn | Malignant vs benign |

## What I covered
- **Preprocessing:** train/test split, data leakage, missing values (`SimpleImputer`), one-hot encoding, scaling (`StandardScaler`)
- **Models:** Decision Tree, kNN, Naive Bayes (GaussianNB)
- **Evaluation:** k-fold cross-validation, confusion matrices, accuracy / precision / recall / F1
- **Pipelines:** `Pipeline` and `ColumnTransformer`, so preprocessing is refitted on each training fold (no leakage)
- **Feature selection vs dimensionality reduction:** `SelectKBest` vs PCA on Wine

## Results

**5-fold cross-validation accuracy**

| Dataset | Naive Bayes | Decision Tree | kNN |
|---|---|---|---|
| Titanic | 0.777 | 0.790 | **0.804** |
| Wine | **0.966** | 0.894 | 0.949 |
| Breast Cancer | 0.928 | 0.921 | **0.965** |

**Test-set metrics** (one stratified 75/25 split, macro average)

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Titanic | Naive Bayes | 0.762 | 0.750 | 0.755 | 0.752 |
| Titanic | Decision Tree | 0.780 | 0.804 | 0.732 | 0.744 |
| Titanic | kNN | 0.794 | 0.786 | 0.772 | 0.777 |
| Wine | Naive Bayes | 0.978 | 0.979 | 0.981 | 0.980 |
| Wine | Decision Tree | 0.956 | 0.967 | 0.950 | 0.956 |
| Wine | kNN | 0.933 | 0.929 | 0.935 | 0.931 |
| Breast Cancer | Naive Bayes | 0.937 | 0.938 | 0.927 | 0.932 |
| Breast Cancer | Decision Tree | 0.944 | 0.940 | 0.940 | 0.940 |
| Breast Cancer | kNN | 0.979 | 0.984 | 0.972 | 0.977 |

Breast Cancer recall on the **malignant** class (the most important metric there): kNN 0.943, Decision Tree 0.925, Naive Bayes 0.887.

## Key findings
- **Titanic:** kNN is slightly ahead, but all three models are close because the data is noisy.
- **Wine:** Naive Bayes wins. The classes are well separated and the features are roughly Gaussian.
- **Breast Cancer:** scaled kNN wins, including the best malignant recall. Naive Bayes suffers from strongly correlated features.
- **Decision Tree:** easy to interpret, but it overfits and varies most between folds.
- PCA on Wine: the first two components keep 55.4% of the variance and still separate the three classes.

The test sets are small, so one wrong prediction moves accuracy by 0.5 to 2 points. Cross-validation numbers are more reliable than a single split.

## Project files
| File | Contents |
|---|---|
| `monday_titanic.ipynb` | Titanic preprocessing, split vs cross-validation |
| `tuesday_trees_knn.ipynb` | Decision Tree and kNN, overfitting, scaling |
| `wednesday_naive_bayes_pipelines.ipynb` | Naive Bayes and Pipelines |
| `thursday_feature_selection_comparison.ipynb` | SelectKBest, PCA, full comparison, confusion matrices |
| `results_table.csv` | Metrics table from the comparison notebook |
| `week2_notes.md` | Daily notes and the performance comparison summary |

## How to run
```
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
jupyter notebook
```
Open a notebook and choose Run All. Titanic is loaded from seaborn, so an internet connection is needed the first time.

## Tools
Python, pandas, scikit-learn, seaborn, matplotlib
