# Week 3: Unsupervised Learning and Hyperparameter Tuning

This week covers clustering (K-Means and Hierarchical), visualising clusters with PCA, and improving supervised models with GridSearchCV and RandomizedSearchCV.

## Contents

| File | What it does |
|------|--------------|
| `Week3_Monday_KMeans.ipynb` | K-Means on Mall Customers, Elbow method, Silhouette score, cluster profiles |
| `Week3_Tuesday_Hierarchical_PCA.ipynb` | Dendrogram, Agglomerative vs K-Means, four-feature clustering with PCA plot |
| `Week3_Wednesday_GridSearchCV.ipynb` | GridSearchCV on kNN, Decision Tree and Logistic Regression (Breast Cancer) |
| `Week3_Thursday_RandomizedSearchCV.ipynb` | RandomizedSearchCV vs GridSearchCV, tuning heatmap, results table |
| `Mall_Customers.csv` | Mall Customers dataset (200 rows) |
| `week3_notes.md` | Notes for the whole week |

## Datasets
- **Mall Customers** (CSV, Kaggle "Mall Customer Segmentation Data"): Age, Annual Income (k$), Spending Score (1-100), Gender.
- **Breast Cancer** (built into scikit-learn).

## Clustering results
- **K-Means** on scaled Annual Income and Spending Score: the Elbow bends at k = 5 and the Silhouette peaks at k = 5 (0.555), giving 5 customer segments.
- **Hierarchical (Ward)** with 5 clusters agrees closely with K-Means (silhouette 0.554 vs 0.555, Adjusted Rand Index 0.94).
- **Four features** (Age, Income, Spending, Gender) gave much lower silhouette scores (about 0.25 to 0.42). Gender dominated the splits (four of the six clusters were exactly 100% or 0% female), so the two-feature clustering was more useful. PCA's 2 components kept about 59.9% of the variance.

| Customers | Avg income (k$) | Avg spending | Segment |
|-----------|-----------------|--------------|---------|
| 81 | 55.3 | 49.5 | Average customers |
| 39 | 86.5 | 82.1 | High income, high spending |
| 22 | 25.7 | 79.4 | Low income, high spending |
| 35 | 88.2 | 17.1 | High income, low spending |
| 23 | 26.3 | 20.9 | Low income, low spending |

## Tuning results (Breast Cancer, 5-fold CV, held-out test set)

| Model | Default CV | Tuned CV | Default test | Tuned test |
|-------|-----------|----------|--------------|------------|
| kNN | 0.9626 | 0.9714 | 0.9561 | 0.9649 |
| Decision Tree | 0.9165 | 0.9363 | 0.9123 | 0.8772 |
| Logistic Regression | 0.9780 | 0.9824 | 0.9825 | 0.9737 |

- Tuning raised the CV score of all three models, but the test-set gains were small and mixed, and two tuned models scored lower on the test set.
- On a Decision Tree, **RandomizedSearchCV** (50 combinations) was about 98 times faster (1.3 s vs 124.5 s) than **GridSearchCV** (720 combinations), with a CV score within the fold-to-fold spread.

## How to run
1. Install: `pip install numpy pandas matplotlib seaborn scikit-learn scipy`
2. Keep `Mall_Customers.csv` in the same folder as the notebooks.
3. Open a notebook in Jupyter or VS Code and use Run All.

The Mall data file may name the gender column `Gender` or `Genre`; the notebooks rename it automatically.
