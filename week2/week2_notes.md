# Week 2 Notes: Core Classification Models and Data Preprocessing

**Workflow for the whole week:** split data, preprocess inside a Pipeline, train a model, cross-validate, evaluate, visualise.

---

## Monday: Data Preprocessing, Train/Test Split, Cross-Validation

### Train/test split and data leakage
- We split data into a **training set** (the model learns from it) and a **test set** (used only to check performance on unseen data).
- **Data leakage** means information from the test set influences training. Example: filling missing ages with the median of the *whole* dataset lets the test data affect the imputer.
- Rule: fit imputers, scalers and encoders **only on training data**, then only *transform* the test data.
- `stratify=y` keeps the class ratio (e.g. survived vs not) the same in train and test.

### Cross-validation (k-fold CV)
- Data is split into k parts. The model is trained k times, each time using a different part as the test fold and the rest as training. The k scores are averaged.
- Why it is more reliable than one split: a single split depends on luck (which rows landed in the test set). CV averages several splits, and the standard deviation shows how stable the model is.

### Handling missing data
| Option | When to use |
|---|---|
| Drop the column | Mostly missing (e.g. Titanic `deck`: 688 of 891 missing) |
| Drop rows | Very few rows are missing and data is plentiful |
| Fill with mean | Numeric, roughly symmetric |
| Fill with median | Numeric with outliers or skew (e.g. `age`, `fare`) |
| Fill with most frequent | Categorical (e.g. `embarked`) |

### Encoding categorical features
- **One-hot encoding:** one new 0/1 column per category. Use when categories have no order (`sex`, `embarked`).
- **Ordinal encoding:** categories become numbers 0, 1, 2 in a meaningful order. Use when order exists (low < medium < high).
- One-hot on an ordered feature loses the order; ordinal on an unordered feature creates a fake order.

### Feature scaling
- **StandardScaler:** mean 0, standard deviation 1.
- **MinMaxScaler:** squeezes values into the range 0 to 1.
- Needed by distance-based and gradient-based models (kNN, logistic regression). Not needed by decision trees.

### My Titanic results
| Method | Accuracy |
|---|---|
| Single split | 0.7709 |
| 5-fold CV mean | 0.78 (std 0.008) |

Survival rate: overall 38.4%, women 74% vs men 19%, first class 63% vs third class 24%.

---

## Tuesday: Decision Trees and k-Nearest Neighbors

### Decision Tree
- Splits the data with yes/no questions on features (e.g. "is color_intensity <= 3.82?"). At each node it picks the split that makes the groups purest.
- **Gini impurity** (or entropy) measures how mixed the classes are in a node. 0 means pure.
- A very deep tree keeps splitting until every leaf is pure, so it **memorises** the training data (train accuracy 1.0) and overfits.
- Controls: `max_depth`, `min_samples_leaf`.
- Trees do not need feature scaling, because each split looks at one feature at a time.
- Overfitting sign: train accuracy far above test accuracy.

### k-Nearest Neighbors (kNN)
- To predict a point, find its k closest training points (by distance) and take a **majority vote**.
- **Scaling matters a lot:** a feature with big numbers (e.g. Wine `proline` in the hundreds) dominates the distance.
- **k = 1:** the model copies its nearest neighbour, so train accuracy is 1.0 and it follows noise (overfitting).
- **Very large k:** the vote averages over too many points and the boundary becomes too smooth (underfitting).
- Overfitting sign: small k, train accuracy much higher than test.

### My results
| Experiment | Result |
|---|---|
| Wine tree, first split | `color_intensity <= 3.82` |
| Breast Cancer tree, first split | `worst radius <= 16.795` |
| Tree best depth | max_depth = 3 on both datasets (train accuracy reaches 1.0 at depth 15) |
| kNN on Wine, no scaling vs scaling | 0.778 vs 0.933 |
| Best k | about 15 on Wine, 5 on Breast Cancer |

---

## Wednesday: Naive Bayes and Pipelines

### Naive Bayes
- Uses **Bayes' theorem** to compute the probability of each class given the features, and predicts the most probable class.
- **"Naive"** means it assumes all features are independent of each other given the class. This is rarely true in real data, but the model still works well.
- **GaussianNB** assumes each continuous feature follows a normal distribution within each class.
- Strengths: very fast, works well on small data. Weakness: the independence assumption can hurt when features are strongly correlated.

### Pipelines
- A **Pipeline** chains preprocessing and a model into one object: `fit` runs every step in order, `predict` takes raw data and returns predictions.
- **ColumnTransformer** applies different preprocessing to different columns (numeric: imputer + scaler, categorical: imputer + one-hot encoder).
- **Why it prevents leakage:** inside `cross_val_score`, the pipeline refits the imputer, scaler and encoder on each training fold only, so the test fold never influences preprocessing.
- Swapping the model needs only one line change (the classifier step).

### My results
Wine and Breast Cancer (5-fold CV mean):

| Dataset | Naive Bayes | Decision Tree | kNN (scaled) |
|---|---|---|---|
| Wine | 0.966 | 0.871 | 0.949 |
| Breast Cancer | 0.939 | 0.919 | 0.965 |

Titanic, same pipeline with three classifiers (5-fold CV mean): Naive Bayes 0.777, Decision Tree 0.790, kNN 0.805.

---

## Thursday: Feature Selection vs Dimensionality Reduction

### Feature selection
- Keeps a **subset of the original features**, so they stay interpretable.
- Example: `SelectKBest` on Wine picked flavanoids, proline, od280/od315, alcohol and color_intensity as the top 5.
- Tree feature importances are another way to select features.

### Dimensionality reduction
- Creates **new features that mix the originals**. Less interpretable, but compresses the data.
- Example: PCA on Wine turned 13 features into 2 components. PC1 keeps 36.2% and PC2 keeps 19.2% of the variance (55.4% together), and the three classes still separate well in the plot.
- PCA needs scaled data. `explained_variance_ratio_` shows how much variance each component keeps.

### Metrics
- **Confusion matrix:** the diagonal shows correct predictions, off-diagonal cells show mistakes.
- **Precision:** of the predicted positives, how many were right. **Recall:** of the real positives, how many were found. **F1:** the balance of the two.
- Macro average is used for the 3-class Wine data.

---

## Friday: Performance Comparison Summary

All models were run in the same pipeline (imputer, scaler or encoder as needed, then the classifier). CV = 5-fold cross-validation accuracy. Test metrics come from one stratified 75/25 split. (These numbers differ slightly from Wednesday's table because this pipeline uses `max_depth=4` for the tree and also scales the features for Naive Bayes.)

| Dataset | Naive Bayes | Decision Tree | kNN | Best (CV) |
|---|---|---|---|---|
| Titanic | 0.777 | 0.790 | **0.805** | kNN |
| Wine | **0.966** | 0.894 | 0.949 | Naive Bayes |
| Breast Cancer | 0.928 | 0.921 | **0.965** | kNN |


- **Titanic: kNN (CV 0.805).** It is ahead of the Decision Tree by 1.5 points and Naive Bayes by 2.8 points.
- **Wine: Naive Bayes (CV 0.966, test accuracy 0.978).** It is 1.7 points ahead of kNN and 7.2 points ahead of the Decision Tree. The classes are well separated and the features are roughly bell-shaped within each class, which suits Gaussian Naive Bayes. The tree overfits more on this small dataset (178 rows).
- **Breast Cancer: kNN (CV 0.965, test accuracy 0.979).** It is about 4 points ahead of Naive Bayes (0.928) and Decision Tree (0.921). After scaling, malignant and benign cases form fairly consistent neighbourhoods. Naive Bayes is likely hurt because many features are strongly correlated (radius, perimeter and area), which breaks its independence assumption.

### The metric that matters most: recall on the malignant class
In Breast Cancer, missing a malignant tumour is the worst mistake, so recall on the malignant class is the key metric. kNN is best here too (**0.943**), then Decision Tree (0.925), then Naive Bayes (0.887).

### Strength and weakness of each model
- **Decision Tree:** easy to interpret and needs no scaling, but it overfits and its score changes a lot between folds.
- **kNN:** accurate once the features are scaled and has no training step, but it needs scaling and becomes slow on large data.
- **Naive Bayes:** very fast and works well on small data, but it assumes features are independent, which fails when features are correlated.

### Caution
The test sets are small (45 rows for Wine, 143 for Breast Cancer, 223 for Titanic), so one wrong prediction moves the accuracy by 0.5 to 2 points. This is why the CV numbers are more reliable than a single split.

---

## Friday: 2-Minute Explanations (practice)

### Cross-validation
A single train/test split depends on luck: the test set may be easy or hard. In k-fold cross-validation I split the data into k parts and train k times, each time keeping a different part for testing, then I average the k scores. Every row gets used for both training and testing, the average is more trustworthy, and the standard deviation tells me how stable the model is. For example, on Titanic the 5-fold CV mean was 0.78 with a standard deviation of only 0.008.

### Pipeline and why it prevents leakage
A pipeline chains preprocessing (imputer, scaler, encoder) and the model into one object. Data leakage happens when information from the test set affects training, for example if I fill missing ages with the median of the whole dataset. In a pipeline, `fit` learns the median, mean and categories from the training data only, and cross-validation refits all of it on each training fold. The test fold stays unseen, so the score is honest. A pipeline also lets me go from raw data to a prediction, and swap the model by changing one line.

### The three classifiers
- **Decision Tree:** asks yes/no questions on features and picks the split that makes the groups purest (lowest Gini impurity). A very deep tree memorises the training data and overfits, so I limit `max_depth`.
- **kNN:** to predict a point, it finds the k nearest training points and takes a majority vote. It uses distance, so scaling is essential. k = 1 overfits and a very large k underfits.
- **Naive Bayes:** uses Bayes' theorem to compute the probability of each class given the features. "Naive" means it assumes the features are independent of each other given the class. It is very fast and works well on small data.

---

## Checkpoint checklist
- [x] Full pipeline with preprocessing and 3 classifiers
- [x] Confusion matrices for all three models
- [x] Written performance comparison summary
- [ ] Notebook runs end to end (Restart and run all) and is on GitHub
- [ ] Can explain: train/test split, cross-validation, scaling, missing data, encoding, feature selection vs dimensionality reduction
