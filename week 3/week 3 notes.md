# Week 3 Notes: Unsupervised Learning and Hyperparameter Tuning

## 1. Clustering (Unsupervised Learning)
- Clustering is **unsupervised learning**: the data has no labels, and the algorithm groups similar points together by itself.
- **Difference from classification:** classification predicts a known category from labeled data, so we can measure accuracy. Clustering has no correct answer to compare against, so we judge results with the **Elbow method**, the **Silhouette score**, and whether the clusters make sense to a human.
- Example used this week: grouping 200 mall customers into segments by income and spending.

## 2. K-Means
1. Pick k and place k centres (centroids), using k-means++ so the starting centres are far apart.
2. Assign each point to its nearest centre.
3. Move each centre to the mean of the points assigned to it.
4. Repeat steps 2 and 3 until the centres stop moving.

- **Inertia** = the sum of squared distances from each point to its own cluster centre (within-cluster sum of squares). Lower means tighter clusters.
- Inertia always falls as k grows (at k = number of points it is 0), so inertia alone cannot pick k.
- K-Means needs k in advance, and the result depends on the starting centres, so I used `n_init=10` and a fixed `random_state`.

### Why scaling matters
K-Means uses distances, like kNN. A feature with large values would dominate a feature with small values. I used `StandardScaler` so all features count equally. Always scale before K-Means and hierarchical clustering.

## 3. Choosing k: Elbow and Silhouette

**Elbow method**
- Plot inertia against k and look for the bend where adding more clusters stops helping much.
- It is a visual judgement, so the bend is not always clear.

**Silhouette score**
- For each point it compares the distance to its own cluster with the distance to the nearest other cluster. Range is -1 to 1. Close to 1 means tight, well-separated clusters.
- The average silhouette score is plotted for each k, and the peak suggests the best k.
- The per-cluster silhouette plot shows if one cluster is weak (many bars below the average line) or very uneven in size.

**Why they can disagree:** the Elbow is a visual judgement about the shape of a curve, while the Silhouette is a number that rewards separation. They measure different things, so I use both plus common sense about what the clusters mean.

## 4. Mall Customers: K-Means Results
- 200 rows. Columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100).
- Clustered on **Annual Income and Spending Score** only, scaled.
- From the raw scatter plot I could see about 5 groups: a big blob in the middle and four smaller groups in the corners.

| k | Inertia | Silhouette |
|---|---------|------------|
| 2 | 269.7 | 0.321 |
| 3 | 157.7 | 0.467 |
| 4 | 108.9 | 0.494 |
| **5** | **65.6** | **0.555** |
| 6 | 55.1 | 0.540 |
| 7 | 44.9 | 0.528 |
| 8 | 37.2 | 0.455 |
| 9 | 32.4 | 0.457 |
| 10 | 30.0 | 0.443 |

The Elbow bends at k = 5 and the Silhouette peaks at k = 5 (0.555), so **k = 5**. Final model: silhouette 0.5547, inertia 65.57.

| Customers | Avg age | Avg income (k$) | Avg spending | % female | Meaning |
|-----------|---------|-----------------|--------------|----------|---------|
| 81 | 42.7 | 55.3 | 49.5 | 59.3 | Average customers, the largest group |
| 39 | 32.7 | 86.5 | 82.1 | 53.8 | High income, high spending (best target) |
| 22 | 25.3 | 25.7 | 79.4 | 59.1 | Low income, high spending (impulsive) |
| 35 | 41.1 | 88.2 | 17.1 | 45.7 | High income, low spending (careful) |
| 23 | 45.2 | 26.3 | 20.9 | 60.9 | Low income, low spending (sensible) |

- Gender was not used for clustering, and % female is between about 46% and 61% in every cluster, so the groups are defined by income and spending, not by gender.
- Use in business: the high income and high spending group is the best target for loyalty offers; the high income and low spending group could be reached with better promotions.

## 5. Hierarchical (Agglomerative) Clustering
- Start with every point as its own cluster and repeatedly merge the two closest clusters. The result is a tree, the **dendrogram**.
- **Linkage** decides what "closest" means between clusters: **ward** (merge the pair that increases total within-cluster variance the least, gives compact similar-sized clusters, used here), **complete**, **average**, **single**.
- **Cutting the dendrogram:** draw a horizontal line across the tallest vertical gaps where no merging happens. The number of vertical lines the cut crosses is the number of clusters.

### K-Means vs Hierarchical
- K-Means needs k in advance; hierarchical lets you read k off the dendrogram.
- Hierarchical is slower on big data (it works with distances between all points).
- On the Mall data (Income and Spending, k = 5): Agglomerative silhouette = 0.554 and K-Means = 0.555, and the Adjusted Rand Index = 0.94, so the two methods found almost the same groups. A cut at height 7 gives 5 clusters. The last merge distances were 2.49, 2.70, 3.15, 4.07, 4.31, 4.35, 9.45, 10.14, 15.19, 15.62, so the biggest jump is from 4.35 to 9.45, and any cut between those two heights gives 5 clusters. Cluster numbers are arbitrary, so I compared the groupings, not the labels.

## 6. PCA for Visualising Clusters
- With 3 or more features we cannot plot the data directly. PCA projects the data onto 2 components so the clusters can be drawn in 2-D.
- **Explained variance ratio** shows how much of the data's variation the 2 components keep. If it is low, overlapping clusters in the picture may still be well separated in the full space.
- PCA was used only for the picture; the clustering was done on all the features.

### Four-feature clustering (Age, Income, Spending, Gender)
- Silhouette scores were much lower than with 2 features (about 0.25 to 0.42, against 0.55), and in my run the score kept rising up to k = 10 with no clear peak, so I chose k = 6 by looking at how interpretable the clusters were.
- The 2 PCA components kept about 59.9% of the variance (PC1 33.7%, PC2 26.2%), so the 2-D picture is only a partial summary.
- With k = 6 the silhouette score was 0.3311 and the cluster sizes were 41, 38, 34, 35, 23 and 29.
- Gender is a 0/1 column and, after scaling, separates points strongly. **Four of the six clusters were exactly 100% female or 0% female**, so Gender was driving the split more than behaviour. The 2-feature clustering from Monday was easier to explain and more useful.

| Customers | Avg age | Avg income (k$) | Avg spending | % female | Name |
|-----------|---------|-----------------|--------------|----------|------|
| 41 | 50.6 | 49.7 | 40.1 | 100.0 | Older women, mid income, mid spending |
| 38 | 28.1 | 58.3 | 71.3 | 0.0 | Younger men, mid income, high spending |
| 34 | 26.0 | 39.5 | 59.5 | 100.0 | Younger women, low income, mid spending |
| 35 | 55.6 | 52.8 | 36.2 | 0.0 | Older men, mid income, low spending |
| 23 | 32.1 | 90.0 | 81.4 | 91.3 | Younger, high income, high spending (almost all women) |
| 29 | 36.5 | 89.5 | 18.0 | 55.2 | Middle-aged, high income, low spending (mixed gender) |

Names follow the notebook's rules (age under 35 is younger and 45 or more is older; income under 40 is low and 60 or more is high; spending under 40 is low and 60 or more is high), with gender added because four clusters are entirely one gender. Some values sit close to a cutoff (for example income 39.5 and spending 59.5 in the third cluster), so the labels are approximate.

What a business might do with each group:
- Older women, mid income, mid spending: steady customers, keep them with regular offers.
- Younger men, mid income, high spending: spend a lot for their income, target with trendy products and promotions.
- Younger women, low income, mid spending: price-sensitive, discounts and student-style offers.
- Older men, mid income, low spending: spend little, try personal offers to raise their visits.
- Younger, high income, high spending: best customers, loyalty rewards and premium products.
- Middle-aged, high income, low spending: have money but do not spend here, find out what they want and promote it.

**Cluster profiling:** compute the mean Age, Income and Spending per cluster and give each cluster a plain-English name, then write what a business could do with it.

## 7. Hyperparameter Tuning
- **Parameters** are learned by the model from the data (weights in logistic regression, split points in a tree). **Hyperparameters** are set by me before training (`n_neighbors` in kNN, `max_depth` in a tree, `C` in logistic regression and SVM).
- **GridSearchCV** tries every combination in a grid, scores each with cross-validation, and reports the best (`best_params_`, `best_score_`, `cv_results_`).
- `best_score_` is the mean **CV accuracy on the training data**, not the test score. `std_test_score` in `cv_results_` is the spread across folds.
- **A final test set is still needed.** The test set is split off before tuning and used once at the end. If I tuned on it, the final score would be too optimistic.
- **Pipelines:** the scaler goes inside the Pipeline so that each CV fold re-fits it on its own training part only (no data leakage). Parameter names use the step name and two underscores: `model__n_neighbors`, `model__max_depth`, `model__C`.

## 8. GridSearchCV Results (Breast Cancer, 569 samples, 30 features)
- 80/20 stratified split, 5-fold CV, scaler plus classifier in a Pipeline. Malignant = class 0.

| Model | Default CV | Tuned CV | Default test | Tuned test |
|-------|-----------|----------|--------------|------------|
| kNN | 0.9626 | 0.9714 | 0.9561 | 0.9649 |
| Decision Tree | 0.9165 | 0.9363 | 0.9123 | 0.8772 |
| Logistic Regression | 0.9780 | 0.9824 | 0.9825 | 0.9737 |

- Best parameters: kNN = `metric='manhattan'`, `n_neighbors=3`, `weights='uniform'`; Decision Tree = `criterion='entropy'`, `max_depth=4`, `min_samples_leaf=5`; Logistic Regression = `C=0.1`.
- **CV gain vs fold-to-fold spread:** kNN +0.0088 (std 0.0132), Decision Tree +0.0198 (std 0.0189), Logistic Regression +0.0044 (std 0.0054). Only the Decision Tree gain is about as large as the spread, so none of the gains is clearly bigger than noise.
- The top kNN combinations were nearly tied (CV 0.9714, then 0.9692, then 0.9670), all within one standard deviation of each other. The best `n_neighbors = 3` is not on the edge of the searched range (1 to 30), and the CV curve falls to about 0.952 for k = 27 to 30.
- For Logistic Regression, `C = 0.1` is in the middle of the grid (0.001 to 1000), so the grid does not need widening. CV accuracy was lower on both sides (0.9516 at C = 0.01 and 0.9780 at C = 1).
- **Tuning raised the CV score of all three models, but the tuned Decision Tree and Logistic Regression scored lower on the test set than their defaults.** The search picks the combination that is luckiest on the CV folds, and the test set has only about 114 samples, so one extra correct prediction changes accuracy by about 0.9 percentage points.
- For Breast Cancer, **recall on the malignant class** matters most, because missing a malignant case is the costly mistake. I chose the final model by CV score, not by test score, so the test set stays untouched.

## 9. RandomizedSearchCV and Reading Tuning Results
- **RandomizedSearchCV** samples `n_iter` random combinations from lists or distributions (`randint` for integers, `loguniform` for scales such as `C` and `gamma`). It is faster on big spaces and often almost as good as a full grid.
- **Grid vs Random:** grid is exhaustive but its cost explodes as parameters are added; random covers wide ranges cheaply.

| | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| Combinations tried | 720 | 50 |
| Time (my machine) | 124.5 s | 1.3 s |
| Best CV score | 0.9385 | 0.9341 |
| Test accuracy | 0.9035 | 0.9123 |

Random search was about 98 times faster (124.5 s vs 1.3 s). Its best CV score (0.9341) was only 0.0044 below the grid's (0.9385), which is smaller than the fold-to-fold spread (0.0139 and 0.0192), so the two were effectively tied. Random search even scored slightly higher on the test set (0.9123 vs 0.9035), which is noise, not a real advantage.

| Model (random search) | Default CV | Tuned CV | CV gain | CV std | Default test | Tuned test | Malignant recall | Search time |
|-----------------------|-----------|----------|---------|--------|--------------|------------|------------------|-------------|
| kNN | 0.9626 | 0.9714 | 0.0088 | 0.0132 | 0.9561 | 0.9649 | 0.9286 | 1.0 s |
| SVM (RBF) | 0.9692 | 0.9758 | 0.0066 | 0.0108 | 0.9825 | 0.9825 | 0.9762 | 1.0 s |
| Decision Tree | 0.9165 | 0.9341 | 0.0176 | 0.0139 | 0.9123 | 0.9123 | 0.9048 | 1.3 s |

### How to read the results
- **Is the gain bigger than the spread?** For kNN (0.0088 vs std 0.0132) and SVM (0.0066 vs 0.0108) the CV gain was smaller than the fold-to-fold standard deviation, so it could be noise. Only the Decision Tree gain (0.0176 vs 0.0139) was a little larger than its std, and it did not carry over to the test set (0.9123 before and after).
- **Malignant recall:** the SVM had the highest recall on the malignant class (0.9762), which matters most for Breast Cancer, even though its test accuracy did not change with tuning (0.9825).
- **Is the best value on the edge of the range?** If yes, widen the range and search again.
- **Heatmap of CV score** (`max_depth` vs `min_samples_leaf`): look for a bright region, not one bright cell. If many neighbouring cells score almost the same, the exact best value does not matter much.
- **Overfitting the search:** with many combinations, the best CV score is partly luck. The honest number is the score on the untouched test set.

### Was tuning worth it?
Tuning gave small CV gains and no clear test gain, but random search found the same quality for a fraction of the time. Defaults were already strong on this clean dataset, so tuning matters more on harder data.

## 10. Explanation of the Tuning Results

**What I tuned.** I used the Breast Cancer data (569 samples). I kept 20% of it aside as a test set and used 5-fold cross-validation on the rest. Every model was in a Pipeline with a scaler, so nothing leaked from the validation folds.

With GridSearchCV I tuned three models:
- kNN: `n_neighbors` (1 to 30), `weights` and `metric` (120 combinations)
- Decision Tree: `max_depth`, `min_samples_leaf` and `criterion` (72 combinations)
- Logistic Regression: `C` (7 values from 0.001 to 1000)

With RandomizedSearchCV I tuned a Decision Tree, kNN and an SVM (RBF) with random samples from the ranges.

**Best parameters.**
- kNN: `manhattan`, `n_neighbors=3`, `uniform`
- Decision Tree: `entropy`, `max_depth=4`, `min_samples_leaf=5`
- Logistic Regression: `C=0.1`

**How much did tuning help?** Only a little. In cross-validation every tuned model scored a bit higher than its default (for example kNN 0.9626 to 0.9714 and Logistic Regression 0.9780 to 0.9824). But most of these gains were smaller than the fold-to-fold spread (the standard deviation), so they could just be noise. Only the Decision Tree gain was a bit bigger than its spread.

**Did it help on the test set?** Mostly no. The tuned Decision Tree and Logistic Regression scored lower on the test set than the default models. The tuned kNN was better by one sample, and the tuned SVM and Decision Tree stayed the same in the random search. The test set is small (114 samples), so one sample changes accuracy by about 0.9 points. The search picks what looks best on the CV folds, and that can be luck. So a better CV score does not always mean a better test score.

**Was the best value at the edge of the range?** No, not in the GridSearch runs. The best values were inside the ranges I searched (for example `C=0.1` was in the middle of the grid and CV was lower on both sides), so I did not need to widen them.

**Grid vs random.** On the same Decision Tree problem, GridSearchCV tried 720 combinations and took 124.5 seconds (best CV 0.9385). RandomizedSearchCV tried only 50 combinations and took 1.3 seconds (best CV 0.9341). Random search was about 98 times faster. The CV scores differ by only 0.0044, which is less than the spread, so the two are about equal.

**Was tuning worth it?** On this clean, small dataset the default models were already good, so tuning gave small gains and no clear improvement on unseen data. Grid search is fine for small grids. For bigger search spaces, random search gives almost the same result in much less time. I should always report the final score on a test set that the search never saw.

**What matters most for Breast Cancer?** Missing a malignant case is the worst mistake, so I checked recall on the malignant class and not only accuracy. The test set has 42 malignant cases. The SVM found 41 of 42 (recall 0.9762), Logistic Regression found 40 (0.9524), and kNN and the Decision Tree found 39 or fewer. These differences are only one or two cases, so I should be careful not to over-read them. Logistic Regression had the best CV score (0.9824), while the SVM had the best malignant recall. For a real medical use I would pick the model that misses the fewest malignant cases and check it on more data.

## 11. Questions I still have
- When Elbow and Silhouette give different answers, which one should I trust more?
- How does K-Means handle clusters that are not round, or that have very different sizes?
- Should categorical features like Gender be used in distance-based clustering at all?
- How many random-search iterations (`n_iter`) are enough?
