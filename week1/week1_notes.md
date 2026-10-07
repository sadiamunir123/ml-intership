# Week 1 Notes: ML Fundamentals and Basic Supervised Models

## 1. What is Machine Learning?
Machine learning is a way of building software that learns patterns from data instead of following hand-written rules. For example, a spam filter learns from thousands of labeled emails rather than a programmer writing every rule.

## 2. Types of ML

### Supervised Learning
- The data comes with the correct answers (labels). The model learns the mapping from input to output.
- **Regression:** predicts a number. Example: house features -> house price (California Housing).
- **Classification:** predicts a category. Example: flower measurements -> species (Iris).

### Unsupervised Learning
- The data has no labels. The model finds structure by itself.
- Example: clustering customers into groups.

### Key terms
- **Feature:** an input column.
- **Label / target:** the value we want to predict.
- **Training data:** data the model learns from.
- **Test data:** unseen data used to check the model.
- **Model:** the function that maps features to a prediction.

## 3. ML Workflow
1. **Problem definition:** decide what to predict and what counts as success.
2. **Data collection:** gather the data.
3. **Preprocessing:** handle missing values, remove duplicates, encode categories, scale features, split into train/test.
4. **Model training:** fit the model on the training set.
5. **Evaluation:** measure performance on unseen data using suitable metrics.
6. **Deployment:** put the model into a real application and monitor it.

It is a loop, not a line: poor evaluation sends us back to preprocessing or data collection, and deployed models need monitoring because real-world data changes over time.

## 4. Exploratory Data Analysis (EDA)
EDA means understanding the data before modelling: shape, types, missing values, distributions, outliers and correlations.

**Iris**
- 150 rows, 5 columns (4 features + target). Target has 3 classes, 50 samples each (balanced).
- No missing values. Classification problem.
- Petal length and petal width are strongly correlated with the target and separate the classes best. Setosa is clearly separated; versicolor and virginica overlap slightly.

**California Housing**
- 20,640 rows, 9 columns (8 features + target `MedHouseVal`). Regression problem.
- No missing values.
- `MedInc` (median income) has the strongest correlation with price (about 0.69).
- `MedHouseVal` is capped at 5.0 and `HouseAge` at 52. `AveRooms`, `AveOccup` and `Population` have large outliers.

Note: the Boston Housing dataset was removed from scikit-learn (v1.2+) for ethical reasons, so California Housing was used instead.

## 5. Linear Regression
- Predicts a continuous value as a weighted sum of the features plus a bias term.
- Trained on California Housing (80/20 split): MSE about 0.56, R² about 0.58. The model explains about 58% of the variation in prices.

## 6. Gradient Descent
- An optimization method that repeatedly updates the weights in the direction that reduces the cost (error).
- **Batch GD:** uses all samples per step. Smooth but slow on big data.
- **Stochastic GD:** uses 1 sample per step. Fast but noisy.
- **Mini-batch GD:** uses small chunks. A balance of both and the most common choice.
- **Learning rate:** too small means very slow learning, too large makes the cost jump or diverge.
- Features were scaled with StandardScaler because gradient descent converges poorly on very different scales.
- My NumPy implementation: MSE = 0.5546, R² = 0.5768. sklearn: MSE = 0.5559, R² = 0.5758. Almost identical.

## 7. Logistic Regression
- Used for classification. It applies the sigmoid function to a linear score, so the output is a probability between 0 and 1.
- Trained on Iris (stratified 80/20 split, 30 test samples): accuracy = 0.97.
- Only one mistake: one versicolor was predicted as virginica.

### Linear vs Logistic
- Linear predicts a continuous number (MSE, R²). Logistic predicts a class probability (accuracy, precision, recall, F1).
- On 0/1 labels, a straight line goes below 0 and above 1, while the sigmoid stays between 0 and 1. This is why logistic regression exists.

## 8. Evaluation Metrics

**Classification** (based on the confusion matrix: TP, TN, FP, FN)
- **Accuracy** = correct predictions / all predictions. Can be misleading on imbalanced data (a model that always says "no disease" can be 99% accurate and still useless).
- **Precision** = TP / (TP + FP). Of everything predicted positive, how much was really positive. Important when false alarms are costly (spam filter).
- **Recall** = TP / (TP + FN). Of all real positives, how many we found. Important when missing a case is costly (disease detection).
- **F1-score** = harmonic mean of precision and recall. Useful when we need a balance, especially on imbalanced data.

**Regression**
- **MSE** = average of squared errors. Lower is better. Large errors are punished more.
- **R²** = fraction of the variance in the target explained by the model. 1 is perfect, 0 is no better than predicting the mean.

## 9. Bias-Variance Tradeoff
- **Bias:** error from overly simple assumptions. High bias leads to **underfitting**: the model misses the pattern and both train and test error are high.
- **Variance:** error from being too sensitive to the training data. High variance leads to **overfitting**: train error is very low but test error is higher, because the model learned the noise.
- As model complexity increases, bias goes down and variance goes up. The goal is the point where test error is lowest.
- In my polynomial experiment on noisy sine data: degree 1-2 underfit (error about 0.18-0.21), degree 3-4 fit well (about 0.03-0.04), and for degree 15 the train error kept falling (about 0.025) while the test error stayed higher (about 0.045).
- Ways to reduce overfitting: more data, simpler model, regularization, cross-validation.

## 10. Questions I still have
- (apne sawal yahan likhein)