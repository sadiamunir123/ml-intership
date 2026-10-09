# ML Internship

## Week 1: ML Fundamentals and Basic Supervised Models

### Files
- `week1/week1_eda.ipynb`: Exploratory Data Analysis on Iris and California Housing
- `week1/week1_models.ipynb`: Linear Regression, Logistic Regression, Gradient Descent from scratch, Bias-Variance, Linear vs Logistic comparison
- `week1/week1_notes.md`: Notes on ML concepts, metrics and bias-variance

### Results
- Linear Regression (California Housing): MSE about 0.56, R² about 0.58
- Gradient Descent from scratch (NumPy): MSE 0.5546, R² 0.5768 (matches scikit-learn)
- Logistic Regression (Iris): accuracy 0.97

Note: Boston Housing was removed from scikit-learn (v1.2+), so California Housing is used instead.

### Setup
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipykernel
```