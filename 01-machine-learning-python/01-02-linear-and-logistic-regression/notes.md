# Linear and Logistic Regression

## Introduction to Regression

- **Regression** is a supervised learning technique for modeling the relationship between a continuous target variable and explanatory features.
- **Applications:** Predicting sales, maintenance costs, rainfall, disease spread, house prices, and more.
- **Types of Regression:**
  - **Simple regression:** One independent variable.
  - **Multiple regression:** Two or more independent variables.
  - **Can be linear or nonlinear** depending on the underlying relationship.

---

## Simple Linear Regression

### Concept

- Models a linear relationship between a single feature (independent variable) and a continuous target (dependent variable).
- **Equation:** `y = θ₀ + θ₁x`
  - `y`: predicted value
  - `x`: feature value
  - `θ₀`: intercept (bias)
  - `θ₁`: slope (coefficient)

### How It Works

- Fits a best-fit line through the data to minimize prediction errors.
- **Error Metric:** Mean Squared Error (MSE)
- **Ordinary Least Squares (OLS):** Closed-form solution for finding θ₀ and θ₁ by minimizing MSE.

### Strengths & Weaknesses

- **Pros:** Easy to interpret, fast for small datasets.
- **Cons:** Sensitive to outliers, cannot capture nonlinear relationships.

---

## Multiple Linear Regression

### Concept

- Extends simple regression to multiple features: `y = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`
- Can model more complex relationships and analyze the impact of each feature.

### Feature Selection & Pitfalls

- **Collinearity:** Avoid using highly correlated features together.
- **Overfitting:** Too many features can cause the model to memorize training data and perform poorly on new data.
- **Categorical Variables:** Convert to numerical (e.g., one-hot encoding).

### Error Metric

- **Mean Squared Error (MSE):** Same as in simple regression.
- **Goal:** Find coefficients that minimize MSE.

### Model Fitting

- **Ordinary Least Squares (OLS):** Uses linear algebra for parameter estimation.
- **Gradient Descent:** Iterative optimization, especially useful for large datasets.

---

## Polynomial & Nonlinear Regression

### Polynomial Regression

- Models nonlinear relationships by introducing polynomial terms: `y = θ₀ + θ₁x + θ₂x² + ... + θₐxᵈ`
- **Pros:** Captures curves.
- **Cons:** High-degree polynomials can overfit.

### Other Nonlinear Regression

- **Exponential, logarithmic, periodic (e.g., sinusoidal) regression**
- Used when data follows growth, decay, or cyclical patterns.

### Model Selection

- Use scatterplots to visually assess the relationship.
- Try different models and compare error metrics.

---

## Logistic Regression

### Concept

- **Logistic regression** is used for binary classification (target is 0 or 1).
- **Outputs probability** that an observation belongs to class 1.
- **Sigmoid function (logit):** `σ(x) = 1 / (1 + e^(-x))`
- **Model equation:** `p̂ = σ(θ₀ + θ₁x₁ + ... + θₙxₙ)`
  - `p̂`: predicted probability of class 1

### Decision Boundary

- Choose a threshold (e.g., 0.5) to assign classes:
  - `p̂ > 0.5` → class 1
  - `p̂ ≤ 0.5` → class 0

### Use Cases

- Predicting customer churn, disease presence, loan default, etc.

---

## Model Training & Optimization

### Cost Functions

- **Regression:** Mean Squared Error (MSE)
- **Logistic Regression:** Log Loss (Cross-Entropy Loss)

### Optimization Algorithms

- **Gradient Descent:**
  - Iteratively updates parameters in the direction of steepest decrease in the cost function.
  - **Learning Rate:** Controls the step size; too large can overshoot, too small can be slow.
- **Stochastic Gradient Descent (SGD):**
  - Uses random subsets (mini-batches) for faster, scalable optimization.
  - More likely to find global minima, but can fluctuate near the optimum.

---

## Practical Examples: Code Walkthroughs

### Multiple Linear Regression (CO2 Emissions Example)

```python
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create dummy dataset for CO2 emissions prediction
data = {
    'ENGINESIZE': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 1.8, 2.2, 2.8, 3.2, 1.6, 2.4, 2.9, 3.4, 3.8],
    'FUELCONSUMPTION_COMB_MPG': [28, 25, 22, 19, 16, 14, 26, 24, 20, 18, 27, 23, 21, 17, 15],
    'CO2EMISSIONS': [180, 220, 250, 290, 330, 370, 200, 240, 270, 310, 190, 260, 280, 320, 350]
}
df = pd.DataFrame(data)

# Select relevant features
features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
target = 'CO2EMISSIONS'

# Split data into train/test sets
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.2f}")

# Plotting predicted vs. actual
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Multiple Linear Regression: Actual vs. Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.show()
```

### Logistic Regression (Customer Churn Example)

```python
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, classification_report

# Create dummy customer churn dataset
data = {
    'tenure': [12, 24, 6, 36, 18, 3, 48, 9, 15, 30, 2, 42, 21, 8, 33],
    'age': [25, 45, 32, 55, 28, 23, 62, 35, 41, 50, 22, 58, 38, 29, 47],
    'income': [35000, 65000, 42000, 85000, 38000, 28000, 95000, 45000, 58000, 75000, 25000, 88000, 52000, 33000, 72000],
    'monthly_charges': [65, 85, 70, 95, 60, 55, 105, 75, 80, 90, 50, 100, 85, 65, 92],
    'total_charges': [780, 2040, 420, 3420, 1080, 165, 5040, 675, 1200, 2700, 100, 4200, 1785, 520, 3036],
    'churn': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
}
churn_df = pd.DataFrame(data)

# Select features and target
features = ['tenure', 'age', 'income', 'monthly_charges', 'total_charges']
X = churn_df[features].values
y = churn_df['churn'].values

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Train logistic regression model
LR = LogisticRegression(random_state=42)
LR.fit(X_train, y_train)

# Predict class and probabilities
y_pred = LR.predict(X_test)
y_pred_prob = LR.predict_proba(X_test)

# Evaluate with multiple metrics
loss = log_loss(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Log Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Model Coefficients: {LR.coef_[0]}")
print(f"Model Intercept: {LR.intercept_[0]:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show probability predictions for first few test samples
print("\nProbability Predictions (first 3 test samples):")
for i in range(min(3, len(y_test))):
    prob_no_churn = y_pred_prob[i][0]
    prob_churn = y_pred_prob[i][1]
    actual = y_test[i]
    predicted = y_pred[i]
    print(f"Sample {i+1}: P(No Churn)={prob_no_churn:.3f}, P(Churn)={prob_churn:.3f}, Actual={actual}, Predicted={predicted}")
```

---

## Key Terms & Takeaways

### Key Terms

- **Regression:** Predicting continuous values.
- **Classification:** Predicting discrete classes.
- **Linear Regression:** Predicts a continuous target using a linear relationship.
- **Multiple Linear Regression:** Uses several features for prediction.
- **Polynomial Regression:** Uses polynomial terms for nonlinear data.
- **Logistic Regression:** Predicts probability for binary classification.
- **Overfitting:** Model fits training data too closely, performs poorly on new data.
- **Collinearity:** Features are highly correlated; can distort model interpretation.
- **Gradient Descent:** Iterative optimization algorithm.
- **Stochastic Gradient Descent (SGD):** Uses random data subsets for optimization.
- **Mean Squared Error (MSE):** Average squared difference between actual and predicted values.
- **Log Loss:** Measures performance of a classification model where the prediction is a probability.

### Actionable Tips

- **Always visualize your data** (scatterplots, correlation matrices) before modeling.
- **Standardize features** for models sensitive to scale (e.g., logistic regression).
- **Use train/test splits** to evaluate model generalization.
- **Avoid overfitting** by limiting features and using regularization if needed.
- **Check for collinearity** and remove redundant features.
- **Interpret model coefficients** to understand feature impact.

---