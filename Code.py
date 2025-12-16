# House Price Prediction – Code Overview

This document outlines the complete code workflow implemented for the House Price Prediction regression project.

---

## 1. Import Required Libraries
```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## 2.Load dataset
```python
df = pd.read_csv("data/house_data.csv")

## 3.Initial Data exploration
```python
df.shape
df.info()
df.head()
df.describe()

## 4.Data cleaning and preprocessing

# Handle missing values
df.isnull().sum()

# Drop or fill missing values
df.fillna(df.mean(), inplace=True)

# Remove unnecessary columns (if any)
df.drop(columns=["id"], inplace=True, errors="ignore")


## 5.Exploratory data analysis(EDA)

plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


## 6.Feature target selection

X = df.drop("price", axis=1)
y = df["price"]


## 7.Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


## 8.Model initialization

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor()
}


## 9.Model training and predictiion

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


##10.Model Evaluation

for name, model in models.items():
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name}")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
    print("-" * 40)


## 11.Model Comparison Summary

Compared multiple regression models using MSE, MAE, and R²

Identified Linear Regression as the best-performing model

Observed limitations of tree-based models due to overfitting


## 12.Conclusion

The project demonstrates an end-to-end regression workflow including data preprocessing, EDA, feature selection, model training, evaluation, and comparison using standard regression metrics.



