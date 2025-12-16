# House Price Prediction Using Machine Learning

## Project Overview
This project focuses on predicting house prices using supervised machine learning regression techniques. The objective is to analyze housing data, identify key factors influencing prices, and build regression models to estimate house values accurately.

The project demonstrates an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, model building, evaluation, and comparison of multiple regression algorithms.

---

## Tools & Technologies
- **Programming Language:** Python  
- **IDE / Environment:** VS Code, Jupyter Notebook  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy  

---

## Dataset Description
The dataset contains various housing-related features such as property size, location attributes, and structural characteristics that influence house prices.  
The target variable represents the house price.

---

## Project Workflow

### 1. Data Loading & Exploration
- Loaded the dataset using Pandas
- Explored data structure, data types, and summary statistics
- Identified missing values and data inconsistencies

### 2. Data Cleaning & Preprocessing
- Handled missing values using appropriate imputation techniques
- Removed unnecessary or redundant columns
- Prepared features for regression modeling

### 3. Exploratory Data Analysis (EDA)
- Performed correlation analysis to understand feature relationships
- Visualized correlations using heatmaps
- Identified key variables influencing house prices

### 4. Model Implementation
The following regression models were implemented and compared:
- Linear Regression  
- Random Forest Regressor  
- Decision Tree Regressor  
- K-Nearest Neighbors (KNN) Regressor  

### 5. Model Evaluation
Model performance was evaluated using standard regression metrics:
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- R-Squared (R²)  

---

## Model Performance Summary

| Model | MSE | MAE | R² |
|-----|-----|-----|----|
| Linear Regression | 41,194,072,741.06 | 139,590.96 | **0.448** |
| Random Forest | 44,186,308,125.30 | 141,811.79 | 0.408 |
| KNN | 54,270,710,864.73 | 155,957.75 | 0.272 |
| Decision Tree | 73,020,339,854.87 | 184,562.71 | 0.021 |

---

## Key Insights
- Linear Regression achieved the best performance with the highest R² score and lowest error values.
- Correlation analysis revealed strong linear relationships between certain features and house prices.
- Tree-based models showed signs of overfitting, resulting in lower generalization performance.

---

## Conclusion
Linear Regression emerged as the most effective model for this dataset, indicating that house prices are largely influenced by linear relationships among features. Proper data preprocessing and exploratory analysis were critical in achieving reliable results.

---

## Future Enhancements
- Feature engineering and selection
- Hyperparameter tuning for ensemble and KNN models
- Cross-validation for robust evaluation
- Deployment using Flask or Streamlit

---

## Author
**Arun Santhosh M**  
Aspiring Data Analyst | Machine Learning Enthusiast  

---

## Acknowledgements
- Open-source Python community  
- Scikit-learn documentation  

