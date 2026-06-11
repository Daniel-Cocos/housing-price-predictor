# Housing Price Predictor

Machine learning project that investigates the key factors influencing house prices, applying full data-science workflow from exploratory analysis and data cleaning through to training, comparing, and interpreting three regression models.

---

## Features

- Exploratory Data Analysis (EDA) with visualisations
- Missing-value analysis and threshold-based removal
- Data cleaning and median/mode imputation
- Outlier detection via distribution inspection
- One-hot encoding for categorical features
- Model training: Linear Regression, Random Forest, XGBoost
- Feature importance comparison across all three models
- SHAP (SHapley Additive exPlanations) model interpretability
- 5-fold cross-validation for robust performance evaluation
- Side-by-side model performance visualisation

---

## Technologies Used

| Category | Libraries / Tools |
|---|---|
| Language | Python 3 |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (LinearRegression, RandomForestRegressor) |
| Gradient Boosting | XGBoost |
| Interpretability | SHAP |
| Environment | Jupyter Notebook |

---

## Dataset

- Dataset is **not** included in the repository
- [Download from Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place the files inside `data/raw/`

```
data/raw/
├── data_description.txt
├── sample_submission.csv
├── test.csv
└── train.csv
```

---

## Project Structure

```
assets/
├── correlation_matrix.png
├── house_price_distribution.png
├── numerical_feature_distributions.png
└── validation_model_comparison.png
data/
└── raw
    ├── data_description.txt
    ├── sample_submission.csv
    ├── test.csv
    └── train.csv
docs/
└── report.md
src/
├── main.ipynb
└── main.py
```

---

## Analysis Workflow

### 1. Initial Exploration
Examine dataset shape, data types, and summary statistics to understand the raw data.

### 2. Missing Value Analysis
Identify columns with missing values and calculate their percentage. Features with **more than 20% missing data** are flagged for removal to avoid introducing excessive uncertainty into the models.

### 3. Data Cleaning & Imputation
- Drop the `Id` column as it is not needed
- Remove features exceeding the 20% missing-value threshold
- Impute missing **numeric** values with the **median** (robust to outliers)
- Impute missing **categorical** values with `"None"` which indicates missing features: e.g. no garage, no pool

### 4. Visualisation
- **House price distribution**: understand target skewness
- **Numerical feature histograms**: detect outliers and skewed distributions
- **Correlation matrix heatmap**: identify relationships with `SalePrice`

### 5. Feature Engineering
Categorical variables are one-hot encoded (`drop_first=True`) to avoid multicollinearity before model training.

### 6. Model Training
Three models are trained on an 80/20 train-validation split (`random_state=50`):

| Model | Configuration |
|---|---|
| Linear Regression | Default sklearn settings |
| Random Forest | 1 000 estimators, `n_jobs=-1` |
| XGBoost | 1 000 estimators, `learning_rate=0.05`, `max_depth=6` |

### 7. Feature Importance
Top-20 features extracted from each model using their native importance metric:
- **Linear Regression**: absolute coefficient values
- **Random Forest**: mean decrease in impurity
- **XGBoost**: gain-based importance

### 8. SHAP Analysis
SHAP values are computed for the XGBoost model to explain individual predictions and provide a global view of feature impact via bar and beeswarm plots.

### 9. Cross-Validation
5-fold cross-validation is run on all three models to provide a robust, performance estimate alongside holdout validation MAE.

---

## Results

Model performance is evaluated using **Mean Absolute Error (MAE)**: the average absolute difference between predicted and actual sale prices in dollars. Lower is better.

### Key Findings

- **XGBoost** achieves the lowest MAE on both holdout and cross-validation splits, making it the best-performing model for this dataset.
- **Random Forest** outperforms Linear Regression and produces stable cross-validation results.
- **Linear Regression** serves as an interpretable baseline but is sensitive to rare one-hot encoded categories, leading to inflated coefficients for infrequent feature values.

### Feature Importance Observations

- **Linear Regression** weights rare categorical dummies very highly (e.g. `Condition2_PosN`, `RoofMatl_WdShngl`), a sign of sensitivity to low-frequency categories.
- **Random Forest** ranks `OverallQual` as the dominant feature (importance ≈ 0.57), followed by continuous size features (`GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`) and garage metrics.
- **XGBoost** shows a more balanced importance distribution (`OverallQual` ≈ 0.39) and ranks select categorical features higher than Random Forest, benefiting from gradient boosting regularisation.

---

## Running the project

1. Clone the repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place the files in `data/raw/`
3. Download the dependencies
```zsh
poetry install
```
4. Launch Jupyter and open `src/main.ipynb`
```zsh
jupyter lab src/main.ipynb
```

---

## Further Reading

- [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)

