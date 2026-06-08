import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Loading Data
df = pd.read_csv("../data/raw/train.csv")
print(df.info(), df.head(3), "\n", df.describe())

# Missing Value Analysis
threshold = 20
missing_count = df.isnull().sum()
missing_count = missing_count[missing_count > 0]
print(missing_count)

missing_percent = df.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0]
print(missing_percent)

missing_summary = pd.DataFrame(
    {"Mising Count": missing_count, "Missing Percent": missing_percent}
).sort_values("Missing Percent", ascending=True)
high_missing = missing_summary[missing_summary["Missing Percent"] > threshold]
print(missing_summary)

# Data Cleaning
df = df.drop("Id", axis=1)  # I can grab columns by index or other field names
df = df.drop(columns=high_missing.index)  # Dropping out missing values

df_nums = df.select_dtypes(include=["number"])
df_words = df.select_dtypes(exclude=["number"])

df[df_nums.columns] = df[df_nums.columns].fillna(df[df_nums.columns].median())
df[df_words.columns] = df[df_words.columns].fillna("None")

# Plot House Price Distribution
plt.figure(figsize=(10, 7))
sns.histplot(df["SalePrice"], color="green", kde=True)
plt.title("House Price Distribution")
plt.savefig("../assets/house_price_distribution.png")

# Plot Distribution for all numerical features
df_nums.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.savefig("../assets/numerical_feature_distributions.png")

# Plot histogram of missing values
plt.figure(figsize=(10, 20))
sns.histplot(missing_summary["Missing Percent"], color="gray", kde=True)
plt.title("Missing value Histogram")
plt.savefig("../assets/missing_value_hist")

# Plotting correlation heatmap
correlation = df_nums.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation)
plt.savefig("../assets/correlation_matrix.png")
correlation["SalePrice"].sort_values(ascending=False)

# Feature Engineering
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("SalePrice", axis=1)  # What model uses to predict
y = df_encoded["SalePrice"]  # What model is trying to predict

# Train / Validation Split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # splitting data into 80/20 training/test data

# Baseline Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)  # find best fit hyperplane for data
preds = linear_model.predict(X_valid)  # apply learned coefficient to validation feats
mae = mean_absolute_error(y_valid, preds)
print(f"Linear Regression MAE: {mae:,.2f}")

# Random Forests
rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_valid)
rf_mae = mean_absolute_error(y_valid, rf_preds)
print(f"Random Forest MAE: {rf_mae:,.2f}")

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_valid)
xgb_mae = mean_absolute_error(y_valid, xgb_preds)
print(f"XGBoost MAE: {xgb_mae:,.2f}")
