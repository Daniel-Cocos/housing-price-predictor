import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/raw/train.csv")
print(df.info(), df.head(3), "\n", df.describe())
df = df.drop("Id", axis=1)  # I can grab columns by index or other field names
df_num = df.select_dtypes(include = ["int64", "float64"])

# Plot House Price Distribution
plt.figure(figsize=(10, 7))
sns.histplot(df["SalePrice"], color="green", kde=True)
plt.title("House Price Distribution")
plt.savefig("../assets/house_price_distribution.png")

# Plot Distribution for all numerical features
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.savefig("../assets/hist1")

# Missing Values
threshold = 20
missing_count = df.isnull().sum()
missing_count = missing_count[missing_count > 0]
print(missing_count)

missing_percent = df.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0]
print(missing_percent)

missing_summary = pd.DataFrame({"Mising Count":missing_count, "Missing Percent":missing_percent}).sort_values("Missing Percent", ascending=True)
high_missing = missing_summary[missing_summary["Missing Percent"] > threshold]
print(missing_summary)

# Plot histogram of missing values
plt.figure(figsize=(10, 20))
sns.histplot(missing_summary["Missing Percent"], color="gray", kde=True)
plt.title("Missing value Histogram")
plt.savefig("../assets/missing_value_hist")
