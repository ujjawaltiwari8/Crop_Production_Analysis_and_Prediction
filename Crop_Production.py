
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the crop production dataset
df = pd.read_csv("cropproduction.csv")

# Basic information about the dataset
print("First five records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset details:")
df.info()

# Check missing and duplicate values
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

# Get a summary of numerical columns
print("\nStatistical summary:")
print(df.describe())

# Check the number of unique categories
print("\nUnique states:", df["State"].nunique())
print("Unique districts:", df["District"].nunique())
print("Unique crops:", df["Crop"].nunique())
print("Available seasons:", df["Season"].unique())


# -------------------------------
# Exploratory Data Analysis
# -------------------------------

# Distribution of crop production
plt.figure(figsize=(8, 5))
plt.hist(df["Production-2024-25"], bins=30, edgecolor="black")
plt.title("Distribution of Crop Production")
plt.xlabel("Production")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

# Relationship between numerical features
plt.figure(figsize=(8, 5))
sns.heatmap(
    df[["Area-2024-25", "Production-2024-25",
        "Yield-2024-25"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.show()

# Find the ten crops with the highest total production
top_crops = (
    df.groupby("Crop")["Production-2024-25"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
top_crops.plot(kind="bar")
plt.title("Top 10 Crops by Production")
plt.xlabel("Crop")
plt.ylabel("Total Production")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------------------
# Checking Zero Values
# -------------------------------

area = df["Area-2024-25"]
production = df["Production-2024-25"]

print("\nRecords with zero area:", (area == 0).sum())
print("Records with zero production:", (production == 0).sum())

print("\nProduction skewness:")
print(production.skew())

print("\nZero area and zero production:",
      ((area == 0) & (production == 0)).sum())

print("Zero area but positive production:",
      ((area == 0) & (production > 0)).sum())

print("Positive area but zero production:",
      ((area > 0) & (production == 0)).sum())

# Study records where production is positive
positive_data = df[production > 0]

print("\nPositive production records:")
print(positive_data.shape)

print("\nProduction summary for these records:")
print(positive_data["Production-2024-25"].describe())

# Check records with zero area and positive production
zero_area_data = df[(area == 0) & (production > 0)]

print("\nZero area but positive production records:")
print(zero_area_data[
    ["State", "District", "Crop", "Season",
     "Area-2024-25", "Production-2024-25",
     "Yield-2024-25"]
].head(10).to_string(index=False))

# Check records with positive area and zero production
zero_production_data = df[(area > 0) & (production == 0)]

print("\nPositive area but zero production records:")
print(zero_production_data[
    ["State", "District", "Crop", "Season",
     "Area-2024-25", "Production-2024-25",
     "Yield-2024-25"]
].head(10).to_string(index=False))

print("\nTotal records with positive area and zero production:")
print(len(zero_production_data))

print("\nYield summary for these records:")
print(zero_production_data["Yield-2024-25"].describe())

print("\nSample records:")
print(zero_production_data[
    ["State", "District", "Crop", "Season",
     "Area-2024-25", "Production-2024-25",
     "Yield-2024-25"]
].head(15).to_string(index=False))


# -------------------------------
# Preparing Data for Machine Learning
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Select input features and the target
X = df[
    ["State", "District", "Crop", "Season", "Area-2024-25"]
]

y = df["Production-2024-25"]

# Convert categorical columns into numerical values
categorical_columns = ["State", "District", "Crop", "Season"]
numerical_columns = ["Area-2024-25"]

preprocessor = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"),
     categorical_columns),
    ("numbers", "passthrough", numerical_columns)
])

# Divide the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -------------------------------
# Training Five Regression Models
# -------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

model_results = []

for model_name, model in models.items():

    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regression", model)
    ])

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    model_results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

# Compare the performance of all models
results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values(by="RMSE")

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

# Model Performance Comparison Graph

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# -------------------------------
# Actual vs Predicted Values
# -------------------------------

best_model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

best_model.fit(X_train, y_train)

predicted_values = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicted_values, alpha=0.5)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Crop Production")
plt.tight_layout()
plt.show()


# -------------------------------
# K-Means Clustering
# -------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Use area, production and yield for clustering
cluster_data = df[
    ["Area-2024-25", "Production-2024-25",
     "Yield-2024-25"]
].copy()

# Apply log transformation to reduce skewness
cluster_data = np.log1p(cluster_data)

# Scale the selected features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Divide the records into three clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_data)

# Display the number of records in each cluster
print("\nK-Means Clustering Results:")

print("\nRecords in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Compare the average values of each cluster
print("\nAverage values for each cluster:")
print(
    df.groupby("Cluster")[
        ["Area-2024-25", "Production-2024-25",
         "Yield-2024-25"]
    ].mean().round(3)
)

# Check how well the clusters are separated
score = silhouette_score(scaled_data, df["Cluster"])

print("\nSilhouette Score:", round(score, 3))

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Area-2024-25"],
    df["Production-2024-25"],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.5
)

plt.xlabel("Area")
plt.ylabel("Production")
plt.title("K-Means Clustering of Crop Production")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()
