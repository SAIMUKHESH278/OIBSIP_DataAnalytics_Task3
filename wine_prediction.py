# Wine Quality Prediction - Alternative Version

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


# Load Dataset
wine = pd.read_csv("WineQT.csv")

print("Dataset loaded successfully!")

# Basic Information
print("\nShape of dataset:")
print(wine.shape)

print("\nFirst 10 rows:")
print(wine.head(10))

print("\nDataset Information:")
wine.info()

print("\nMissing Values:")
print(wine.isnull().sum())


# Remove unnecessary column if present
if "Id" in wine.columns:
    wine.drop("Id", axis=1, inplace=True)


# Statistical Summary
print("\nSummary Statistics:")
print(wine.describe())


# Duplicate Check
print("\nDuplicate Records:", wine.duplicated().sum())


# Visualization 1: Quality Distribution
plt.figure(figsize=(8,5))
sns.countplot(x="quality", data=wine)
plt.title("Distribution of Wine Quality")
plt.show()


# Visualization 2: Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(wine.corr(), annot=True, cmap="viridis")
plt.title("Feature Correlation Matrix")
plt.show()
# ======================================
# Part 2: Data Preprocessing and Model Building
# ======================================

# Separate features and target variable
X = wine.drop("quality", axis=1)
y = wine["quality"]

print("\nFeatures and Target separated successfully!")

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData Scaling Completed!")


# --------------------------------------
# Random Forest Classifier
# --------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)


# --------------------------------------
# Stochastic Gradient Descent Classifier
# --------------------------------------

sgd_model = SGDClassifier(
    max_iter=1000,
    random_state=42
)

sgd_model.fit(X_train_scaled, y_train)

sgd_predictions = sgd_model.predict(X_test_scaled)

sgd_accuracy = accuracy_score(y_test, sgd_predictions)

print("\nSGD Classifier Accuracy:")
print(sgd_accuracy)


# --------------------------------------
# Support Vector Classifier
# --------------------------------------

svc_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svc_model.fit(X_train_scaled, y_train)

svc_predictions = svc_model.predict(X_test_scaled)

svc_accuracy = accuracy_score(y_test, svc_predictions)

print("\nSupport Vector Classifier Accuracy:")
print(svc_accuracy)


# --------------------------------------
# Model Comparison
# --------------------------------------

models = [
    "Random Forest",
    "SGD Classifier",
    "SVC"
]

scores = [
    rf_accuracy,
    sgd_accuracy,
    svc_accuracy
]

plt.figure(figsize=(8, 5))

sns.barplot(
    x=models,
    y=scores
)

plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.show()
# ======================================
# Part 3: Model Evaluation and Final Analysis
# ======================================


# --------------------------------------
# Confusion Matrix - Random Forest
# --------------------------------------

plt.figure(figsize=(6,5))

sns.heatmap(
    confusion_matrix(y_test, rf_predictions),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# --------------------------------------
# Classification Reports
# --------------------------------------

print("\n==============================")
print("Random Forest Classification Report")
print("==============================")

print(classification_report(y_test, rf_predictions))


print("\n==============================")
print("SGD Classifier Classification Report")
print("==============================")

print(classification_report(y_test, sgd_predictions))


print("\n==============================")
print("Support Vector Classifier Classification Report")
print("==============================")

print(classification_report(y_test, svc_predictions))


# --------------------------------------
# Feature Importance (Random Forest)
# --------------------------------------

importance = rf_model.feature_importances_

feature_names = X.columns

feature_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_data = feature_data.sort_values(
    by="Importance",
    ascending=False
)


plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_data
)

plt.title("Important Features Affecting Wine Quality")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()


# --------------------------------------
# Best Model Selection
# --------------------------------------

accuracy_results = {
    "Random Forest": rf_accuracy,
    "SGD Classifier": sgd_accuracy,
    "Support Vector Classifier": svc_accuracy
}

best_model = max(
    accuracy_results,
    key=accuracy_results.get
)

print("\n=================================")
print("FINAL RESULT")
print("=================================")

for model, score in accuracy_results.items():
    print(f"{model}: {score:.4f}")

print("\nBest Performing Model:")
print(best_model)

print("\nWine Quality Prediction Analysis Completed Successfully!")