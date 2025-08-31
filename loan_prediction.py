# ========================================
# 🏦 Loan Approval Prediction - ML Project
# ========================================

# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import joblib

# ========================================
# 📌 Step 2: Load Dataset
# ========================================
df = pd.read_csv("loan_prediction.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# ========================================
# 📌 Step 3: Explore Data
# ========================================
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Distribution of target variable
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Status Distribution")
plt.show()

# Extra EDA: Income, Loan Amount, Credit History
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
sns.histplot(df["ApplicantIncome"], kde=True)
plt.title("Applicant Income Distribution")

plt.subplot(1,3,2)
sns.histplot(df["LoanAmount"], kde=True)
plt.title("Loan Amount Distribution")

plt.subplot(1,3,3)
sns.countplot(x="Credit_History", data=df)
plt.title("Credit History Distribution")
plt.show()

# ========================================
# 📌 Step 4: Define Features & Target
# ========================================
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})   # Encode target

# Identify numeric & categorical columns
num_features = X.select_dtypes(include=[np.number]).columns
cat_features = X.select_dtypes(exclude=[np.number]).columns

# Preprocessing pipelines
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# ========================================
# 📌 Step 5: Train-Test Split
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================
# 📌 Step 6: Try Multiple Models
# ========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n🔹 {name} Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]))

# ========================================
# 📌 Step 7: Hyperparameter Tuning (Random Forest)
# ========================================
rf_params = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [4, 6, 8, None],
    "classifier__min_samples_split": [2, 5, 10]
}

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

grid = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

print("\n✅ Best Random Forest Params:", grid.best_params_)
print("Best ROC-AUC Score:", grid.best_score_)

best_model = grid.best_estimator_

# ========================================
# 📌 Step 8: Evaluate Best Model
# ========================================
y_pred = best_model.predict(X_test)
print("\n🔹 Final Model Performance (Tuned RF):")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

# ========================================
# 📌 Step 9: Feature Importance
# ========================================
ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
cat_names = ohe.get_feature_names_out(cat_features)
all_features = np.concatenate([num_features, cat_names])

importances = best_model.named_steps["classifier"].feature_importances_
feat_importances = pd.Series(importances, index=all_features).sort_values(ascending=False)

feat_importances.head(10).plot(kind="barh", figsize=(8,6))
plt.title("Top 10 Important Features")
plt.show()

# ========================================
# 📌 Step 10: Save & Load Model
# ========================================
joblib.dump(best_model, "loan_model.pkl")
print("✅ Model saved as loan_model.pkl")

loaded_model = joblib.load("loan_model.pkl")
print("🔁 Model reloaded successfully!")

# ========================================
# 📌 Step 11: Sample Prediction
# ========================================
sample = X_test.iloc[[0]]
print("\n🔍 Sample Applicant Data:\n", sample)

prediction = loaded_model.predict(sample)[0]
print("✅ Loan Approval Prediction:", "Approved" if prediction==1 else "Rejected")
