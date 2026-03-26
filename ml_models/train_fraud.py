"""
train_fraud.py
Full GradientBoosting fraud detection pipeline on carclaims.csv (15k rows).
Architecture: single sklearn Pipeline (ColumnTransformer + GradientBoostingClassifier)
Saved as one joblib file — no separate preprocessor.
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless save
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score

import shap

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot")
CSV_PATH     = BASE_DIR / "Insurance_Fraud_Detection-main" / "carclaims.csv"
MODEL_PATH   = BASE_DIR / "ml_models" / "fraud_model.pkl"
SHAP_PATH    = BASE_DIR / "ml_models" / "fraud_shap.png"

# ─── STEP 1: LOAD & CLEAN ─────────────────────────────────────────────────────
print("\n=== STEP 1: Load & Clean ===")
df = pd.read_csv(CSV_PATH)

# Drop irrelevant / high-cardinality identifiers
DROP_COLS = ["PolicyNumber", "RepNumber"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Handle any missing values if they sneak in
df = df.fillna("Unknown")

# Map target
df["FraudFound"] = df["FraudFound"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["FraudFound"])
df["FraudFound"] = df["FraudFound"].astype(int)

print(f"Shape after cleaning: {df.shape}")
print("Class distribution:")
print(df["FraudFound"].value_counts().rename({1: "fraud (1)", 0: "legit (0)"}))

# ─── STEP 2: FEATURE ENGINEERING ─────────────────────────────────────────────
print("\n=== STEP 2: Feature Engineering ===")
X = df.drop(columns=["FraudFound"])
y = df["FraudFound"]

# We use df dtypes to separate (Unknown fillna turns everything to object if there was a mix, but carclaims is clean)
# So let's just infer from the dataframe directly for numericals that are int64/float64
numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])

# ─── STEP 3: TRAIN ────────────────────────────────────────────────────────────
print("\n=== STEP 3: Training GradientBoostingClassifier ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

gbc = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
)

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", gbc),
])

# Fit with sample weights for class imbalance
sample_weights = compute_sample_weight("balanced", y_train)
full_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
print("Training complete.")

# ─── STEP 4: EVALUATE ─────────────────────────────────────────────────────────
print("\n=== STEP 4: Evaluation ===")
y_pred       = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# SHAP Analysis
print("\n--- SHAP Feature Importance ---")
X_train_transformed = full_pipeline.named_steps["preprocessor"].transform(X_train)

# Get feature names from the fitted preprocessor
ohe_feature_names = (
    full_pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
    .tolist()
)
all_feature_names = numerical_cols + ohe_feature_names

explainer   = shap.TreeExplainer(full_pipeline.named_steps["classifier"])
shap_values = explainer.shap_values(X_train_transformed)

# Mean absolute SHAP per feature
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df   = pd.DataFrame({"feature": all_feature_names, "shap_importance": mean_shap})
shap_df   = shap_df.sort_values("shap_importance", ascending=False)

print("\nTop 10 Most Important Features (SHAP):")
print(shap_df.head(10).to_string(index=False))

# Save SHAP bar chart
fig, ax = plt.subplots(figsize=(10, 6))
top10 = shap_df.head(10)
ax.barh(top10["feature"][::-1], top10["shap_importance"][::-1], color="#E84855")
ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
ax.set_title("Top 10 Features — GradientBoosting Fraud Model (15k)", fontsize=14, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(str(SHAP_PATH), dpi=150)
plt.close()
print(f"\nSHAP chart saved to {SHAP_PATH}")

# ─── STEP 5: SAVE ─────────────────────────────────────────────────────────────
print("\n=== STEP 5: Saving Pipeline ===")
joblib.dump(full_pipeline, str(MODEL_PATH))
print(f"Pipeline saved to {MODEL_PATH}")
print("All done.")
