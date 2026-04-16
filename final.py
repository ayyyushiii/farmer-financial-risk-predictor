"""
Farmer Financial Risk Prediction Model
AI-Based Predictive Framework using Climate, Crop Yield, and Market Price Data

Fixes applied vs original:
  - Proper financial risk label engineering (not a constant column)
  - Correct handling of categorical features (LabelEncoder)
  - NaN-safe cross-validation with StratifiedKFold
  - Class imbalance handling with SMOTE + class_weight
  - Proper feature importance (only numeric/encoded features)
  - Full evaluation: AUC, Accuracy, Classification Report, Confusion Matrix
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import pickle

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH = "final_dataset.csv"          # adjust if your file is named differently
WEATHER_PATH = "final_dataset_with_weather.csv"
COST_PATH = "cost_added.csv"
MODEL_SAVE_PATH = "crop_risk_model_v2.pkl"
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("1. LOADING DATA")
print("="*50)

def load_best_dataset():
    """Try loading datasets in order of preference (richest first)."""
    candidates = [WEATHER_PATH, DATA_PATH]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {path}  →  shape: {df.shape}")
            return df
    raise FileNotFoundError(
        f"No dataset found. Tried: {candidates}\n"
        "Place final_dataset.csv or final_dataset_with_weather.csv in the project root."
    )

df = load_best_dataset()
print(df.head(3))
print("\nColumn dtypes:\n", df.dtypes.value_counts())

# ─────────────────────────────────────────────
# 2. IDENTIFY KEY COLUMNS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("2. COLUMN AUDIT")
print("="*50)
print(df.columns.tolist())

# Flexible column detection (case-insensitive, partial match)
def find_col(df, keywords, required=False):
    cols = df.columns.str.lower()
    for kw in keywords:
        matches = [c for c in df.columns if kw.lower() in c.lower()]
        if matches:
            return matches[0]
    if required:
        raise KeyError(f"Could not find a column matching any of: {keywords}")
    return None

yield_col   = find_col(df, ["production", "yield", "output"], required=True)
area_col    = find_col(df, ["area"])
price_col   = find_col(df, ["modal_price", "price", "modal"])
cost_col    = find_col(df, ["cost", "cultivation"])
crop_col    = find_col(df, ["crop"])
state_col   = find_col(df, ["state"])
district_col= find_col(df, ["district"])
year_col    = find_col(df, ["year"])
season_col  = find_col(df, ["season"])

print(f"\nDetected columns:")
print(f"  Yield/Production : {yield_col}")
print(f"  Area             : {area_col}")
print(f"  Market Price     : {price_col}")
print(f"  Cost             : {cost_col}")
print(f"  Crop             : {crop_col}")
print(f"  State            : {state_col}")
print(f"  Year             : {year_col}")

# ─────────────────────────────────────────────
# 3. MERGE COST DATA (if separate file)
# ─────────────────────────────────────────────
if cost_col is None and os.path.exists(COST_PATH):
    print(f"\nMerging cost data from {COST_PATH} ...")
    cost_df = pd.read_csv(COST_PATH)
    print("Cost file columns:", cost_df.columns.tolist())
    merge_keys = [c for c in ["state_name", "crop", "year"] if c in df.columns and c in cost_df.columns]
    if merge_keys:
        df = df.merge(cost_df, on=merge_keys, how="left")
        cost_col = find_col(df, ["cost", "cultivation"])
        print(f"  After merge shape: {df.shape}, cost column: {cost_col}")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING — FINANCIAL RISK LABEL
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("3. FINANCIAL RISK LABEL ENGINEERING")
print("="*50)

# ── 4a. Yield per hectare (proxy for actual production efficiency)
if area_col and yield_col:
    df["yield_per_hectare"] = pd.to_numeric(df[yield_col], errors="coerce") / \
                              pd.to_numeric(df[area_col], errors="coerce").replace(0, np.nan)
else:
    df["yield_per_hectare"] = pd.to_numeric(df[yield_col], errors="coerce")

# ── 4b. Revenue = yield × price (if price available)
if price_col:
    df["revenue"] = df["yield_per_hectare"] * pd.to_numeric(df[price_col], errors="coerce")
else:
    df["revenue"] = df["yield_per_hectare"]  # fallback: treat yield as revenue proxy

# ── 4c. Profit/Loss = revenue - cost  (if cost available)
if cost_col:
    df["profit"] = df["revenue"] - pd.to_numeric(df[cost_col], errors="coerce")
    label_basis = "profit"
    print("  Using profit = revenue − cost as risk basis")
else:
    # Fallback: use yield relative to crop-wise median as risk signal
    df["crop_median_yield"] = df.groupby(crop_col or state_col or [])["yield_per_hectare"] \
                                .transform("median") if crop_col else df["yield_per_hectare"].median()
    df["profit"] = df["yield_per_hectare"] - df["crop_median_yield"]
    label_basis = "relative yield vs crop median"
    print(f"  No cost data found. Using {label_basis} as risk basis")

# ── 4d. Create 3-class risk label: Low / Medium / High
# Use crop-level percentiles so the label is relative to crop type, not absolute
if crop_col:
    df["p33"] = df.groupby(crop_col)["profit"].transform(lambda x: x.quantile(0.33))
    df["p67"] = df.groupby(crop_col)["profit"].transform(lambda x: x.quantile(0.67))
else:
    df["p33"] = df["profit"].quantile(0.33)
    df["p67"] = df["profit"].quantile(0.67)

def assign_risk(row):
    if pd.isna(row["profit"]):
        return np.nan
    if row["profit"] < row["p33"]:
        return 2   # High risk
    elif row["profit"] < row["p67"]:
        return 1   # Medium risk
    else:
        return 0   # Low risk

df["financial_risk"] = df.apply(assign_risk, axis=1)

print("\nRisk label distribution:")
print(df["financial_risk"].value_counts().sort_index()
        .rename({0: "Low", 1: "Medium", 2: "High"}))
print(f"  NaN labels: {df['financial_risk'].isna().sum()}")

# Drop rows where we couldn't compute the label
df = df.dropna(subset=["financial_risk"])
df["financial_risk"] = df["financial_risk"].astype(int)

assert df["financial_risk"].nunique() > 1, (
    "ERROR: Target label has only one class — check your data!"
)

# ─────────────────────────────────────────────
# 5. FEATURE SELECTION & ENCODING
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("4. FEATURE PREPARATION")
print("="*50)

# Drop leakage columns (directly derived from the label)
DROP_COLS = ["financial_risk", "profit", "revenue", "p33", "p67",
             "crop_median_yield"]

# ── Numeric features
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features = [c for c in num_features if c not in DROP_COLS]

# ── Categorical features to encode
cat_features = []
for col in [crop_col, state_col, district_col, season_col]:
    if col and col in df.columns and col not in DROP_COLS:
        cat_features.append(col)

print(f"  Numeric features  : {len(num_features)}")
print(f"  Categorical feats : {cat_features}")

# Encode categoricals
le_dict = {}
for col in cat_features:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
    le_dict[col] = le

encoded_cats = [c + "_enc" for c in cat_features]
feature_cols = num_features + encoded_cats

# Remove any feature that is purely NaN
feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]

X = df[feature_cols].copy()
y = df["financial_risk"].copy()

# Fill remaining NaNs with column median
X = X.fillna(X.median(numeric_only=True))

print(f"\n  Final feature matrix: {X.shape}")
print(f"  Class distribution : {y.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 6. HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    min_class_count = y.value_counts().min()
    k_neighbors = min(5, min_class_count - 1)
    if k_neighbors >= 1:
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"\n  SMOTE applied: {X.shape[0]} → {X_res.shape[0]} samples")
    else:
        X_res, y_res = X, y
        print("\n  Skipped SMOTE (too few samples in minority class)")
except ImportError:
    X_res, y_res = X, y
    print("\n  imbalanced-learn not installed — skipping SMOTE")
    print("  Install with: pip install imbalanced-learn --break-system-packages")

# ─────────────────────────────────────────────
# 7. MODEL TRAINING WITH CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("5. MODEL TRAINING & CROSS-VALIDATION")
print("="*50)

# Use StratifiedKFold to preserve class ratios in each fold
n_splits = min(5, y_res.value_counts().min())   # don't exceed minority class size
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",        # handles imbalance inside the model too
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Cross-validate
print(f"\n  Running {n_splits}-fold StratifiedKFold cross-validation ...")

auc_scores, acc_scores = [], []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_res, y_res)):
    X_tr, X_val = X_res.iloc[train_idx], X_res.iloc[val_idx]
    y_tr, y_val = y_res.iloc[train_idx], y_res.iloc[val_idx]

    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_val)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    acc_scores.append(acc)

    # AUC — only if more than 1 class present in the fold
    if len(np.unique(y_val)) > 1:
        try:
            auc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")
            auc_scores.append(auc)
        except Exception:
            pass

    print(f"  Fold {fold+1}: Acc={acc:.4f}" +
          (f"  AUC={auc_scores[-1]:.4f}" if auc_scores else ""))

print(f"\n  Average Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
if auc_scores:
    print(f"  Average AUC      : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
else:
    print("  AUC could not be computed (check class distribution per fold)")

# ─────────────────────────────────────────────
# 8. FINAL MODEL — TRAIN ON ALL DATA
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("6. FINAL MODEL (full dataset)")
print("="*50)

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_model.fit(X_res, y_res)

# Save model + metadata
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump({
        "model": final_model,
        "feature_cols": feature_cols,
        "label_encoders": le_dict,
        "class_names": ["Low Risk", "Medium Risk", "High Risk"]
    }, f)
print(f"  ✅ Model saved to: {MODEL_SAVE_PATH}")

# ─────────────────────────────────────────────
# 9. EVALUATION ON FULL TRAINING SET
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("7. FULL TRAINING SET EVALUATION")
print("="*50)
y_pred_full = final_model.predict(X_res)
print(classification_report(
    y_res, y_pred_full,
    target_names=["Low Risk", "Medium Risk", "High Risk"]
))

# ─────────────────────────────────────────────
# 10. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("8. TOP FEATURE IMPORTANCES")
print("="*50)

feat_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

print(feat_df.head(15).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    x="importance", y="feature",
    data=feat_df.head(15),
    palette="Blues_r"
)
plt.title("Top 15 Feature Importances — Farmer Financial Risk Model")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("  Plot saved → feature_importance.png")

# ─────────────────────────────────────────────
# 11. CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(y_res, y_pred_full)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)
plt.title("Confusion Matrix — Financial Risk Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("  Plot saved → confusion_matrix.png")

# ─────────────────────────────────────────────
# 12. RISK DISTRIBUTION CHART
# ─────────────────────────────────────────────
plt.figure(figsize=(6, 4))
risk_counts = y.value_counts().sort_index()
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
bars = plt.bar(["Low Risk", "Medium Risk", "High Risk"],
               risk_counts.values, color=colors, edgecolor="white")
for bar, val in zip(bars, risk_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha="center", fontweight="bold")
plt.title("Financial Risk Label Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("risk_distribution.png", dpi=150)
plt.show()
print("  Plot saved → risk_distribution.png")

# ─────────────────────────────────────────────
# 13. PREDICT FUNCTION (for inference)
# ─────────────────────────────────────────────
def predict_risk(input_dict: dict, model_path=MODEL_SAVE_PATH) -> dict:
    """
    Predict financial risk for a new farmer record.

    Parameters
    ----------
    input_dict : dict
        Keys should match the original dataset columns.
        Example:
          {
            "state_name": "Karnataka",
            "district_name": "Mysuru",
            "crop": "Rice",
            "season": "Kharif",
            "area": 2.5,
            "production": 4.0,
            "modal_price": 1800,
            "year": 2023
          }

    Returns
    -------
    dict with keys: risk_label, risk_class, probabilities
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    mdl    = bundle["model"]
    cols   = bundle["feature_cols"]
    les    = bundle["label_encoders"]
    names  = bundle["class_names"]

    row = pd.DataFrame([input_dict])

    # Encode categoricals
    for orig_col, le in les.items():
        enc_col = orig_col + "_enc"
        val = str(row[orig_col].iloc[0]) if orig_col in row.columns else "Unknown"
        if val in le.classes_:
            row[enc_col] = le.transform([val])[0]
        else:
            row[enc_col] = -1   # unseen category

    # Fill missing features
    for c in cols:
        if c not in row.columns:
            row[c] = 0

    X_in = row[cols].fillna(0)
    proba = mdl.predict_proba(X_in)[0]
    pred  = mdl.predict(X_in)[0]

    return {
        "risk_class"    : int(pred),
        "risk_label"    : names[int(pred)],
        "probabilities" : {names[i]: round(float(p), 4) for i, p in enumerate(proba)}
    }


# ─────────────────────────────────────────────
# 14. DEMO PREDICTION
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("9. DEMO PREDICTION")
print("="*50)

sample = df.iloc[0].to_dict()   # use first real row as demo
result = predict_risk(sample)
print(f"  Input sample (first row in dataset)")
print(f"  → Predicted Risk : {result['risk_label']}")
print(f"  → Probabilities  : {result['probabilities']}")

print("\n" + "="*50)
print("✅ PIPELINE COMPLETE")
print("="*50)