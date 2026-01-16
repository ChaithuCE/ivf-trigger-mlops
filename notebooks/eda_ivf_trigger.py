import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- Paths and data loading --------
PROJECT_ROOT = r"C:\AI_IVF_Trigger_day"  # project base folder
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ivf_trigger_preprocessed.csv")

# Load preprocessed IVF trigger dataset
df = pd.read_csv(CSV_PATH)

# -------- Quick structure checks --------
print("Shape (rows, columns):", df.shape)          # dataset size
print("\nColumn dtypes:\n", df.dtypes)             # data types per column
print("\nSample rows:\n", df.head())               # first few records

# -------- Target distribution --------
target_counts = df["trigger_recommended"].value_counts(normalize=True)
print("\nTrigger distribution (proportion):\n", target_counts)

plt.figure(figsize=(4, 4))
sns.countplot(x="trigger_recommended", data=df)
plt.title("Trigger Recommended (0/1) counts")
plt.show()

# -------- Summary statistics for numeric features --------
numeric_cols = [
    "age",
    "amh_ng_ml",
    "day",
    "avg_follicle_size_mm",
    "follicle_count",
    "estradiol_pg_ml",
    "progesterone_ng_ml",
]
print("\nNumeric summary:\n", df[numeric_cols].describe())

# -------- Histograms for numeric features --------
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# -------- Boxplots vs target (to see separation) --------
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x="trigger_recommended", y=col, data=df)
    plt.title(f"{col} by trigger_recommended")
plt.tight_layout()
plt.show()

# -------- Correlation matrix --------
plt.figure(figsize=(8, 6))
corr = df[numeric_cols + ["trigger_recommended"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix (numeric features + target)")
plt.show()



from sklearn.model_selection import GroupShuffleSplit

groups = df["patient_id"]
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=groups))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# -------- Feature and target definition --------
feature_cols = [
    "age",
    "amh_ng_ml",
    "day",
    "avg_follicle_size_mm",
    "follicle_count",
    "estradiol_pg_ml",
    "progesterone_ng_ml",
]

X_train = train_df[feature_cols]
y_train = train_df["trigger_recommended"]

X_test = test_df[feature_cols]
y_test = test_df["trigger_recommended"]


"""#### Train a baseline logistic regression ####
"""
# -------- Scale features (important for LR) --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- Class-weighted logistic regression --------
log_reg = LogisticRegression(
    class_weight="balanced",  # handle strong class imbalance
    max_iter=1000,
    random_state=42,
)
log_reg.fit(X_train_scaled, y_train)

# -------- Evaluation on test set --------
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred = log_reg.predict(X_test_scaled)

auc = roc_auc_score(y_test, y_prob)
print("Test AUC:", round(auc, 3))

print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

 
from sklearn.ensemble import RandomForestClassifier

# -------- Random Forest baseline --------

# Use same features and train/test split as before
feature_cols = [
    "age",
    "amh_ng_ml",
    "day",
    "avg_follicle_size_mm",
    "follicle_count",
    "estradiol_pg_ml",
    "progesterone_ng_ml",
]

X_train = train_df[feature_cols]
y_train = train_df["trigger_recommended"]

X_test = test_df[feature_cols]
y_test = test_df["trigger_recommended"]

# Class‑weighted RF to handle imbalance
rf = RandomForestClassifier(
    n_estimators=300,           # number of trees
    max_depth=None,            # let trees grow deep; can tune later
    min_samples_leaf=3,        # avoid overfitting tiny leaves
    class_weight="balanced",   # up‑weight rare trigger=1
    random_state=42,
    n_jobs=-1,                 # use all cores
)

rf.fit(X_train, y_train)

# Predictions and metrics
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict(X_test)

auc_rf = roc_auc_score(y_test, y_prob_rf)
print("Random Forest - Test AUC:", round(auc_rf, 3))

print("\nRandom Forest - classification report:\n",
      classification_report(y_test, y_pred_rf))
print("\nRandom Forest - confusion matrix:\n",
      confusion_matrix(y_test, y_pred_rf))

# Optional: feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nRandom Forest - feature importances:\n", importances)


from sklearn.ensemble import GradientBoostingClassifier

# -------- Gradient Boosting baseline --------

feature_cols = [
    "age",
    "amh_ng_ml",
    "day",
    "avg_follicle_size_mm",
    "follicle_count",
    "estradiol_pg_ml",
    "progesterone_ng_ml",
]

X_train = train_df[feature_cols]
y_train = train_df["trigger_recommended"]

X_test = test_df[feature_cols]
y_test = test_df["trigger_recommended"]

# Basic GBDT config; can tune later
gb = GradientBoostingClassifier(
    n_estimators=300,      # number of boosting stages
    learning_rate=0.05,   # shrinkage; lower is safer
    max_depth=3,          # depth of individual trees
    random_state=42,
)

gb.fit(X_train, y_train)

y_prob_gb = gb.predict_proba(X_test)[:, 1]
y_pred_gb = gb.predict(X_test)

auc_gb = roc_auc_score(y_test, y_prob_gb)
print("Gradient Boosting - Test AUC:", round(auc_gb, 3))

print("\nGradient Boosting - classification report:\n",
      classification_report(y_test, y_pred_gb))
print("\nGradient Boosting - confusion matrix:\n",
      confusion_matrix(y_test, y_pred_gb))

