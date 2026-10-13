"""
Forensic Glass Type Classification
----------------------------------
Predict the type of a glass fragment (window, vehicle, container, tableware,
headlamp) from its refractive index and chemical composition.

Run:  python glass_classifier.py
Out:  figures/*.png, results.txt, glass_model.joblib
"""
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
CSV = os.path.join(HERE, "glass.csv")
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/fgl.csv"

LABELS = {"WinF": "Window (float)", "WinNF": "Window (non-float)", "Veh": "Vehicle window",
          "Con": "Container", "Tabl": "Tableware", "Head": "Headlamp"}
out = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.append(s)

# ------------------------------------------------------------------ 1. Load
if not os.path.exists(CSV):
    urllib.request.urlretrieve(URL, CSV)
df = pd.read_csv(CSV)
log("Shape:", df.shape)
log(df.head(), "\n")
log(df.describe().round(3), "\n")

# ------------------------------------------------------------------ 2. Preprocessing
df = df.drop(columns=["rownames"])                      # row index, not a feature
log("Missing values:", int(df.isna().sum().sum()))
log("Duplicate rows:", int(df.duplicated().sum()))
df = df.drop_duplicates().reset_index(drop=True)

features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
le = LabelEncoder()
y = le.fit_transform(df["type"])
X = df[features]
log("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))
log(df["type"].value_counts().to_string(), "\n")

# ------------------------------------------------------------------ 3. EDA plots
fig, ax = plt.subplots(figsize=(7, 4))
order = df["type"].value_counts().index
sns.countplot(data=df, x="type", order=order, hue="type", palette="viridis", legend=False, ax=ax)
ax.set_title("Class distribution (imbalanced)"); ax.set_xlabel("Glass type"); ax.set_ylabel("Samples")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
fig.tight_layout(); fig.savefig(f"{FIG}/01_class_distribution.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature correlation matrix")
fig.tight_layout(); fig.savefig(f"{FIG}/02_correlation_heatmap.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(3, 3, figsize=(11, 8))
for a, f in zip(axes.ravel(), features):
    sns.boxplot(data=df, x="type", y=f, order=order, hue="type", palette="viridis", legend=False, ax=a)
    a.set_xlabel(""); a.set_title(f)
fig.suptitle("Feature distributions by glass type", y=1.0)
fig.tight_layout(); fig.savefig(f"{FIG}/03_feature_boxplots.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 4. Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
log(f"Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}\n")

# ------------------------------------------------------------------ 5. Compare models (5-fold CV on training set only)
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "KNN":                 Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
    "SVM (RBF)":           Pipeline([("sc", StandardScaler()), ("m", SVC(class_weight="balanced", random_state=SEED))]),
    "Gradient Boosting":   Pipeline([("sc", StandardScaler()), ("m", GradientBoostingClassifier(random_state=SEED))]),
    "Random Forest":       Pipeline([("sc", StandardScaler()), ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_res = {}
for name, pipe in models.items():
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_res[name] = s
    log(f"{name:22s} CV macro-F1 = {s.mean():.3f} +/- {s.std():.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
names = list(cv_res); means = [cv_res[n].mean() for n in names]; stds = [cv_res[n].std() for n in names]
ax.barh(names, means, xerr=stds, color=sns.color_palette("viridis", len(names)), capsize=4)
ax.set_xlabel("5-fold CV macro-F1 (training set)"); ax.set_title("Model comparison")
fig.tight_layout(); fig.savefig(f"{FIG}/04_model_comparison.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 6. Tune Random Forest
best_name = max(cv_res, key=lambda n: cv_res[n].mean())
log(f"\nBest in CV: {best_name}. Tuning Random Forest (chosen algorithm)...")
rf_pipe = Pipeline([("sc", StandardScaler()),
                    ("m", RandomForestClassifier(class_weight="balanced", random_state=SEED))])
grid = {"m__n_estimators": [200, 400], "m__max_depth": [None, 8, 15],
        "m__min_samples_leaf": [1, 2], "m__max_features": ["sqrt", 0.5]}
gs = GridSearchCV(rf_pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
gs.fit(X_tr, y_tr)
model = gs.best_estimator_
log("Best params:", gs.best_params_)
log(f"Best CV macro-F1: {gs.best_score_:.3f}\n")

# ------------------------------------------------------------------ 7. Evaluate on held-out test set
pred = model.predict(X_te)
acc = accuracy_score(y_te, pred)
log("=== TEST SET RESULTS ===")
log(f"Accuracy          : {acc:.3f}")
log(f"Precision (macro) : {precision_score(y_te, pred, average='macro', zero_division=0):.3f}")
log(f"Recall (macro)    : {recall_score(y_te, pred, average='macro', zero_division=0):.3f}")
log(f"F1 (macro)        : {f1_score(y_te, pred, average='macro', zero_division=0):.3f}")
log(f"F1 (weighted)     : {f1_score(y_te, pred, average='weighted', zero_division=0):.3f}\n")
log(classification_report(y_te, pred, target_names=le.classes_, zero_division=0))

cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix (test set)")
fig.tight_layout(); fig.savefig(f"{FIG}/05_confusion_matrix.png", dpi=150); plt.close(fig)

imp = pd.Series(model.named_steps["m"].feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot.barh(color="teal", ax=ax); ax.set_title("Random Forest feature importance")
ax.set_xlabel("Mean decrease in impurity")
fig.tight_layout(); fig.savefig(f"{FIG}/06_feature_importance.png", dpi=150); plt.close(fig)
log("Feature importance:\n" + imp.sort_values(ascending=False).round(3).to_string())

# per-class F1 plot
f1s = f1_score(y_te, pred, average=None, labels=range(len(le.classes_)), zero_division=0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(le.classes_, f1s, color=sns.color_palette("viridis", len(f1s)))
ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score"); ax.set_title("Per-class F1 (test set)")
for i, v in enumerate(f1s): ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
fig.tight_layout(); fig.savefig(f"{FIG}/07_per_class_f1.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 8. Save
joblib.dump({"model": model, "label_encoder": le, "features": features}, os.path.join(HERE, "glass_model.joblib"))
with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write("\n".join(out))
log("\nSaved model, figures and results.txt")
