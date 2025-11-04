import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve
)
import joblib

DATA_PATH = "C:\\Users\\Farahan\\Downloads\\data.csv.csv" 
df = pd.read_csv(DATA_PATH)

display(df.head())

display(df.info())
display(df.describe(include='all'))
dup_count = df.duplicated().sum()
print(f"\nRemoved {dup_count} duplicate rows.")
df = df.drop_duplicates()

missing_summary = df.isnull().sum()
display(missing_summary[missing_summary > 0])

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().str.lower()

numeric_cols_all = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols_all:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nOutliers handled via IQR method.")
display(df.describe())

possible_targets = [
    'failure', 'fail', 'machine_failure', 'Machine failure',
    'MachineFailure', 'target', 'label', 'is_failed', 'broken'
]

target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    cols_lower = {c.lower(): c for c in df.columns}
    for t in possible_targets:
        if t.lower() in cols_lower:
            target_col = cols_lower[t.lower()]
            break
if target_col is None:
    target_col = df.columns[-1]

print("\nUsing target column:", target_col)

y = df[target_col].copy()

if y.dtype == 'object':
    pos_tokens = ['fail', 'failed', 'yes', 'true', '1', 'y', 'broken']
    neg_tokens = ['no', 'ok', 'normal', 'false', '0', 'n', 'working']
    mapping = {}
    for uv in y.dropna().unique():
        s = str(uv).strip().lower()
        if any(tok in s for tok in pos_tokens):
            mapping[uv] = 1
        elif any(tok in s for tok in neg_tokens):
            mapping[uv] = 0
    if mapping:
        y = y.map(mapping)
    else:
        rare = y.value_counts().idxmin()
        y = (y == rare).astype(int)

if pd.api.types.is_numeric_dtype(y) and set(y.unique()) - {0, 1}:
    uniq = sorted(y.dropna().unique())
    if len(uniq) == 2:
        y = y.map({uniq[0]: 0, uniq[1]: 1})
    else:
        thresh = np.percentile(y.dropna(), 90)
        y = (y >= thresh).astype(int)

X = df.drop(columns=[target_col]).copy()

id_like = [
    c for c in X.columns
    if 'id' in c.lower() or c.lower().startswith('product') or c.lower().startswith('machine')
]
X = X.drop(columns=id_like, errors='ignore')

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"\nNumeric features: {numeric_cols}")
print(f"Categorical features: {cat_cols}")

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', cat_pipeline, cat_cols)
], remainder='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=(y if y.nunique() > 1 and y.value_counts().min() >= 2 else None)
)

clf = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf.named_steps['clf'], 'predict_proba') else None

print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
if y_proba is not None and len(y_test.unique()) == 2:
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

if y_proba is not None and len(y_test.unique()) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

try:
    ohe_features = []
    if cat_cols:
        ohe_features = clf.named_steps['pre'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    feature_names = numeric_cols + list(ohe_features)
    importances = clf.named_steps['clf'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, hue='Feature', palette='viridis', legend=False)
    plt.title("Top 15 Feature Importances")
    plt.show()
except Exception as e:
    print("Feature importance plot skipped due to:", e)

joblib.dump(clf, "model.pkl")
print("\n Model saved as model.pkl")
