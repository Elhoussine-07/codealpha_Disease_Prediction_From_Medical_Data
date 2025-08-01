import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib, json, time, logging
from src.preprocessing import clean_data, preprocess_features


datasets = "heart_disease.csv"
base_data_path = "data"


print(f"\n Training on: {datasets}")
file_path = os.path.join(base_data_path, datasets)
df = pd.read_csv(file_path)
df = clean_data(df)
X = df.drop("target", axis=1)
y = df["target"]

X = preprocess_features(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

best_model = None
best_acc = 0
results = {}

for name, clf in models.items():
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))


        results[name] = {
            "accuracy": acc,
            "training_time": end - start
        }

        if acc > best_acc:
            best_acc = acc
            best_model = clf

# S'assurer que le dossier models existe
os.makedirs("models", exist_ok=True)

model_name = datasets.replace(".csv", "") + "_best_model.pkl"
joblib.dump(best_model, f"models/{model_name}")

metrics_file = datasets.replace(".csv", "") + "_metrics.json"
with open(f"models/{metrics_file}", "w") as f:
    json.dump(results, f, indent=4)

