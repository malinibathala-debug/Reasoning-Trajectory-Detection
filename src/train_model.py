import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load preprocessed dataset
df = pd.read_csv("dataset/trajectories_preprocessed.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Define input and target
X = df["model_text"]
y = df["label"]
groups = df["group_id"]


# 3. Split data using question groups
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_groups = set(groups.iloc[train_indices])
test_groups = set(groups.iloc[test_indices])

print("\nTRAINING DATA:", len(X_train))
print("TESTING DATA:", len(X_test))
print("TRAINING GROUPS:", len(train_groups))
print("TESTING GROUPS:", len(test_groups))

print(
    "Overlapping groups:",
    len(train_groups.intersection(test_groups))
)


# 4. Build TF-IDF and Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# 5. Train the model
print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate the model
print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))

print(
    "Precision:",
    precision_score(y_test, y_pred, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, y_pred, zero_division=0)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred, zero_division=0)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 8. Save trained model
joblib.dump(model, "models/reasoning_model.pkl")

print("\nModel saved successfully!")
print("Location: models/reasoning_model.pkl")