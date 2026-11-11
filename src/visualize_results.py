import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASET_PATH = BASE_DIR / "dataset" / "trajectories_preprocessed.csv"

OUTPUTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Load dataset
df = pd.read_csv(DATASET_PATH)

# 2. Label Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label")

plt.title("Dataset Label Distribution")
plt.xlabel("Answer Correctness (0 = Incorrect, 1 = Correct)")
plt.ylabel("Number of Trajectories")

plt.savefig(OUTPUTS_DIR / "label_distribution.png", dpi=300)
plt.close()

# 3. Model Performance Comparison
results = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")

results_melted = results.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_melted,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylim(0, 1)
plt.ylabel("Performance Score")

plt.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=300)
plt.close()

# 4. Confusion Matrix
X = df["model_text"]
y = df["label"]
groups = df["group_id"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

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

model.fit(X_train, y_train)

predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Incorrect", "Correct"],
    yticklabels=["Incorrect", "Correct"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# 5. Correct vs Incorrect Predictions
errors = pd.read_csv(OUTPUTS_DIR / "incorrect_predictions.csv")

correct_count = len(y_test) - len(errors)
incorrect_count = len(errors)

prediction_data = pd.DataFrame({
    "Prediction Result": ["Correct", "Incorrect"],
    "Count": [correct_count, incorrect_count]
})

plt.figure(figsize=(7, 5))

sns.barplot(
    data=prediction_data,
    x="Prediction Result",
    y="Count"
)

plt.title("Correct vs Incorrect Predictions")
plt.ylabel("Number of Predictions")

plt.savefig(OUTPUTS_DIR / "prediction_results.png", dpi=300)
plt.close()

print("\nAll visualizations created successfully!")

print("\nSaved files:")
for file in OUTPUTS_DIR.glob("*.png"):
    print(file.name)