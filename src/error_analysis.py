import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


# 1. Load dataset
df = pd.read_csv("dataset/trajectories_preprocessed.csv")

X = df["model_text"]
y = df["label"]
groups = df["group_id"]


# 2. Split dataset
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


# 3. Build Logistic Regression model
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


# 4. Train model
print("Training model...")
model.fit(X_train, y_train)

print("Training completed!")


# 5. Predict test data
predictions = model.predict(X_test)


# 6. Create error analysis dataframe
analysis_df = df.iloc[test_idx].copy()

analysis_df["predicted_label"] = predictions
analysis_df["actual_label"] = y_test.values

analysis_df["prediction_result"] = (
    analysis_df["actual_label"] ==
    analysis_df["predicted_label"]
)

# Select incorrectly predicted records
errors = analysis_df[
    analysis_df["prediction_result"] == False
].copy()


# 7. Display error summary
print("\n========== ERROR ANALYSIS ==========")

print("Total test samples:", len(analysis_df))
print("Correct predictions:", analysis_df["prediction_result"].sum())
print("Incorrect predictions:", len(errors))

print("\nError Types:")
print(pd.crosstab(
    errors["actual_label"],
    errors["predicted_label"]
))


# 8. Display sample errors
print("\n========== SAMPLE INCORRECT PREDICTIONS ==========")

for index, row in errors.head(10).iterrows():

    print("\nQuestion:", row["question"])
    print("Generated Reasoning:", row["generated_reasoning"])
    print("Actual Label:", row["actual_label"])
    print("Predicted Label:", row["predicted_label"])
    print("----------------------------------------")


# 9. Save error records
errors.to_csv(
    "outputs/incorrect_predictions.csv",
    index=False
)

print("\nError analysis completed!")
print("Saved to: outputs/incorrect_predictions.csv")