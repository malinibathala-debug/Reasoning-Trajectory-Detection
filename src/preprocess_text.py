import pandas as pd
import re

# 1. Load cleaned dataset
df = pd.read_csv("dataset/trajectories_cleaned.csv")

print("Original dataset shape:", df.shape)


# 2. Combine question and generated reasoning
df["model_text"] = (
    "question: " + df["question"].astype(str)
    + " reasoning: " + df["generated_reasoning"].astype(str)
)


# 3. Convert text to lowercase
def clean_text(text):

    text = text.lower()

    # Replace multiple spaces and newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


df["model_text"] = df["model_text"].apply(clean_text)


# 4. Create question groups
group_columns = ["dataset", "split", "question"]

df["group_id"] = (
    df.groupby(group_columns, sort=False).ngroup()
)


# 5. Save preprocessed dataset
output_path = "dataset/trajectories_preprocessed.csv"

df.to_csv(output_path, index=False, encoding="utf-8")


# 6. Display results
print("\nTEXT PREPROCESSING COMPLETED!")

print("\nDataset shape:", df.shape)

print("\nNew columns:")
print(["model_text", "group_id"])

print("\nSample original question:")
print(df["question"].iloc[0])

print("\nSample generated reasoning:")
print(df["generated_reasoning"].iloc[0])

print("\nCombined preprocessed text:")
print(df["model_text"].iloc[0])

print("\nUnique question groups:", df["group_id"].nunique())

print("\nSaved successfully at:", output_path)