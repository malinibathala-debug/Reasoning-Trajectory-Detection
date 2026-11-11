import pandas as pd
import ast

# 1. Load original dataset
df = pd.read_csv("dataset/trajectories.csv")

print("Original dataset shape:", df.shape)

# 2. Remove records with missing reasoning
df = df.dropna(subset=["generated_reasoning"])

# 3. Remove empty reasoning text
df["generated_reasoning"] = df["generated_reasoning"].astype(str).str.strip()

df = df[df["generated_reasoning"] != ""]

# 4. Clean model names
def extract_model_name(value):

    try:
        model_info = ast.literal_eval(value)

        if isinstance(model_info, dict):
            return model_info.get("name", "Unknown")

        return str(model_info)

    except (ValueError, SyntaxError):
        return str(value)


df["model_name"] = df["model_name"].apply(extract_model_name)

# 5. Clean text columns
text_columns = [
    "question",
    "human_reasoning",
    "generated_reasoning",
    "final_answer"
]

for column in text_columns:
    df[column] = df[column].fillna("")

# 6. Reset index
df = df.reset_index(drop=True)

# 7. Save cleaned dataset
df.to_csv(
    "dataset/trajectories_cleaned.csv",
    index=False,
    encoding="utf-8"
)

# 8. Display results
print("\nCLEANING COMPLETED!")

print("\nCleaned dataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nModel names:")
print(df["model_name"].unique())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nCleaned dataset saved successfully!")