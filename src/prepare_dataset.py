import json
import pandas as pd

# 1. Load the original JSON dataset
input_path = r"ThoughtSource\notebooks\internal_documentation\paper_2\ts_100_None.json"

with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)


# 2. Select non-medical datasets
selected_datasets = [
    "commonsense_qa",
    "open_book_qa",
    "strategy_qa",
    "worldtree"
]

rows = []

# 3. Extract generated reasoning trajectories
for dataset_name in selected_datasets:

    for split_name, records in data[dataset_name].items():

        for record in records:

            question = record.get("question", "")

            # Extract human reasoning safely
            human_cot = record.get("cot", [])

            if isinstance(human_cot, list):
                human_reasoning = " ".join(str(x) for x in human_cot)
            else:
                human_reasoning = str(human_cot)

            # Process AI-generated trajectories
            for trajectory in record.get("generated_cot", []):

                reasoning = trajectory.get("cot", "")

                # Extract model information safely
                model_info = trajectory.get("model", {})

                if isinstance(model_info, dict):
                    model_name = model_info.get("name", "Unknown")
                else:
                    model_name = str(model_info)

                api_service = trajectory.get(
                    "api_service", "Unknown"
                )

                answers = trajectory.get("answers", [])

                for answer in answers:

                    correct = answer.get("correct_answer")

                    if correct is True:
                        label = 1

                    elif correct is False:
                        label = 0

                    else:
                        continue

                    rows.append({
                        "dataset": dataset_name,
                        "split": split_name,
                        "question": question,
                        "human_reasoning": human_reasoning,
                        "generated_reasoning": reasoning,
                        "final_answer": answer.get("answer", ""),
                        "model_name": model_name,
                        "api_service": api_service,
                        "label": label
                    })


# 4. Convert data into a DataFrame
df = pd.DataFrame(rows)


# 5. Save the CSV file
output_path = r"dataset\trajectories.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)


# 6. Display dataset information
print("\n================================")
print("DATASET CREATED SUCCESSFULLY!")
print("================================")

print("\nTotal records:", len(df))

print("\nDataset distribution:")
print(df["dataset"].value_counts())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst five records:")
print(df.head())

print("\nCSV saved at:", output_path)