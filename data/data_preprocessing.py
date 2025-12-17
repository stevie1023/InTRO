from datasets import load_dataset
import json

def extract_answer_math(s):
        ans = s.split("boxed")
        if len(ans) == 1:
            return s
        ans = ans[-1]
        if len(ans) == 0:
            return ""
        try:
            if ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
        except:
            return ""
        return a
from datasets import load_dataset
import json

# List of configuration names
config_names = ["algebra", "counting_and_probability", "geometry","intermediate_algebra","number_theory","prealgebra","precalculus"]  # Replace with actual config names

# Initialize empty lists for train and test splits
train_data = []
test_data = []
train_data_ans=[]
# Load and concatenate train and test splits for each configuration
for config_name in config_names:
    ds = load_dataset("hendrycks_math", config_name)

    train_data.extend([
        example for example in ds["train"] ])  # Filter train split
    train_data.extend([example for example in ds["test"] ])    # Filter test split


# Prepare the data for saving
train_output = []
test_output = []
import re

for example in train_data:
    if example["level"].startswith('Level ') and example["level"].split('Level ')[1].isdigit() and int(example["level"].split('Level ')[1]) >= 3:

        ans = extract_answer_math(example["solution"])
        if ans == "":
            print(f"Empty answer for example: {example}")
            continue

        # steps = re.split(r'[\.\?!]', example["solution"])
        train_output.append(
            {
                "problem": example["problem"],
                "solution": example["solution"], 
                "answer": ans,
            }
        )
    elif example["level"] != "Level 1" and example["level"] != "Level 2":
        print(f"Invalid level: {example['level']}")


# Save the concatenated train and test splits to JSON files
json.dump(train_output, open("save_path", "w"))
json.dump(test_output, open("save_path", "w"))