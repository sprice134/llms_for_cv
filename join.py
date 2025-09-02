import pandas as pd

# Paths to your two CSVs
doe_path         = "/home/sprice/AIM2025/attemptV2/doe.csv"
experiments_path = "/home/sprice/AIM2025/experiments/experiments.csv"
output_path      = "/home/sprice/AIM2025/attemptV2/experiments.csv"

# Load
doe_df         = pd.read_csv(doe_path)
experiments_df = pd.read_csv(experiments_path)

# Define the key columns to match on
key_cols = [
    "Model",
    "Data_Type",
    "Context",
    "Image_ID",
    "Morph_Characteristic",
    "Statistical_Metric"
]

# The metric columns we want to pull in
metric_cols = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "compute_time",
    "response",
    "ground_truth",
    "pct_error"
]

# Perform a left merge: keep all rows in doe_df, attach metrics where they exist
merged = doe_df.merge(
    experiments_df[key_cols + metric_cols],
    on=key_cols,
    how="left"
)

# Save out
merged.to_csv(output_path, index=False)
print(f"Wrote enriched DOE file with metrics to {output_path}")
