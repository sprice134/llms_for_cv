import os
import time
import pandas as pd
import json
from openai import OpenAI

# Load API key
with open("../openAiToken.txt", "r") as key_file:
    api_key = key_file.read().strip()
os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI(api_key=api_key)

def run_prompt(MESSAGES, MODEL='o4-mini', EFFORT=None, TEMPERATURE=0):
    """
    Returns (content, prompt_tokens, completion_tokens, total_tokens, elapsed_time)
    """
    if MODEL == 'o4-mini-low':
        MODEL = 'o4-mini'
        EFFORT = 'low'
        params = {
            "model": 'o4-mini',
            "reasoning_effort": 'low',
            "messages": MESSAGES,
        }
    else:
        params = {
            "model": MODEL,
            "messages": MESSAGES,
            "temperature": TEMPERATURE
        }
    
    start = time.perf_counter()
    completion = client.chat.completions.create(**params)
    elapsed = time.perf_counter() - start

    usage = completion.usage
    return (
        completion.choices[0].message.content.strip(),
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        elapsed
    )

def generate_and_run(
    csv_path,
    templates_path,
    images_map_path,
    csvs_root,
    metrics_path="metrics.txt",
    batch_size=500,
    models_to_run=None,
    token_limit=7_500_000
):
    # If you provide a list of models here, only those will be run;
    # leave as None or [] to run all.
    if models_to_run is None:
        models_to_run = []

    # Mapping from requested stat names to actual CSV column headers
    stat_to_col = {
        "mean": "Mean",
        "median": "Median",
        "mode": "Mode",
        "range": "Range",
        "min": "Min",
        "max": "Max",
        "standard deviation": "Std Dev",
        "std dev": "Std Dev",
        "variance": "Variance",
        "iqr": "IQR",
        "q1": "Q1",
        "q3": "Q3",
        "skewness": "Skewness",
        "kurtosis": "Kurtosis",
        "coefficient of variation": "Coeff of Variation",
        "coeff of variation": "Coeff of Variation",
        "entropy": "Entropy"
    }

    # Load experiment definitions
    df = pd.read_csv(csv_path)

    # Ensure necessary columns exist
    for col in ("prompt_tokens", "completion_tokens", "total_tokens",
                "compute_time", "response", "ground_truth", "pct_error"):
        if col not in df.columns:
            df[col] = pd.NA

    # Count total vs pending (filtered by models_to_run if provided)
    total = len(df)
    if models_to_run:
        pending = df[df['Model'].isin(models_to_run) & df['response'].isna()].shape[0]
        print(f"Total experiments: {total}  (filtering to models: {models_to_run})")
    else:
        pending = df['response'].isna().sum()
        print(f"Total experiments: {total}  (no model filter)")
    print(f"Pending experiments: {pending}")

    # Load prompt templates & image map
    with open(templates_path, 'r') as f:
        templates = json.load(f)
    with open(images_map_path, 'r') as f:
        img_map = json.load(f)

    # Start with a fresh metrics file
    open(metrics_path, "w").close()

    # Initialize token counter
    total_tokens_consumed = 0

    for idx in df.index:
        row       = df.loc[idx]
        model_val = row['Model']

        # Skip models not in our list (if list non-empty)
        if models_to_run and model_val not in models_to_run:
            continue

        # Skip if already run
        if pd.notna(row['response']):
            continue

        exp_id     = row['Exp_ID']
        data_type  = row['Data_Type']
        metric     = row['Morph_Characteristic']
        stat       = row['Statistical_Metric']
        img_id_str = str(row['Image_ID'])
        img_name   = img_map.get(img_id_str)
        if not img_name:
            print(f"Skipping {exp_id}: no folder for Image_ID {img_id_str}")
            continue

        # Determine input filename
        if data_type == "Raw/Table":
            filename = f"{img_name}_raw.csv"
        elif data_type == "Raw/Embedded":
            filename = f"{img_name}_raw.txt"
        elif data_type == "Processed/Table":
            filename = f"{img_name}_processed.csv"
        elif data_type == "Processed/Embeded":
            filename = f"{img_name}.txt"
        else:
            print(f"Skipping {exp_id}: unknown Data_Type {data_type}")
            continue

        file_path = os.path.join(csvs_root, img_name, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {exp_id}: missing file {file_path}")
            continue

        with open(file_path, 'r') as f:
            data_text = f.read().strip()

        template = templates.get(data_type)
        if not template:
            print(f"Skipping {exp_id}: no template for {data_type}")
            continue

        prompt = (
            template
            .replace("{METRIC}", metric)
            .replace("{CHARACTERISTIC}", stat)
            .replace("{DATA}", data_text)
        )
        messages = [{"role": "user", "content": prompt}]

        # Derive effort for o4-mini-X
        effort = None
        if model_val.startswith("o4-mini-"):
            effort = model_val.split("o4-mini-", 1)[1]

        # Run the LLM and collect metrics
        resp_text, p_toks, c_toks, tot_toks, elapsed = run_prompt(
            messages,
            MODEL=model_val,
            EFFORT=effort
        )

        # Update token counter and check limit
        total_tokens_consumed += tot_toks
        if total_tokens_consumed >= token_limit:
            print(f"\nToken limit reached: {total_tokens_consumed} tokens (>= {token_limit}). Stopping execution.\n")
            break

        # Parse numeric prediction
        try:
            pred_value = float(resp_text)
        except ValueError:
            pred_value = None
            print(f"Warning: could not parse '{resp_text}' as float for {exp_id}")

        # Ground-truth lookup: match Metric & mapped column header
        proc_path    = os.path.join(csvs_root, img_name, f"{img_name}_processed.csv")
        ground_truth = pd.NA
        pct_error    = pd.NA
        if pred_value is not None and os.path.exists(proc_path):
            df_proc = pd.read_csv(proc_path)
            mask = df_proc['Metric'] == metric
            col_key = stat.lower()
            col_name = stat_to_col.get(col_key)
            if col_name and col_name in df_proc.columns and mask.any():
                truth = df_proc.loc[mask, col_name].iloc[0]
                ground_truth = truth
                pct_error = abs((pred_value - truth) / truth) * 100

        # Write into dataframe
        df.at[idx, "prompt_tokens"]     = p_toks
        df.at[idx, "completion_tokens"] = c_toks
        df.at[idx, "total_tokens"]      = tot_toks
        df.at[idx, "compute_time"]      = round(elapsed, 4)
        df.at[idx, "response"]          = resp_text
        df.at[idx, "ground_truth"]      = ground_truth
        df.at[idx, "pct_error"]         = round(pct_error, 4) if pd.notna(pct_error) else pd.NA

        # Append to metrics file
        with open(metrics_path, "a") as mf:
            mf.write(
                f"{exp_id},{model_val},{p_toks},{c_toks},{tot_toks},"
                f"{elapsed:.4f},{ground_truth},{pct_error:.4f}\n"
            )

        print(
            f"{exp_id} ({model_val}): {resp_text}\n"
            f"  → tokens: prompt={p_toks}, completion={c_toks}, total={tot_toks};"
            f" time={elapsed:.2f}s\n"
            f"  → ground_truth={ground_truth}, pct_error={pct_error:.2f}%"
        )

        # Every batch_size, flush to CSV and clear metrics
        completed = df['response'].notna().sum()
        if completed and completed % batch_size == 0:
            df.to_csv(csv_path, index=False)
            open(metrics_path, "w").close()
            print(f"— Flushed {batch_size} completed to {csv_path} and cleared {metrics_path}")

    # Final flush of whatever has been processed so far
    df.to_csv(csv_path, index=False)
    print(f"Done. Flushed all processed results to {csv_path}")
    print(f"Total tokens consumed: {total_tokens_consumed}")

if __name__ == "__main__":
    generate_and_run(
        csv_path="experiments.csv",
        templates_path="questions.json",
        images_map_path="images.json",
        csvs_root="../createMetrics/csvs",
        metrics_path="metrics.txt",
        batch_size=50,
        models_to_run=[
            "gpt-4.1-mini"
        ],
        token_limit= 2_000_000
    )
