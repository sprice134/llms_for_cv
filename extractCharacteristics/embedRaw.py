import os
import pandas as pd

def embed_raw_summaries(metrics_root='csvs'):
    """
    For each image subdirectory under `metrics_root`, reads the *_raw.csv,
    and writes a .txt file in that same subdirectory containing one sentence
    per object of the form:
      Object {object_id} had an {metric1} of {value1}, a {metric2} of {value2}, ...
    """
    for subdir in os.listdir(metrics_root):
        subdir_path = os.path.join(metrics_root, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Locate the raw CSV
        raw_csv = next(
            (os.path.join(subdir_path, fn)
             for fn in os.listdir(subdir_path)
             if fn.endswith('_raw.csv')),
            None
        )
        if raw_csv is None:
            print(f"No *_raw.csv in {subdir_path}, skipping.")
            continue

        # Read the raw metrics
        df = pd.read_csv(raw_csv)
        metric_cols = [c for c in df.columns if c != 'object_id']

        # Build one sentence per object
        sentences = []
        for _, row in df.iterrows():
            oid = row['object_id']
            parts = []
            for metric in metric_cols:
                val = row[metric]
                parts.append(f"a {metric} of {val:.3f}")
            sentence = f"Object {oid} had " + ", ".join(parts) + "."
            sentences.append(sentence)

        # Write to a .txt file in the same subdirectory
        out_txt = os.path.join(subdir_path, f"{subdir}_raw.txt")
        with open(out_txt, 'w') as f:
            f.write("\n".join(sentences))

        print(f"Wrote raw-object summaries → {out_txt}")

if __name__ == '__main__':
    embed_raw_summaries(metrics_root='csvs')
