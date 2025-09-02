import os
import pandas as pd

def embed_detailed_summaries(metrics_root='csvs'):
    """
    For each image subdirectory under `metrics_root`, reads the *_processed.csv,
    and writes a .txt file in that same subdirectory containing one paragraph
    per metric. Each paragraph lists all statistic sentences for that metric.
    """
    for subdir in os.listdir(metrics_root):
        subdir_path = os.path.join(metrics_root, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Locate the processed CSV
        proc_csv = next(
            (os.path.join(subdir_path, fn)
             for fn in os.listdir(subdir_path)
             if fn.endswith('_processed.csv')),
            None
        )
        if proc_csv is None:
            print(f"No *_processed.csv in {subdir_path}, skipping.")
            continue

        # Read the processed stats
        df = pd.read_csv(proc_csv)
        stat_cols = [c for c in df.columns if c != 'Metric']

        # Build one paragraph per metric
        paragraphs = []
        for _, row in df.iterrows():
            metric = row['Metric']
            sentences = [
                f"The {stat} of {metric} was {row[stat]:.3f}."
                for stat in stat_cols
            ]
            paragraph = " ".join(sentences)
            paragraphs.append(paragraph)

        # Write to a .txt file in the same subdirectory
        out_txt = os.path.join(subdir_path, f"{subdir}.txt")
        with open(out_txt, 'w') as f:
            f.write("\n\n".join(paragraphs))

        print(f"Wrote detailed summaries → {out_txt}")

if __name__ == '__main__':
    embed_detailed_summaries(metrics_root='csvs')

