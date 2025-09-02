import json
import os
import numpy as np
import pandas as pd
from skimage.draw     import polygon
from skimage.measure  import regionprops, label
from skimage.io       import imread
from skimage.color    import rgb2gray
from skimage.util     import img_as_ubyte
from skimage.exposure import rescale_intensity
from scipy import stats

def compute_metrics_per_image(json_filepath, output_dir='metrics_by_image', levels=8):
    # Load the COCO‐style JSON
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # Build lookup for images
    image_info = {img['id']: img for img in data['images']}

    # Prepare per‐image storage
    metrics_by_image = {img_id: [] for img_id in image_info}

    # Process each annotation
    for ann in data['annotations']:
        img_id   = ann['image_id']
        img_meta = image_info[img_id]
        h, w     = img_meta['height'], img_meta['width']

        # Rasterize segmentation into mask
        mask = np.zeros((h, w), dtype=bool)
        for seg in ann['segmentation']:
            xs, ys       = seg[0::2], seg[1::2]
            rr, cc       = polygon(ys, xs, shape=mask.shape)
            mask[rr, cc] = True

        # Compute region properties
        lbl   = label(mask)
        props = regionprops(lbl)
        if not props:
            continue
        p = props[0]

        # Compute bbox‐area
        minr, minc, maxr, maxc = p.bbox
        area_bbox = (maxr - minr) * (maxc - minc)

        # Compute equivalent‐diameter‐area
        eq_diam = p.equivalent_diameter
        eq_diam_area = np.pi * (eq_diam / 2) ** 2

        metrics_by_image[img_id].append({
            "area":                     p.area,
            "area_convex":              p.convex_area,
            "equivalent_diameter_area": eq_diam_area,
            "major_axis_length":        p.major_axis_length,
            "minor_axis_length":        p.minor_axis_length,
            "eccentricity":             p.eccentricity,
            "orientation":              p.orientation,
            "equivalent_diameter":      eq_diam,
            "feret_diameter_max":       p.feret_diameter_max,
            "solidity":                 p.solidity,
            "extent":                   p.area / ((maxr - minr) * (maxc - minc)),
            "circularity":              4 * np.pi * p.area / (p.perimeter ** 2)
                                         if p.perimeter > 0 else np.nan
        })

    # Create top‐level output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write one CSV per image, in its own subfolder
    for img_id, rows in metrics_by_image.items():
        if not rows:
            continue

        img_meta   = image_info[img_id]
        fname      = img_meta['file_name']                   # e.g. "foo_png123.png"
        base_split = fname.split('_png', 1)[0]               # e.g. "foo"
        subdir     = os.path.join(output_dir, base_split)
        os.makedirs(subdir, exist_ok=True)

        # Build DataFrame and add object_id starting from 0
        df = pd.DataFrame(rows)
        df.insert(0, 'object_id', range(len(df)))

        # Write raw CSV
        raw_csv_name = f"{base_split}_raw.csv"
        raw_out_path = os.path.join(subdir, raw_csv_name)
        df.to_csv(raw_out_path, index=False)

        # Compute processed statistics
        stats_names = [
            "Mean", "Median", "Mode", "Range", "Min", "Max",
            "Std Dev", "Variance", "IQR", "Q1", "Q3",
            "Coeff of Variation", "Entropy"
        ]
        processed = []
        for metric in df.columns.drop('object_id'):
            series = df[metric].dropna()
            if series.empty:
                values = [np.nan] * len(stats_names)
            else:
                mean   = series.mean()
                median = series.median()
                mode_v = series.mode()
                mode   = mode_v.iloc[0] if not mode_v.empty else np.nan
                mn     = series.min()
                mx     = series.max()
                range_ = mx - mn
                std    = series.std(ddof=0)
                var    = series.var(ddof=0)
                q1     = series.quantile(0.25)
                q3     = series.quantile(0.75)
                iqr    = q3 - q1
                cv     = std / mean if mean != 0 else np.nan
                counts = series.value_counts(normalize=True)
                entropy = stats.entropy(counts, base=2)
                values = [
                    mean, median, mode, range_, mn, mx,
                    std, var, iqr, q1, q3,
                    cv, entropy
                ]
            processed.append([metric] + values)

        proc_df     = pd.DataFrame(processed, columns=['Metric'] + stats_names)
        proc_csv_name = f"{base_split}_processed.csv"
        proc_out_path = os.path.join(subdir, proc_csv_name)
        proc_df.to_csv(proc_out_path, index=False)

        print(f"Wrote raw and processed metrics for image {img_id} → {subdir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute region metrics per-image from COCO JSON"
    )
    parser.add_argument('json_filepath', help="Path to COCO JSON file")
    parser.add_argument(
        '--output_dir',
        default='metrics_by_image',
        help="Directory to write per-image CSV subfolders"
    )
    args = parser.parse_args()
    compute_metrics_per_image(args.json_filepath, args.output_dir)


    '''
    python extractCharacteristics.py test_annotations.coco.json --output_dir csvs
    '''