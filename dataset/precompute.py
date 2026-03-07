"""Pre-compute the dataset pipeline and save results as parquet files.

Run at Docker build time to avoid loading the full HuggingFace dataset at runtime.
Outputs are saved to DATA_DIR (default: /app/data/).
"""

import json
import os

import datasets
import Levenshtein
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    ds = datasets.load_dataset("zhk/wiki-edits", split="train")
    df_raw = ds.to_pandas()

    intent_counts = df_raw["intent"].value_counts()
    intents_ordered = [i for i in df_raw["intent"].unique().tolist() if i != "Fluency"] + ["Fluency"]

    df_raw.head(10).to_parquet(f"{DATA_DIR}/raw_head.parquet")

    intent_frames = []
    for intent in intents_ordered:
        intent_frames.append(df_raw[df_raw["intent"] == intent].head(5))
    pd.concat(intent_frames).to_parquet(f"{DATA_DIR}/intent_samples.parquet")

    df_fluency = df_raw[df_raw["intent"] == "Fluency"].drop(columns=["intent"])
    fluency_count = len(df_fluency)

    df_fluency.head(10).to_parquet(f"{DATA_DIR}/fluency_head.parquet")

    df_with_dist = df_fluency.copy()
    df_with_dist["edit_distance"] = df_with_dist.apply(
        lambda row: Levenshtein.distance(row["source"], row["target"]), axis=1
    )

    df_with_dist[df_with_dist["edit_distance"] > 8].head(3).to_parquet(f"{DATA_DIR}/high_ed_samples.parquet")
    df_with_dist[df_with_dist["edit_distance"] == 1].head(3).to_parquet(f"{DATA_DIR}/low_ed_samples.parquet")

    max_ed = int(df_with_dist["edit_distance"].max())
    ed_distribution = df_with_dist["edit_distance"].value_counts().sort_index().head(20)

    ed_frames = []
    ed_sample_counts = {}
    for ed in range(1, 21):
        subset = df_with_dist[df_with_dist["edit_distance"] == ed]
        ed_sample_counts[str(ed)] = len(subset)
        if len(subset) > 0:
            ed_frames.append(subset.head(5))
    if ed_frames:
        pd.concat(ed_frames).to_parquet(f"{DATA_DIR}/ed_samples.parquet")

    df_final = df_with_dist[df_with_dist["edit_distance"] == 1][["source", "target", "edit_distance"]]
    df_final.to_parquet(f"{DATA_DIR}/final.parquet")

    stats = {
        "total_rows": len(df_raw),
        "intent_counts": {k: int(v) for k, v in intent_counts.items()},
        "intents_ordered": intents_ordered,
        "fluency_count": fluency_count,
        "max_ed": max_ed,
        "ed_distribution": {str(k): int(v) for k, v in ed_distribution.items()},
        "ed_sample_counts": ed_sample_counts,
        "final_count": len(df_final),
    }
    with open(f"{DATA_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    total_bytes = sum(
        os.path.getsize(os.path.join(DATA_DIR, fname))
        for fname in os.listdir(DATA_DIR)
    )
    print(f"Pre-computed {len(os.listdir(DATA_DIR))} files ({total_bytes / 1024 / 1024:.1f} MB) to {DATA_DIR}")


if __name__ == "__main__":
    main()
