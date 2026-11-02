"""Small EDA summaries derived from the read-only validation report."""

from __future__ import annotations

from typing import Any

import pandas as pd


def class_distribution(
    validation_df: pd.DataFrame,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Summarize valid object counts by source and class."""
    valid = validation_df.loc[validation_df["status"].eq("valid")]
    rows: list[dict[str, Any]] = []
    for source in sorted(validation_df["source"].dropna().unique()):
        source_rows = valid.loc[valid["source"].eq(source)]
        for class_id, class_name in class_names.items():
            rows.append(
                {
                    "source": source,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "object_count": int(source_rows[f"num_{class_name}"].sum()),
                }
            )
    return pd.DataFrame(rows)


def image_shape_summary(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Return compact image-dimension statistics for each source."""
    valid = validation_df.loc[validation_df["status"].eq("valid")].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "num_images",
                "median_width",
                "median_height",
                "min_width",
                "min_height",
                "max_width",
                "max_height",
            ]
        )
    return (
        valid.groupby("source", as_index=False)
        .agg(
            num_images=("image_path", "count"),
            median_width=("image_width", "median"),
            median_height=("image_height", "median"),
            min_width=("image_width", "min"),
            min_height=("image_height", "min"),
            max_width=("image_width", "max"),
            max_height=("image_height", "max"),
        )
        .sort_values("source", kind="stable")
        .reset_index(drop=True)
    )


def validation_decision(validation_df: pd.DataFrame) -> dict[str, Any]:
    """Return counts and the proceed/block decision for a source audit."""
    invalid_rows = int(validation_df["status"].ne("valid").sum())
    warning_rows = int(validation_df["warnings"].fillna("").ne("").sum())
    return {
        "valid_rows": int(validation_df["status"].eq("valid").sum()),
        "invalid_rows": invalid_rows,
        "warning_rows": warning_rows,
        "can_proceed": invalid_rows == 0 and not validation_df.empty,
    }
