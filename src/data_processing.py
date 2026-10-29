"""Utilities for reading, cleaning, validating, and summarizing the dataset.

The app uses this module for:
- CSV loading
- automatic column mapping from common header variants
- validation and cleaning of experimental observations
- bounds calculation for optimization and extrapolation checks
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import (
    COLUMN_ALIASES,
    LAYER_HEIGHT,
    MATERIAL_LIMITS,
    MIN_ROWS_HARD,
    MIN_ROWS_RECOMMENDED,
    MATERIAL_TYPE,
    MODEL_INPUT_COLUMNS,
    NUMERIC_COLUMNS,
    PRINTER_LIMITS,
    REQUIRED_COLUMNS,
    TENSILE_STRENGTH,
)


class DataError(ValueError):
    """Friendly exception raised for user-uploaded data issues."""


def _normalize_name(value: str) -> str:
    """Lowercase and strip spaces, underscores, hyphens from a column name."""
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized


def load_csv(path_or_buffer: str | Any) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Raises DataError with a beginner-friendly explanation when the file does not
    exist, is empty, or cannot be read.
    """
    try:
        df = pd.read_csv(path_or_buffer)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {path_or_buffer}. Place the CSV in the data folder or upload it again.") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataError("The CSV file is empty. Please upload a file with at least one data row.") from exc
    except Exception as exc:  # pragma: no cover - user-facing guard
        raise DataError(f"The CSV could not be read: {exc}") from exc

    if df.empty:
        raise DataError("The CSV file has no rows. Add experimental observations before training.")

    return df


def apply_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Rename standard and alternative CSV headers to the project schema.

    The function is case and space insensitive and keeps extra columns untouched.
    """
    renamed = df.copy()
    alias_lookup = {
        _normalize_name(key): value for key, value in COLUMN_ALIASES.items()
    }

    rename_map: dict[str, str] = {}
    for col in renamed.columns:
        norm = _normalize_name(col)
        if norm in alias_lookup and alias_lookup[norm] not in renamed.columns:
            rename_map[col] = alias_lookup[norm]
    if rename_map:
        renamed = renamed.rename(columns=rename_map)

    return renamed


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize values, remove invalid rows, and return a cleaned DataFrame and a report.

    This function intentionally does not remove duplicate rows because multiple
    experimental replicates are legitimate and often useful for modelling.
    """
    cleaned = apply_column_mapping(df.copy())

    missing = [col for col in REQUIRED_COLUMNS if col not in cleaned.columns]
    if missing:
        missing_text = ", ".join(f"'{c}'" for c in missing)
        available = ", ".join(cleaned.columns.tolist()) if len(cleaned.columns) else "none"
        raise DataError(
            f"Column(s) missing: {missing_text}. Found: [{available}]. "
            "Use the column mapping tool or rename your CSV columns to match the required schema."
        )

    cleaned = cleaned.loc[:, REQUIRED_COLUMNS].copy()
    cleaned.columns = REQUIRED_COLUMNS

    cleaned[MATERIAL_TYPE] = cleaned[MATERIAL_TYPE].astype(str).str.strip().str.upper()

    numeric_cols = [
        "layer_height",
        "nozzle_temperature",
        "print_speed",
        "infill_density",
        "tensile_strength",
    ]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    report: dict[str, Any] = {
        "rows_original": int(len(cleaned)),
        "rows_kept": 0,
        "rows_dropped": 0,
        "reasons": {},
        "warnings": [],
        "ignored_columns": [],
    }

    all_reasons = {
        "non_positive_tensile": 0,
        "missing_values": 0,
        "negative_layer_height": 0,
        "negative_temp": 0,
        "negative_speed": 0,
        "bad_infill": 0,
    }

    # Remove rows with missing values in critical fields.
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=[
        "layer_height",
        "nozzle_temperature",
        "print_speed",
        "infill_density",
        "material_type",
        "tensile_strength",
    ]).copy()
    all_reasons["missing_values"] = before - len(cleaned)

    # Keep only positive tensile strength and sensible process values.
    valid_mask = pd.Series(True, index=cleaned.index)
    valid_mask &= cleaned[TENSILE_STRENGTH] > 0
    all_reasons["non_positive_tensile"] = int((~(cleaned[TENSILE_STRENGTH] > 0)).sum())

    valid_mask &= cleaned[LAYER_HEIGHT] > 0
    all_reasons["negative_layer_height"] = int((cleaned[LAYER_HEIGHT] <= 0).sum())

    valid_mask &= cleaned["nozzle_temperature"] > 0
    all_reasons["negative_temp"] = int((cleaned["nozzle_temperature"] <= 0).sum())

    valid_mask &= cleaned["print_speed"] > 0
    all_reasons["negative_speed"] = int((cleaned["print_speed"] <= 0).sum())

    valid_mask &= (cleaned["infill_density"] >= 0) & (cleaned["infill_density"] <= 100)
    all_reasons["bad_infill"] = int(((cleaned["infill_density"] < 0) | (cleaned["infill_density"] > 100)).sum())

    cleaned = cleaned[valid_mask].copy()

    rows_removed = report["rows_original"] - len(cleaned)
    report["rows_dropped"] = int(rows_removed)
    report["rows_kept"] = int(len(cleaned))
    report["reasons"] = {k: int(v) for k, v in all_reasons.items() if v > 0}

    if len(cleaned) < MIN_ROWS_HARD:
        raise DataError(
            f"Too few valid rows remain for training: {len(cleaned)}. "
            f"Minimum required: {MIN_ROWS_HARD}. Please add more data or clean the dataset."
        )

    if len(cleaned) < MIN_ROWS_RECOMMENDED:
        report["warnings"].append(
            f"The dataset has {len(cleaned)} rows, which is below the recommended minimum of {MIN_ROWS_RECOMMENDED}. "
            "Model metrics may be optimistic or unstable."
        )

    return cleaned.reset_index(drop=True), report


def compute_bounds(clean_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall and material-specific parameter bounds.

    Returns the training range used by the optimizer. Where material-specific
    intersection with the printer limits is empty, the dataset range is used and a
    warning is added.
    """
    bounds: dict[str, Any] = {
        "overall": {},
        "by_material": {},
        "fixed": [],
        "warnings": [],
    }

    numeric_params = [
        "layer_height",
        "nozzle_temperature",
        "print_speed",
        "infill_density",
    ]

    for param in numeric_params:
        min_v = float(clean_df[param].min())
        max_v = float(clean_df[param].max())
        bounds["overall"][param] = (min_v, max_v)
        if np.isclose(min_v, max_v):
            bounds["fixed"].append(param)

    for material, group in clean_df.groupby("material_type"):
        material_bounds: dict[str, tuple[float, float]] = {}
        for param in numeric_params:
            data_min = float(group[param].min())
            data_max = float(group[param].max())

            if param == "nozzle_temperature":
                material_min, material_max = MATERIAL_LIMITS.get(material.upper(), PRINTER_LIMITS[param])
            else:
                material_min, material_max = PRINTER_LIMITS[param]

            lower_bound = max(data_min, material_min)
            upper_bound = min(data_max, material_max)

            if lower_bound > upper_bound:
                lower_bound, upper_bound = data_min, data_max
                bounds["warnings"].append(
                    f"Material {material} had no valid overlap between its dataset range and the allowed limits for {param}. "
                    "The dataset range was used instead."
                )

            material_bounds[param] = (float(lower_bound), float(upper_bound))
        bounds["by_material"][material] = material_bounds

    return bounds


def is_extrapolating(inputs: dict[str, float], bounds: dict[str, Any]) -> bool:
    """Return True if any input value lies outside the learned material bounds."""
    material = str(inputs.get("material_type", "")).upper()
    material_bounds = bounds["by_material"].get(material, bounds["overall"])
    for param in ["layer_height", "nozzle_temperature", "print_speed", "infill_density"]:
        value = float(inputs.get(param, 0.0))
        lower, upper = material_bounds[param]
        if value < lower or value > upper:
            return True
    return False
