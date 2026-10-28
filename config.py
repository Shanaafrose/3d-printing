"""Project configuration for the 3D print optimization app.

This file is the single source of truth for the project metadata and parameter
limits. Beginners can edit the values here without hunting through the app.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and folder paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# ---------------------------------------------------------------------------
# Academic project metadata (use this terminology exactly as requested)
# ---------------------------------------------------------------------------
TITLE = "AI-Based Optimization of 3D Printing Parameters for Maximum Tensile Strength Using Random Forest and Genetic Algorithm"
COURSE = "U23AM499 - Artificial Intelligence and Machine Learning in Mechanical Systems (Sem V)"
INSTITUTION = "Sri Eshwar College of Engineering | Batch No: 23"
STUDENT = "SHANA AFROSE CM, Reg. no 722824114023"
SUPERVISOR_NAME = "Enter supervisor name"

# ---------------------------------------------------------------------------
# Printer setup and standard column names
# ---------------------------------------------------------------------------
PRINTER_NAME = "Enter your printer model"
NOZZLE_DIAMETER_MM = 0.4

LAYER_HEIGHT = "layer_height"
NOZZLE_TEMPERATURE = "nozzle_temperature"
PRINT_SPEED = "print_speed"
INFILL_DENSITY = "infill_density"
MATERIAL_TYPE = "material_type"
TENSILE_STRENGTH = "tensile_strength"

NUMERIC_COLUMNS = [
    LAYER_HEIGHT,
    NOZZLE_TEMPERATURE,
    PRINT_SPEED,
    INFILL_DENSITY,
    TENSILE_STRENGTH,
]

CATEGORICAL_COLUMNS = [MATERIAL_TYPE]
REQUIRED_COLUMNS = [
    LAYER_HEIGHT,
    NOZZLE_TEMPERATURE,
    PRINT_SPEED,
    INFILL_DENSITY,
    MATERIAL_TYPE,
    TENSILE_STRENGTH,
]

# We track the order in a fixed way so the model and GA always use the same
# column names even if the CSV has different spellings.
MODEL_INPUT_COLUMNS = [
    LAYER_HEIGHT,
    NOZZLE_TEMPERATURE,
    PRINT_SPEED,
    INFILL_DENSITY,
    MATERIAL_TYPE,
]

# Units: mm, degC, mm/s, %, categorical, MPa.
PARAM_STEP = {
    LAYER_HEIGHT: 0.05,
    NOZZLE_TEMPERATURE: 5,
    PRINT_SPEED: 5,
    INFILL_DENSITY: 5,
}

# Absolute hard limits. These are a safety guard for the app and the optimizer.
PRINTER_LIMITS = {
    LAYER_HEIGHT: (0.05, 0.40),
    NOZZLE_TEMPERATURE: (180, 300),
    PRINT_SPEED: (10, 150),
    INFILL_DENSITY: (5, 100),
}

# Typical material-specific nozzle temperature ranges. The student should check
# their filament datasheet before trusting these values.
MATERIAL_LIMITS = {
    "PLA": (190, 230),
    "PETG": (220, 250),
    "ABS": (220, 265),
}

# Rule: the chosen layer height must be no more than 80% of the nozzle diameter.
# This is a common FDM constraint to avoid poor bead adhesion or poor print quality.
LAYER_HEIGHT_MAX_DUE_TO_NOZZLE = 0.8 * NOZZLE_DIAMETER_MM

# Minimum dataset size rules for training reliability warnings and hard failures.
MIN_ROWS_HARD = 10
MIN_ROWS_RECOMMENDED = 30

RANDOM_SEED = 42

# Common alternative CSV headers that may appear in student datasets.
# The mapping is case/space insensitive during import.
COLUMN_ALIASES = {
    "tension_strength": TENSILE_STRENGTH,
    "ultimate_tensile_strength": TENSILE_STRENGTH,
    "uts": TENSILE_STRENGTH,
    "nozzle_temp": NOZZLE_TEMPERATURE,
    "printspeed": PRINT_SPEED,
    "print_speed_mm_s": PRINT_SPEED,
    "infill": INFILL_DENSITY,
    "infill_pct": INFILL_DENSITY,
    "material": MATERIAL_TYPE,
    "material_type_name": MATERIAL_TYPE,
    "layer": LAYER_HEIGHT,
    "layer_height_mm": LAYER_HEIGHT,
    "temp": NOZZLE_TEMPERATURE,
    "nozzle_temperature_c": NOZZLE_TEMPERATURE,
}

# Helper for beginner-friendly validation of user inputs.
def validate_layer_height(value: float) -> float:
    """Return the value if it satisfies the nozzle-limit rule."""
    if value > LAYER_HEIGHT_MAX_DUE_TO_NOZZLE:
        raise ValueError(
            "Layer height exceeds the safe rule for a 0.4 mm nozzle "
            f"(must be <= {LAYER_HEIGHT_MAX_DUE_TO_NOZZLE:.2f} mm)."
        )
    return value


def ensure_directories() -> None:
    """Create data and models folders when the app starts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# End of configuration.
