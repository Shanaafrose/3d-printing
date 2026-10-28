"""Illustrative synthetic data for demo mode only.

This file is intentionally labelled as illustrative. The values are synthesized to
show the workflow clearly without pretending to represent validated experimental
results. The app must only use this dataset when the user explicitly toggles the
sidebar option for demo data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic FDM 3D printing data for PLA, PETG and ABS.

    The formula is deliberately simple and physics-flavoured, with noise added so
    the app can demonstrate the Random Forest model and optimizer without claiming
    actual measured tensile strength values.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | str]] = []

    material_profiles = {
        "PLA": {"base": 38.0, "layer_bias": 0.9, "speed_bias": 0.5, "temp_bias": 0.08},
        "PETG": {"base": 42.0, "layer_bias": 1.0, "speed_bias": 0.45, "temp_bias": 0.09},
        "ABS": {"base": 35.0, "layer_bias": 0.8, "speed_bias": 0.4, "temp_bias": 0.11},
    }

    for _ in range(n):
        material = rng.choice(["PLA", "PETG", "ABS"], p=[0.5, 0.3, 0.2])
        profile = material_profiles[material]

        layer_height = float(rng.uniform(0.10, 0.33))
        nozzle_temperature = float(rng.uniform(200, 255)) if material == "PLA" else float(rng.uniform(220, 265)) if material == "PETG" else float(rng.uniform(230, 270))
        print_speed = float(rng.uniform(25, 120))
        infill_density = float(rng.uniform(20, 95))

        # Physics-flavoured trend: best strengths occur with moderate layer height,
        # correct nozzle temperature, and moderate-to-high infill. Real experimental
        # data would be needed to calibrate this properly.
        strength = (
            profile["base"]
            + 8.5 * (1.0 - abs(layer_height - 0.18) / 0.2)
            + 0.07 * (nozzle_temperature - 220)
            + 0.10 * (infill_density - 50)
            - 0.04 * (print_speed - 60)
            + 2.0 * profile["layer_bias"]
            + rng.normal(0, 2.8)
        )

        # The material-specific nozzle values and density interact to create a
        # realistic but synthetic performance distribution.
        strength += profile["temp_bias"] * (nozzle_temperature - 220)
        strength -= 0.03 * abs(print_speed - 60)
        strength -= 0.02 * abs(layer_height - 0.18) * 100

        records.append(
            {
                "layer_height": round(layer_height, 3),
                "nozzle_temperature": round(nozzle_temperature, 1),
                "print_speed": round(print_speed, 1),
                "infill_density": round(infill_density, 1),
                "material_type": material,
                "tensile_strength": round(max(strength, 8.0), 3),
            }
        )

    return pd.DataFrame(records, columns=[
        "layer_height",
        "nozzle_temperature",
        "print_speed",
        "infill_density",
        "material_type",
        "tensile_strength",
    ])
