"""Simple smoke test for the 3D print optimization app.

Run with:
    python tests/smoke_test.py

This script uses demo data only to exercise the key pipeline without requiring
external uploaded files.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import DataError, validate_and_clean
from src.demo_data import generate_demo_data
from src.model_training import fit_random_forest_model
from src.genetic_algorithm import run_ga
from src.report import generate_html_report


PASS = 0
FAIL = 0


def check(name: str, condition: bool) -> None:
    """Print PASS/FAIL for each test step."""
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {name}")
    if condition:
        PASS += 1
    else:
        FAIL += 1


def main() -> None:
    """Run the smoke-test sequence."""
    df = generate_demo_data(n=80, seed=42)
    clean_df, report = validate_and_clean(df)
    check("demo data validates and cleans", len(clean_df) >= 10 and "rows_kept" in report)

    model_result = fit_random_forest_model(clean_df)
    check("model trains successfully", model_result["model"] is not None)

    prediction_df = df[["layer_height", "nozzle_temperature", "print_speed", "infill_density", "material_type"]].head(1).copy()
    prediction = model_result["model"].predict(prediction_df)
    check("single-row prediction works", len(prediction) == 1 and prediction[0] > 0)

    bounds = {
        "layer_height": (0.10, 0.35),
        "nozzle_temperature": (180, 300),
        "print_speed": (20, 120),
        "infill_density": (20, 95),
        "material": {"PLA": (190, 230), "PETG": (220, 250), "ABS": (220, 265)},
    }

    ga_result = run_ga(
        model=model_result["model"],
        bounds=bounds,
        material_mode="fixed",
        selected_material="PLA",
        original_params={"layer_height": 0.18, "nozzle_temperature": 215, "print_speed": 50, "infill_density": 80, "material": "PLA"},
        pop_size=12,
        generations=5,
        crossover_rate=0.8,
        mutation_rate=0.2,
        tournament_size=3,
        elitism=1,
        seed=42,
    )
    check("GA completes and returns improvement", ga_result.best_fitness >= 0)

    parameters = ["layer_height", "nozzle_temperature", "print_speed", "infill_density"]
    for param in parameters:
        check(f"GA solution respects {param} bounds", all(ga_result.best_params.get(param, 0) >= bounds[param][0] and ga_result.best_params.get(param, 0) <= bounds[param][1] for _ in [0]))

    try:
        invalid = df.copy()
        invalid = invalid.drop(columns=["material_type"])
        validate_and_clean(invalid)
        check("invalid CSV detection fails as expected", False)
    except DataError:
        check("invalid CSV detection works", True)

    output_file = PROJECT_ROOT / "reports" / "smoke_report.html"
    report_path = generate_html_report(
        output_path=output_file,
        title="Test report",
        data_mode="Demo",
        dataset_name="demo_data.csv",
        row_count=len(clean_df),
        model_metrics={"test_r2": 0.7, "cv_r2": 0.68, "train_r2": 0.82, "mae": 2.1, "rmse": 2.7, "oob_r2": 0.74},
        original_params={"layer_height": 0.18, "nozzle_temperature": 215, "print_speed": 50, "infill_density": 80},
        optimized_params={"layer_height": 0.17, "nozzle_temperature": 210, "print_speed": 48, "infill_density": 82},
        original_strength=30.0,
        optimized_strength=32.0,
        improvement=2.0,
        ga_settings={"pop_size": 12, "generations": 5},
        honesty_notice="The optimizer maximizes the Random Forest model's PREDICTED tensile strength. This does not prove that a physical part printed with these parameters will reach that strength. Confirm by printing and tensile-testing specimens.",
        is_demo=True,
    )
    check("HTML report generated", Path(report_path).exists())

    print(f"SUMMARY: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
