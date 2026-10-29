"""Main Streamlit application entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    COURSE,
    DATA_DIR,
    INSTITUTION,
    MATERIAL_TYPE,
    MIN_ROWS_HARD,
    MODEL_INPUT_COLUMNS,
    PRINTER_NAME,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
    STUDENT,
    TITLE,
    TENSILE_STRENGTH,
    validate_layer_height,
)
from src.data_processing import DataError, apply_column_mapping, load_csv, validate_and_clean
# Import data processing functions
from src.demo_data import generate_demo_data
from src.model_training import fit_random_forest_model, get_importance_chart_data, model_overfitting_warning, predict_with_model
from src.report import generate_html_report
from src.ui import inject_theme_css, metric_card, render_data_badge, render_demo_banner, render_footer
from src.visualizations import (
    plot_actual_vs_predicted,
    plot_ga_convergence,
    plot_histogram,
    plot_metric_importance,
    plot_optimization_comparison,
    plot_response_surface,
    plot_residuals,
    plot_tensile_vs_parameter,
)

st.set_page_config(layout="wide")
inject_theme_css()


def initialise_session_state() -> None:
    """Create session-state keys used by the app."""
    defaults = {
        "df": None,
        "clean_df": None,
        "data_source": "",
        "data_mode": "real",
        "demo_toggle": False,
        "model_result": None,
        "latest_prediction": None,
        "latest_optimization": None,
        "measurement_input": None,
        "selected_material": None,
        "input_params": {
            "layer_height": 0.18,
            "nozzle_temperature": 215,
            "print_speed": 50,
            "infill_density": 80,
            "material_type": "PLA",
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_project_data() -> pd.DataFrame:
    """Return the active dataset for the current session."""
    if st.session_state.get("demo_toggle"):
        return generate_demo_data(n=120, seed=42)

    data_path = DATA_DIR / "experimental_data.csv"
    if not data_path.exists():
        return pd.DataFrame()

    try:
        return load_csv(data_path)
    except DataError:
        return pd.DataFrame()


def get_display_filename() -> str:
    """Return a human-readable filename label for the active dataset."""
    if st.session_state.get("demo_toggle"):
        return "illustrative_demo_data.csv"
    data_path = DATA_DIR / "experimental_data.csv"
    return data_path.name if data_path.exists() else "no_file"


def show_data_mode_banner() -> None:
    """Display the correct label banner for demo or real mode."""
    if st.session_state.get("demo_toggle"):
        render_demo_banner()
    else:
        if st.session_state.get("df") is not None and len(st.session_state["df"]) > 0:
            render_data_badge(get_display_filename(), len(st.session_state["df"]))


def load_data_if_available() -> pd.DataFrame | None:
    """Load the dataset and store it in session state."""
    if st.session_state.get("demo_toggle"):
        df = generate_demo_data(n=120, seed=42)
        st.session_state["df"] = df
        st.session_state["data_source"] = "Demo data"
        return df

    data_path = DATA_DIR / "experimental_data.csv"
    if not data_path.exists():
        st.session_state["df"] = pd.DataFrame()
        st.session_state["data_source"] = "No file"
        return None

    try:
        df = load_csv(data_path)
        st.session_state["df"] = df
        st.session_state["data_source"] = data_path.name
        return df
    except DataError as exc:
        st.error(str(exc))
        st.session_state["df"] = pd.DataFrame()
        st.session_state["data_source"] = "Error"
        return None


def require_data() -> pd.DataFrame:
    """Stop the app if no data is loaded and the user is not in demo mode."""
    if st.session_state.get("demo_toggle"):
        return st.session_state["df"]

    if st.session_state.get("df") is None or st.session_state["df"].empty:
        st.info("Upload a CSV file or put experimental_data.csv in the /data folder before using this page.")
        st.stop()
    return st.session_state["df"]


def render_overview_page() -> None:
    """Render the Overview page with project introduction and workflow cards."""
    st.title(TITLE)
    show_data_mode_banner()
    st.caption(f"{COURSE} | {INSTITUTION}")
    st.write(
        "This project explores an AI-based workflow for FDM 3D printing, where Random Forest predicts tensile strength and a Genetic Algorithm searches for better process parameters. The aim is to reduce trial-and-error experiments and support a more informed process window for student-led optimization work."
    )
    st.write(
        "The application keeps the workflow transparent by showing the dataset, training metrics, prediction checks, and optimization logic in one place. It also highlights the two key challenges in the project brief: preventing model overfitting and validating predicted results before relying on them."
    )

    st.subheader("Workflow")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Data")
        st.write("Load the experimental CSV or switch to demo data.")
    with col2:
        st.markdown("### 2. Random Forest")
        st.write("Train the model and inspect metrics and warnings.")
    with col3:
        st.markdown("### 3. Genetic Algorithm")
        st.write("Search for a stronger process setting using the model.")

    st.subheader("Project details")
    st.write(f"Student: {STUDENT}")
    st.write(f"Supervisor: {st.session_state.get('supervisor_name', 'Enter supervisor name')}")
    st.write(f"Scope: FDM 3D printing only. Random Forest predicts tensile strength; Genetic Algorithm optimizes process parameters.")


def render_dataset_page() -> None:
    """Render dataset upload, data preview, and summary diagnostics."""
    st.title("Dataset")
    show_data_mode_banner()

    with st.sidebar:
        st.subheader("Data source")
        demo_toggle = st.toggle("Use illustrative demo data", value=st.session_state.get("demo_toggle", False), help="Enable only for demonstration. This uses synthetic data, not experimental measurements.")
        st.session_state["demo_toggle"] = demo_toggle

    if st.session_state.get("demo_toggle"):
        st.info("Demo mode uses synthetic data generated in src/demo_data.py. It is for demonstration only and not validated against real experiments.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload your experimental measurements. The file should use the columns listed in the project template.")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.session_state["data_source"] = uploaded_file.name
        except Exception as exc:
            st.error(f"The uploaded CSV could not be read: {exc}")
            return

    if st.session_state["df"] is None or st.session_state["df"].empty:
        if not st.session_state.get("demo_toggle"):
            st.info("No dataset loaded yet. Upload a CSV or use demo data.")
            sample_path = PROJECT_ROOT / "data" / "sample_template.csv"
            if sample_path.exists():
                with open(sample_path, "rb") as file:
                    st.download_button("Download sample template", file, file_name="sample_template.csv", mime="text/csv")
            return

    df = st.session_state["df"]
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    try:
        clean_df, report = validate_and_clean(df)
        st.session_state["clean_df"] = clean_df
    except DataError as exc:
        st.error(str(exc))
        return

    st.subheader("Cleaning summary")
    st.json(report)

    with st.expander("Missing values and ignored columns"):
        st.write("Ignored columns are extra fields not required by the project schema.")
        ignored = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        st.write(f"Ignored columns: {ignored if ignored else 'none'}")
        st.write(df.isna().sum())

    st.subheader("Dataset statistics")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    material_counts = df.groupby("material_type").size().reset_index(name="count") if "material_type" in df.columns else pd.DataFrame()
    st.subheader("Rows per material")
    st.dataframe(material_counts, use_container_width=True)

    st.subheader("Tensile strength distribution")
    st.plotly_chart(plot_histogram(df, "Tensile strength distribution"), use_container_width=True)

    parameter = st.selectbox("Scatter parameter", ["layer_height", "nozzle_temperature", "print_speed", "infill_density"], index=0)
    st.plotly_chart(plot_tensile_vs_parameter(df, parameter, f"Tensile strength vs {parameter}"), use_container_width=True)


def render_model_page() -> None:
    """Render the Model & Metrics page."""
    st.title("Model & Metrics")
    show_data_mode_banner()
    df = require_data()

    if st.session_state.get("clean_df") is None:
        try:
            clean_df, _ = validate_and_clean(df)
            st.session_state["clean_df"] = clean_df
        except DataError as exc:
            st.error(str(exc))
            st.stop()

    data = st.session_state["clean_df"]

    with st.expander("Advanced model settings"):
        n_estimators = st.slider("n_estimators", min_value=50, max_value=500, value=300, step=50, help="Number of trees in the Random Forest.")
        max_depth = st.slider("max_depth", min_value=2, max_value=20, value=10, step=1, help="Maximum tree depth; leave at higher values for a more flexible model.")
        min_samples_leaf = st.slider("min_samples_leaf", min_value=1, max_value=10, value=1, step=1, help="Minimum samples required at a leaf node.")

    if st.button("Train / Retrain model", help="Rebuild the model from the current dataset."):
        try:
            with st.spinner("Training Random Forest..."):
                model_result = fit_random_forest_model(data, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                st.session_state["model_result"] = model_result
            st.success("Training complete.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    model_result = st.session_state.get("model_result")
    if model_result is None:
        if st.button("Train on current dataset", help="Train the model once before evaluating metrics."):
            with st.spinner("Training Random Forest..."):
                model_result = fit_random_forest_model(data, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                st.session_state["model_result"] = model_result
        else:
            st.info("Train the model to view the evaluation metrics.")
            return

    train_r2 = model_result.get("train_r2")
    test_r2 = model_result.get("test_r2")
    cv_r2 = model_result.get("cv_r2")
    mae = model_result.get("test_mae")
    rmse = model_result.get("test_rmse")
    oob_r2 = model_result.get("oob_r2")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Train R2", round(train_r2, 3) if train_r2 is not None else "N/A")
    with col2:
        metric_card("Test R2", round(test_r2, 3) if test_r2 is not None else "N/A")
    with col3:
        metric_card("CV R2", round(cv_r2, 3) if cv_r2 is not None else "N/A")
    with col4:
        metric_card("MAE", round(mae, 3) if mae is not None else "N/A", unit="MPa")

    col5, col6, col7 = st.columns(3)
    with col5:
        metric_card("RMSE", round(rmse, 3) if rmse is not None else "N/A", unit="MPa")
    with col6:
        metric_card("OOB R2", round(oob_r2, 3) if oob_r2 is not None else "N/A")
    with col7:
        metric_card("Rows", len(data), unit="")

    warning = model_overfitting_warning(train_r2, test_r2, len(data))
    if warning:
        st.warning(warning)

    st.caption("Metrics estimate generalization; the final model uses all rows after evaluation.")

    actual_vs_pred = pd.DataFrame({
        "actual": model_result["y_test"],
        "predicted": model_result["model"].predict(model_result["X_test"]),
    })
    st.subheader("Actual vs predicted (test set)")
    st.plotly_chart(plot_actual_vs_predicted(actual_vs_pred, "Actual vs predicted tensile strength"), use_container_width=True)

    st.subheader("Residuals")
    st.plotly_chart(plot_residuals(actual_vs_pred, "Residual histogram"), use_container_width=True)

    st.subheader("Feature importance")
    importance_df = get_importance_chart_data(model_result["importance"])
    st.plotly_chart(plot_metric_importance(importance_df, "Feature importance by aggregated tree impurity"), use_container_width=True)

    st.markdown("### How to read the metrics")
    st.write("- Test R2: shows how well the model predicts unseen data in the hold-out set.")
    st.write("- CV R2: a cross-validation estimate of generalization using multiple training/test splits.")
    st.write("- MAE and RMSE: average prediction error in MPa, lower is better.")
    st.write("- Feature importance: indicates which variables contribute most to the forest's predictions, but impurity-based importance can be biased and should be interpreted with care.")


def render_predict_page() -> None:
    """Render the Predict page."""
    st.title("Predict")
    show_data_mode_banner()
    df = require_data()

    if st.session_state.get("clean_df") is None:
        try:
            clean_df, _ = validate_and_clean(df)
            st.session_state["clean_df"] = clean_df
        except DataError as exc:
            st.error(str(exc))
            st.stop()

    data = st.session_state["clean_df"]
    if st.session_state.get("model_result") is None:
        st.info("Train the model first on the Model page before predicting a parameter set.")
        st.stop()

    materials = sorted(data["material_type"].unique())
    selected_material = st.selectbox("Material type", materials, index=0 if materials else 0, help="Choose the material to evaluate.")

    if selected_material not in materials:
        selected_material = materials[0]

    st.session_state["selected_material"] = selected_material

    material_df = data[data["material_type"] == selected_material].copy()
    param_limits = {
        "layer_height": (float(material_df["layer_height"].min()), float(material_df["layer_height"].max())),
        "nozzle_temperature": (float(material_df["nozzle_temperature"].min()), float(material_df["nozzle_temperature"].max())),
        "print_speed": (float(material_df["print_speed"].min()), float(material_df["print_speed"].max())),
        "infill_density": (float(material_df["infill_density"].min()), float(material_df["infill_density"].max())),
    }

    params = {}
    params["layer_height"] = st.slider("Layer height", min_value=float(param_limits["layer_height"][0]), max_value=float(param_limits["layer_height"][1]), value=float(np.median(material_df["layer_height"])), step=0.05, format="%.2f", help="Layer height in mm. Keep it within the material range and below the nozzle limit.")
    params["nozzle_temperature"] = st.slider("Nozzle temperature", min_value=float(param_limits["nozzle_temperature"][0]), max_value=float(param_limits["nozzle_temperature"][1]), value=float(np.median(material_df["nozzle_temperature"])), step=5.0, format="%.0f", help="Nozzle temperature in degC.")
    params["print_speed"] = st.slider("Print speed", min_value=float(param_limits["print_speed"][0]), max_value=float(param_limits["print_speed"][1]), value=float(np.median(material_df["print_speed"])), step=5.0, format="%.0f", help="Print speed in mm/s.")
    params["infill_density"] = st.slider("Infill density", min_value=float(param_limits["infill_density"][0]), max_value=float(param_limits["infill_density"][1]), value=float(np.median(material_df["infill_density"])), step=5.0, format="%.0f", help="Infill density in percent.")
    params["material_type"] = selected_material

    try:
        validate_layer_height(params["layer_height"])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    predict_input = pd.DataFrame([params])
    predictions = predict_with_model(st.session_state["model_result"]["model"], predict_input)
    predicted_strength = float(predictions["predicted_tensile_strength"].iloc[0])
    tree_spread = float(predictions["spread_across_trees"].iloc[0])

    st.session_state["latest_prediction"] = {**params, "predicted_strength": predicted_strength, "spread_across_trees": tree_spread}

    col1, col2 = st.columns([1, 1])
    with col1:
        metric_card("Predicted tensile strength", round(predicted_strength, 2), unit="MPa")
    with col2:
        metric_card("Spread across trees", round(tree_spread, 3), unit="MPa")

    st.subheader("Response-surface heatmap")
    x_param = st.selectbox("X parameter", ["layer_height", "nozzle_temperature", "print_speed", "infill_density"], index=0)
    y_param = st.selectbox("Y parameter", ["layer_height", "nozzle_temperature", "print_speed", "infill_density"], index=1)
    if x_param == y_param:
        st.warning("Select two different parameters for the response-surface plot.")
    else:
        grid_values = []
        x_values = np.linspace(param_limits[x_param][0], param_limits[x_param][1], 20)
        y_values = np.linspace(param_limits[y_param][0], param_limits[y_param][1], 20)
        for x in x_values:
            for y in y_values:
                entry = {"material_type": selected_material, x_param: x, y_param: y}
                for key in ["layer_height", "nozzle_temperature", "print_speed", "infill_density"]:
                    if key not in entry:
                        entry[key] = params[key]
                row = pd.DataFrame([entry])
                pred = float(predict_with_model(st.session_state["model_result"]["model"], row)["predicted_tensile_strength"].iloc[0])
                grid_values.append({x_param: x, y_param: y, "predicted_strength": pred})
        grid_df = pd.DataFrame(grid_values)
        st.plotly_chart(plot_response_surface(grid_df, x_param, y_param, "Predicted strength surface"), use_container_width=True)


def render_optimization_page() -> None:
    """Render the Genetic Algorithm optimization page."""
    st.title("Optimize")
    show_data_mode_banner()
    df = require_data()
    original_params = st.session_state.get("latest_prediction")
    if original_params is None:
        st.info("Generate a prediction before running the optimizer.")
        return

    if st.session_state.get("clean_df") is None:
        try:
            clean_df, _ = validate_and_clean(df)
            st.session_state["clean_df"] = clean_df
        except DataError as exc:
            st.error(str(exc))
            st.stop()

    data = st.session_state["clean_df"]
    materials = sorted(data["material_type"].unique())
    selected_material = st.session_state.get("selected_material", materials[0])

    with st.expander("GA settings", expanded=True):
        pop_size = st.slider("Population size", min_value=10, max_value=300, value=60, step=10, help="Number of candidate solutions per generation.")
        generations = st.slider("Generations", min_value=5, max_value=300, value=40, step=5, help="Total number of GA cycles.")
        crossover_rate = st.slider("Crossover rate", min_value=0.1, max_value=1.0, value=0.8, step=0.05, help="Chance of mixing two parents.")
        mutation_rate = st.slider("Mutation rate", min_value=0.05, max_value=0.5, value=0.2, step=0.05, help="Chance of random variation.")
        tournament_size = st.slider("Tournament size", min_value=2, max_value=10, value=3, step=1, help="Number of candidates compared for each selection event.")
        elitism = st.slider("Elitism", min_value=1, max_value=10, value=2, step=1, help="Best candidates copied unchanged to the next generation.")
        seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1, help="Seed for reproducible GA runs.")

    if st.button("Run Genetic Algorithm", help="Optimize the current prediction using the Random Forest model."):
        from src.genetic_algorithm import run_ga

        try:
            with st.spinner("Running GA..."):
                ga_result = run_ga(
                    model=st.session_state["model_result"]["model"],
                    bounds={
                        "layer_height": (data["layer_height"].min(), data["layer_height"].max()),
                        "nozzle_temperature": (data["nozzle_temperature"].min(), data["nozzle_temperature"].max()),
                        "print_speed": (data["print_speed"].min(), data["print_speed"].max()),
                        "infill_density": (data["infill_density"].min(), data["infill_density"].max()),
                        "material": {mat: (0, 0) for mat in materials},
                    },
                    material_mode="fixed",
                    selected_material=selected_material,
                    original_params={
                        "layer_height": original_params["layer_height"],
                        "nozzle_temperature": original_params["nozzle_temperature"],
                        "print_speed": original_params["print_speed"],
                        "infill_density": original_params["infill_density"],
                        "material": selected_material,
                    },
                    pop_size=pop_size,
                    generations=generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    tournament_size=tournament_size,
                    elitism=elitism,
                    seed=seed,
                )
            st.session_state["latest_optimization"] = ga_result
        except Exception as exc:
            st.error(f"GA failed: {exc}")

    optimization = st.session_state.get("latest_optimization")
    if optimization is None:
        return

    best = optimization.best_params
    original_strength = float(original_params["predicted_strength"])
    optimized_strength = float(optimization.best_fitness)
    improvement = optimized_strength - original_strength
    percent_gain = 100 * improvement / original_strength if original_strength != 0 else 0

    st.markdown("The optimizer maximizes the Random Forest model's PREDICTED tensile strength. This does not prove that a physical part printed with these parameters will reach that strength. Confirm by printing and tensile-testing specimens.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Original MPa", round(original_strength, 2), unit="MPa")
    with col2:
        metric_card("Optimized MPa", round(optimized_strength, 2), unit="MPa")
    with col3:
        metric_card("Absolute improvement", round(improvement, 2), unit="MPa")
    with col4:
        metric_card("Improvement %", round(percent_gain, 2), unit="%")

    st.subheader("Original vs optimized strength")
    st.plotly_chart(plot_optimization_comparison({"strength": original_strength}, {"strength": optimized_strength}, "Original vs optimized strength"), use_container_width=True)

    st.subheader("GA convergence")
    st.plotly_chart(plot_ga_convergence(optimization.history, "GA convergence"), use_container_width=True)

    st.subheader("Top 5 distinct solutions")
    st.dataframe(optimization.top_5_solutions, use_container_width=True)


def render_results_page() -> None:
    """Render the results and report page."""
    st.title("Results & Report")
    show_data_mode_banner()
    if st.session_state.get("latest_prediction") is None:
        st.info("Generate a prediction and an optimization before viewing the report.")
        return

    prediction = st.session_state["latest_prediction"]
    optimization = st.session_state.get("latest_optimization")
    if optimization is not None:
        st.subheader("Latest optimization summary")
        st.write(f"Original predicted strength: {prediction['predicted_strength']} MPa")
        st.write(f"Optimized predicted strength: {optimization.best_fitness} MPa")

    measured = st.number_input("Measured tensile strength of the optimized specimen (MPa)", min_value=0.0, value=0.0, step=0.1, help="Optional measured value used for simple validation against the model prediction.")
    if measured > 0:
        predicted = float(prediction["predicted_strength"])
        error = abs(measured - predicted)
        pct_error = (error / predicted) * 100 if predicted else 0
        st.write(f"Absolute error: {error:.2f} MPa")
        st.write(f"Percentage error: {pct_error:.2f}%")

    st.subheader("Downloads")
    export_df = pd.DataFrame([{
        "layer_height": prediction["layer_height"],
        "nozzle_temperature": prediction["nozzle_temperature"],
        "print_speed": prediction["print_speed"],
        "infill_density": prediction["infill_density"],
        "material_type": prediction["material_type"],
        "predicted_tensile_strength": prediction["predicted_strength"],
    }])
        csv_path = PROJECT_ROOT / "reports" / "latest_prediction.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
            "Download results CSV",
            data=csv_data,
            file_name="results.csv",
            mime="text/csv",
        )
    if optimization is not None:
        html_path = PROJECT_ROOT / "reports" / "project_report.html"
        generate_html_report(
            output_path=html_path,
            title=TITLE,
            data_mode="Demo" if st.session_state.get("demo_toggle") else "Real",
            dataset_name=get_display_filename(),
            row_count=len(st.session_state.get("clean_df", pd.DataFrame())),
            model_metrics={
                "train_r2": st.session_state["model_result"]["train_r2"] if st.session_state.get("model_result") else None,
                "test_r2": st.session_state["model_result"]["test_r2"] if st.session_state.get("model_result") else None,
                "cv_r2": st.session_state["model_result"]["cv_r2"] if st.session_state.get("model_result") else None,
                "mae": st.session_state["model_result"]["test_mae"] if st.session_state.get("model_result") else None,
                "rmse": st.session_state["model_result"]["test_rmse"] if st.session_state.get("model_result") else None,
                "oob_r2": st.session_state["model_result"]["oob_r2"] if st.session_state.get("model_result") else None,
            },
            original_params={
                "layer_height": prediction["layer_height"],
                "nozzle_temperature": prediction["nozzle_temperature"],
                "print_speed": prediction["print_speed"],
                "infill_density": prediction["infill_density"],
            },
            optimized_params=optimization.best_params,
            original_strength=float(prediction["predicted_strength"]),
            optimized_strength=float(optimization.best_fitness),
            improvement=float(optimization.best_fitness - prediction["predicted_strength"]),
            ga_settings=optimization.settings,
            honesty_notice="The optimizer maximizes the Random Forest model's PREDICTED tensile strength. This does not prove that a physical part printed with these parameters will reach that strength. Confirm by printing and tensile-testing specimens.",
            is_demo=bool(st.session_state.get("demo_toggle")),
        )
        with open(html_path, "rb") as file:
            st.download_button("Download HTML report", file, file_name="project_report.html", mime="text/html")


def render_about_page() -> None:
    """Render methodology and limitations page."""
    st.title("About / Methodology")
    show_data_mode_banner()
    st.subheader("Random Forest")
    st.write("A Random Forest regressor is an ensemble of decision trees. Each tree learns a slightly different view of the data, and the average prediction is used for the final result. This often works well for engineering datasets where there are nonlinear interactions between process parameters and response strength.")
    st.subheader("Genetic Algorithm")
    st.write("A Genetic Algorithm starts with a population of candidate process settings, then repeatedly improves them by selection, crossover, and mutation. It uses the trained machine-learning model as the objective function, which means it searches for the strongest predicted parameter combination rather than a physically measured optimum.")
    st.subheader("Limitations")
    st.write("- Small datasets can make the model unstable and the optimization overly optimistic.")
    st.write("- Extrapolation outside the training range is risky and must be treated as a guess, not a validated optimum.")
    st.write("- A single process window does not guarantee a robust manufacturing setting across all specimens and machine conditions.")
    st.write("- This project is for academic review and for learning the workflow, not a replacement for full experimental validation.")
    st.write("References: placeholder for academic references and project sources.")


def main() -> None:
    """Run the app pages."""
    initialise_session_state()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select a page",
            [
                "Overview",
                "Dataset",
                "Model & Metrics",
                "Predict",
                "Optimize",
                "Results & Report",
                "About / Methodology",
            ],
        )

        st.subheader("Data mode")
        st.session_state["demo_toggle"] = st.toggle("Use illustrative demo data", value=st.session_state.get("demo_toggle", False), help="Turn this on only to test the app with synthetic data.")

        st.caption(f"Student: {STUDENT}")
        st.caption(f"Course: {COURSE}")
        st.caption(f"Printer: {PRINTER_NAME}")

    if page == "Overview":
        render_overview_page()
    elif page == "Dataset":
        render_dataset_page()
    elif page == "Model & Metrics":
        render_model_page()
    elif page == "Predict":
        render_predict_page()
    elif page == "Optimize":
        render_optimization_page()
    elif page == "Results & Report":
        render_results_page()
    elif page == "About / Methodology":
        render_about_page()

    render_footer()


if __name__ == "__main__":
    main()
