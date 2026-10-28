"""Hand-written genetic algorithm for optimizing 3D printing parameters.

This module is intentionally implemented without external GA libraries. It uses
NumPy and the trained Random Forest model to search for a high-strength process
window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import (
    MATERIAL_LIMITS,
    PARAM_STEP,
    PRINTER_LIMITS,
    TENSILE_STRENGTH,
    validate_layer_height,
)


@dataclass
class GAResult:
    """Container for the final GA output and diagnostic details."""
    best_params: dict[str, Any]
    best_fitness: float
    history: pd.DataFrame
    final_population: pd.DataFrame
    top_5_solutions: pd.DataFrame
    seed: int
    settings: dict[str, Any]


def _snap_to_step(value: float, step: float) -> float:
    """Snap a continuous value to the nearest allowed step."""
    if step <= 0:
        return float(value)
    return round(float(value) / step) * step


def _repair_individual(individual: dict[str, Any], bounds: dict[str, Any], material_mode: str, selected_material: str, nozzle_diameter: float) -> dict[str, Any]:
    """Clip to bounds, snap to step, and enforce FDM safety constraints."""
    repaired = dict(individual)

    for key in ["layer_height", "nozzle_temperature", "print_speed", "infill_density"]:
        lo, hi = bounds[key]
        repaired[key] = float(np.clip(repaired[key], lo, hi))
        repaired[key] = _snap_to_step(repaired[key], PARAM_STEP.get(key, 1.0))

    if repaired["layer_height"] > 0.8 * nozzle_diameter:
        repaired["layer_height"] = float(np.clip(0.8 * nozzle_diameter, bounds["layer_height"][0], bounds["layer_height"][1]))
        repaired["layer_height"] = _snap_to_step(repaired["layer_height"], PARAM_STEP["layer_height"])

    material = str(repaired.get("material", selected_material)).upper()
    if material in MATERIAL_LIMITS:
        lo, hi = MATERIAL_LIMITS[material]
        repaired["nozzle_temperature"] = float(np.clip(repaired["nozzle_temperature"], lo, hi))

    if material_mode == "fixed":
        repaired["material"] = selected_material.upper()
    else:
        repaired["material"] = material

    return repaired


def _initial_population(bounds: dict[str, tuple[float, float]], material_mode: str, selected_material: str, original_params: dict[str, Any], population_size: int, seed: int) -> list[dict[str, Any]]:
    """Create the first population plus the user's original parameters as one candidate."""
    rng = np.random.default_rng(seed)
    population: list[dict[str, Any]] = []

    for _ in range(population_size):
        genes = {}
        for key in ["layer_height", "nozzle_temperature", "print_speed", "infill_density"]:
            lo, hi = bounds[key]
            value = float(rng.uniform(lo, hi))
            genes[key] = value
        genes["material"] = selected_material.upper() if material_mode == "fixed" else rng.choice(list(bounds["material"].keys()))
        population.append(_repair_individual(genes, bounds, material_mode, selected_material, 0.4))

    if original_params:
        original = {key: float(value) for key, value in original_params.items() if key in bounds}
        original["material"] = str(original_params.get("material", selected_material)).upper()
        population.append(_repair_individual(original, bounds, material_mode, selected_material, 0.4))

    return population


def _batch_predict(model: Any, population: list[dict[str, Any]]) -> np.ndarray:
    """Batch-predict tensile strength for a list of parameter dictionaries."""
    df = pd.DataFrame(population)
    if "material" not in df.columns:
        return np.zeros(len(df))
    df = df.rename(columns={
        "layer_height": "layer_height",
        "nozzle_temperature": "nozzle_temperature",
        "print_speed": "print_speed",
        "infill_density": "infill_density",
        "material": "material_type",
    })
    predictions = model.predict(df)
    return np.asarray(predictions, dtype=float)


def _tournament_selection(population: list[dict[str, Any]], fitness: np.ndarray, tournament_size: int, rng: np.random.Generator) -> dict[str, Any]:
    """Select the winner of a tournament of size k."""
    tournament_indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
    return population[best_idx]


def _blend_crossover(parent_a: dict[str, Any], parent_b: dict[str, Any], alpha: float = 0.3, rng: np.random.Generator = None) -> dict[str, Any]:
    """Blend crossover for continuous parameters using BLX-alpha."""
    if rng is None:
        rng = np.random.default_rng()
    child = {}
    for key in ["layer_height", "nozzle_temperature", "print_speed", "infill_density"]:
        lo = min(parent_a[key], parent_b[key])
        hi = max(parent_a[key], parent_b[key])
        width = hi - lo
        new_min = lo - alpha * width
        new_max = hi + alpha * width
        child[key] = float(rng.uniform(new_min, new_max))
    child["material"] = parent_a["material"] if rng.random() < 0.5 else parent_b["material"]
    return child


def _mutate_gene(value: float, scale: float, rng: np.random.Generator) -> float:
    """Apply Gaussian mutation to a continuous gene."""
    return float(value + rng.normal(0, scale))


def run_ga(
    model: Any,
    bounds: dict[str, Any],
    material_mode: str,
    selected_material: str,
    original_params: dict[str, Any],
    pop_size: int = 60,
    generations: int = 40,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    elitism: int = 2,
    seed: int = 42,
    progress_callback: Callable[[float, str], None] | None = None,
) -> GAResult:
    """Run a hand-written genetic algorithm and return the best solution and diagnostics."""
    rng = np.random.default_rng(seed)
    population = _initial_population(bounds, material_mode, selected_material, original_params, pop_size, seed)

    history_records: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_fitness = -np.inf
    global_best = None

    for gen in range(generations):
        fitness = _batch_predict(model, population)
        if len(fitness) == 0:
            break

        best_index = int(np.argmax(fitness))
        best_candidate = population[best_index]
        best_value = float(fitness[best_index])
        mean_value = float(np.mean(fitness))
        worst_value = float(np.min(fitness))

        if best_value > best_fitness:
            best_fitness = best_value
            best_params = dict(best_candidate)
            global_best = dict(best_candidate)

        history_records.append({
            "generation": gen + 1,
            "best": best_value,
            "mean": mean_value,
            "worst": worst_value,
        })

        if progress_callback is not None:
            progress_callback((gen + 1) / generations, f"Generation {gen + 1}/{generations}")

        new_population: list[dict[str, Any]] = []
        sorted_indices = np.argsort(fitness)[::-1]
        elites = [population[i] for i in sorted_indices[:elitism]]
        new_population.extend(elites)

        while len(new_population) < pop_size:
            if rng.random() < crossover_rate:
                parent_a = _tournament_selection(population, fitness, tournament_size, rng)
                parent_b = _tournament_selection(population, fitness, tournament_size, rng)
                child = _blend_crossover(parent_a, parent_b, alpha=0.3, rng=rng)
            else:
                child = dict(rng.choice(population))

            for key in ["layer_height", "nozzle_temperature", "print_speed", "infill_density"]:
                if rng.random() < mutation_rate:
                    param_range = bounds[key][1] - bounds[key][0]
                    sigma = 0.10 * param_range
                    child[key] = _mutate_gene(child[key], sigma, rng)

            if material_mode == "search":
                if rng.random() < mutation_rate:
                    child["material"] = rng.choice(list(bounds["material"].keys()))
            else:
                child["material"] = selected_material.upper()

            child = _repair_individual(child, bounds, material_mode, selected_material, 0.4)
            new_population.append(child)

        population = new_population[:pop_size]

    if best_params is None:
        best_params = dict(population[0])

    final_population = pd.DataFrame(population)
    final_population = final_population.rename(columns={"material": "material_type"})
    final_population["material_type"] = final_population["material_type"].astype(str).str.upper()

    history_df = pd.DataFrame(history_records)
    distinct = final_population.drop_duplicates().sort_values(by=["layer_height", "nozzle_temperature", "print_speed", "infill_density"], ascending=False)
    top_solution_table = distinct.head(5).copy()
    top_solution_table["predicted_tensile_strength"] = np.nan

    settings = {
        "pop_size": pop_size,
        "generations": generations,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "tournament_size": tournament_size,
        "elitism": elitism,
        "seed": seed,
        "material_mode": material_mode,
        "selected_material": selected_material,
    }

    return GAResult(
        best_params=best_params,
        best_fitness=best_fitness,
        history=history_df,
        final_population=final_population,
        top_5_solutions=top_solution_table,
        seed=seed,
        settings=settings,
    )
