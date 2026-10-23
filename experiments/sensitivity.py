
"""Run sensitivity experiments for the UAV Genetic Algorithm."""
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import inspect
import random
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.environment import UAVEnvironment
from src.ga import GeneticAlgorithm


# Experiment settings
POPULATIONS = [20, 50, 100, 200]
MUTATION_RATES = [0.01, 0.05, 0.1, 0.2]
TOURNAMENT_SIZES = [2, 3, 5, 7]
SEEDS = [42, 123, 456, 789, 2026]
GENERATIONS = 200

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "sensitivity.csv"
PLOT_PATH = RESULTS_DIR / "sensitivity.png"


def accepted_kwargs(callable_object, candidates):
    """Pass only arguments supported by a constructor."""
    signature = inspect.signature(callable_object)
    parameters = signature.parameters
    kwargs = {}

    for possible_names, value in candidates:
        for name in possible_names:
            if name in parameters:
                kwargs[name] = value
                break

    return kwargs


def get_best_fitness(result, algorithm):
    """Extract the best fitness from common GA return formats."""

    # The GA may return (best_chromosome, history).
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict) and "best_fitness" in item:
                values = item["best_fitness"]
                return float(max(values))

    # The GA may return its history dictionary directly.
    if isinstance(result, dict) and "best_fitness" in result:
        return float(max(result["best_fitness"]))

    # Otherwise, inspect the algorithm's stored history.
    history = getattr(algorithm, "history", None)
    if isinstance(history, dict) and "best_fitness" in history:
        return float(max(history["best_fitness"]))

    # Or inspect the returned chromosome/solution.
    if hasattr(result, "fitness"):
        return float(result.fitness)

    raise ValueError(
        "Could not extract best fitness from GeneticAlgorithm.run(). "
        "Check the return value of src/ga.py."
    )


def run_one_experiment(population, mutation, tournament, seed):
    """Run one GA trial with the specified parameters."""

    random.seed(seed)
    np.random.seed(seed)

    environment_kwargs = accepted_kwargs(
        UAVEnvironment.__init__,
        [
            (("random_seed", "seed"), seed),
        ],
    )
    environment = UAVEnvironment(**environment_kwargs)

    ga_kwargs = accepted_kwargs(
        GeneticAlgorithm.__init__,
        [
            (("environment", "env"), environment),
            (("population_size", "pop_size"), population),
            (("generations", "num_generations", "max_generations"),
             GENERATIONS),
            (("mutation_rate",), mutation),
            (("tournament_size",), tournament),
            (("seed", "random_seed", "random_state"), seed),
        ],
    )

    algorithm = GeneticAlgorithm(**ga_kwargs)
    result = algorithm.run()

    return get_best_fitness(result, algorithm)


def run_sensitivity_experiments():
    """Test each parameter independently and save summary statistics."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    experiments = [
        ("population", POPULATIONS),
        ("mutation_rate", MUTATION_RATES),
        ("tournament_size", TOURNAMENT_SIZES),
    ]

    total_runs = sum(len(values) for _, values in experiments) * len(SEEDS)
    completed = 0
    start_time = time.time()

    for parameter, values in experiments:
        for value in values:
            scores = []

            for seed in SEEDS:
                # Hold other parameters at their standard values.
                population = 100
                mutation = 0.05
                tournament = 3

                if parameter == "population":
                    population = value
                elif parameter == "mutation_rate":
                    mutation = value
                elif parameter == "tournament_size":
                    tournament = value

                score = run_one_experiment(
                    population, mutation, tournament, seed
                )
                scores.append(score)

                completed += 1
                print(
                    f"[{completed}/{total_runs}] "
                    f"{parameter}={value}, seed={seed}, "
                    f"best fitness={score:.4f}"
                )

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores, ddof=1))

            rows.append({
                "parameter": parameter,
                "value": value,
                "mean_fitness": mean_score,
                "std_fitness": std_score,
                "number_of_seeds": len(SEEDS),
            })

            print(
                f"  Summary: mean={mean_score:.4f}, "
                f"std={std_score:.4f}\n"
            )

    # Save CSV results.
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "parameter",
                "value",
                "mean_fitness",
                "std_fitness",
                "number_of_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Create a separate chart for each parameter.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (parameter, values) in zip(axes, experiments):
        selected = [
            row for row in rows if row["parameter"] == parameter
        ]

        means = [row["mean_fitness"] for row in selected]
        stds = [row["std_fitness"] for row in selected]

        ax.errorbar(
            values,
            means,
            yerr=stds,
            marker="o",
            capsize=5,
            linewidth=2,
        )
        ax.set_title(parameter.replace("_", " ").title())
        ax.set_xlabel(parameter.replace("_", " ").title())
        ax.set_ylabel("Mean Best Fitness")
        ax.grid(True, alpha=0.3)

    fig.suptitle("UAV Genetic Algorithm: Sensitivity Analysis")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = (time.time() - start_time) / 60

    print("\nSensitivity experiments completed.")
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")
    print(f"Total runs: {completed}")
    print(f"Elapsed time: {elapsed:.1f} minutes")


if __name__ == "__main__":
    run_sensitivity_experiments()
