
# main.py

from pathlib import Path

from src.environment import UAVEnvironment
from src.ga import GeneticAlgorithm
from src.baseline import RandomSearch
from src.visualize import (
    plot_environment,
    plot_routes,
    plot_convergence,
    plot_fitness_history,
)


def main():
    # Create output folders.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "screenshots").mkdir(parents=True, exist_ok=True)

    # 1. Create the UAV environment.
    environment = UAVEnvironment()

    # 2. Run the Genetic Algorithm.
    ga = GeneticAlgorithm(
        environment=environment,
        population_size=100,
        generations=200,
        mutation_rate=0.05,
        tournament_size=3,
        random_seed=42,
    )

    ga_best, ga_history = ga.run()

    # 3. Run Random Search with the same evaluation budget.
    baseline = RandomSearch(
        environment=environment,
        max_evaluations=20000,
    )

    baseline_best, baseline_history = baseline.run()

    # 4. Plot the environment.
    plot_environment(
        environment,
        show=False,
        save_path=str(results_dir / "environment.png"),
    )

    # 5. Save routes from selected generations.
    snapshot_files = {
        0: "paths_gen0.png",
        25: "paths_gen25.png",
        100: "paths_gen100.png",
        "final": "paths_final.png",
    }

    for generation, filename in snapshot_files.items():
        chromosome = ga.generation_chromosomes.get(generation)

        if chromosome is not None:
            plot_routes(
                environment,
                chromosome,
                show=False,
                save_path=str(results_dir / filename),
            )
            print(f"Saved {filename}")
        else:
            print(
                f"Skipped {filename}: "
                f"generation {generation} was not reached or saved."
            )

    # 6. Save best and average fitness over generations.
    plot_fitness_history(
        ga_history,
        show=False,
        save_path=str(results_dir / "convergence.png"),
    )

    # Keep the original fitness history plot too.
    plot_fitness_history(
        ga_history,
        show=False,
        save_path=str(results_dir / "fitness_history.png"),
    )

    # 7. Compare GA with Random Search.
    plot_convergence(
        ga_history,
        baseline_history,
        ga_population_size=100,
        show=False,
        save_path=str(results_dir / "ga_vs_random.png"),
    )

    print("\nExperiment completed!")
    print(f"Plots saved in: {results_dir.resolve()}")
    print(f"Genetic Algorithm best fitness: {ga_history['best_fitness'][-1]:.4f}")
    print(f"Random Search best fitness: {max(baseline_history['best_fitness']):.4f}")


if __name__ == "__main__":
    main()
