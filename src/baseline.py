
# src/baseline.py

import numpy as np

from src.environment import UAVEnvironment
from src.chromosome import Chromosome
from src.fitness import FitnessEvaluator


class RandomSearch:
    def __init__(
        self,
        environment=None,
        max_evaluations=20000,
        max_range=1000.0,
        random_seed=42
    ):
        self.environment = environment or UAVEnvironment(
            grid_size=50,
            random_seed=random_seed
        )

        self.max_evaluations = max_evaluations
        self.max_range = max_range
        self.rng = np.random.default_rng(random_seed)

        self.evaluator = FitnessEvaluator(
            environment=self.environment,
            max_range=self.max_range
        )

        self.best_chromosome = None
        self.best_fitness = -float("inf")

        self.history = {
            "best_fitness": [],
            "average_fitness": [],
            "evaluations": []
        }

    def random_chromosome(self):
        """Generate one random candidate solution."""

        seed = int(self.rng.integers(0, 2**31))

        chromosome = Chromosome(
            num_drones=4,
            waypoints_per_drone=10,
            grid_size=self.environment.grid_size,
            random_seed=seed
        )

        chromosome.random_init()

        # Match the GA's initial waypoint repair policy.
        mask = self.environment.no_fly_mask
        safe_cells = np.argwhere(~mask)

        for drone in range(chromosome.num_drones):
            for i in range(chromosome.waypoints_per_drone):
                x, y = chromosome.waypoints[drone, i]

                if mask[y, x]:
                    distances_sq = (
                        (safe_cells[:, 1] - x) ** 2
                        + (safe_cells[:, 0] - y) ** 2
                    )

                    nearest = safe_cells[np.argmin(distances_sq)]
                    chromosome.waypoints[drone, i] = [
                        int(nearest[1]),
                        int(nearest[0])
                    ]

        return chromosome

    def run(self):
        """Evaluate random solutions within the evaluation budget."""

        running_total = 0.0

        print("\nStarting Random Search Baseline")
        print("-" * 55)

        for evaluation in range(1, self.max_evaluations + 1):
            chromosome = self.random_chromosome()

            result = self.evaluator.evaluate(chromosome)
            score = result["score"]

            running_total += score

            if score > self.best_fitness:
                self.best_fitness = score
                self.best_chromosome = chromosome.copy()

            # Record progress after every evaluation.
            self.history["best_fitness"].append(
                self.best_fitness
            )
            self.history["average_fitness"].append(
                running_total / evaluation
            )
            self.history["evaluations"].append(evaluation)

            if (
                evaluation == 1
                or evaluation % 1000 == 0
                or evaluation == self.max_evaluations
            ):
                print(
                    f"Evaluation {evaluation:6d} | "
                    f"Current best: {self.best_fitness:.4f} | "
                    f"Running average: "
                    f"{running_total / evaluation:.4f}"
                )

        print("-" * 55)
        print(f"Total evaluations: {self.max_evaluations}")
        print(f"Best random-search fitness: {self.best_fitness:.4f}")

        return self.best_chromosome, self.history


if __name__ == "__main__":
    # Small smoke test.
    # Set to 20000 for the full comparison with a 100 x 200 GA.
    baseline = RandomSearch(
        max_evaluations=50,
        max_range=1000.0,
        random_seed=42
    )

    best_solution, history = baseline.run()

    assert best_solution is not None
    assert len(history["best_fitness"]) == 50
    assert len(history["average_fitness"]) == 50
    assert len(history["evaluations"]) == 50

    assert all(
        history["best_fitness"][i]
        <= history["best_fitness"][i + 1] + 1e-12
        for i in range(len(history["best_fitness"]) - 1)
    )

    print("\nRandom search test passed.")
    print("Best chromosome shape:", best_solution.waypoints.shape)
