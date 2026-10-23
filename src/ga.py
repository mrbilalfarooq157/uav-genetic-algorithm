
# src/ga.py

import numpy as np

from src.chromosome import Chromosome
from src.environment import UAVEnvironment
from src.fitness import FitnessEvaluator
from src.operators import GeneticOperators


class GeneticAlgorithm:
    def __init__(
        self,
        environment=None,
        population_size=100,
        generations=200,
        mutation_rate=0.05,
        tournament_size=3,
        elite_size=2,
        patience=30,
        random_seed=42
    ):
        self.environment = environment or UAVEnvironment(
            grid_size=50,
            random_seed=random_seed
        )

        self.population_size = population_size
        self.generations = generations
        self.elite_size = min(elite_size, population_size)
        self.patience = patience
        self.rng = np.random.default_rng(random_seed)

        self.evaluator = FitnessEvaluator(
            environment=self.environment,
            max_range=1000.0
        )

        self.operators = GeneticOperators(
            environment=self.environment,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            random_seed=random_seed
        )

        self.history = {
            "best_fitness": [],
            "average_fitness": [],
            "diversity": []
        }

        self.best_chromosome = None
        self.best_fitness = -float("inf")

        # Store selected generation snapshots.
        self.generation_chromosomes = {}

    def initialize_population(self):
        """Generate random chromosomes."""
        population = []

        for _ in range(self.population_size):
            chromosome = Chromosome(
                num_drones=4,
                waypoints_per_drone=10,
                grid_size=self.environment.grid_size,
                random_seed=int(self.rng.integers(0, 2**31))
            )

            chromosome.random_init()
            chromosome = self.operators.repair_no_fly(chromosome)
            population.append(chromosome)

        return population

    def evaluate_population(self, population):
        """Return the fitness score of every chromosome."""
        scores = []

        for chromosome in population:
            result = self.evaluator.evaluate(chromosome)
            scores.append(result["score"])

        return np.asarray(scores, dtype=float)

    @staticmethod
    def calculate_diversity(population):
        """Measure normalized genetic diversity."""
        if len(population) <= 1:
            return 0.0

        waypoints = np.asarray(
            [c.waypoints for c in population], dtype=float
        )
        speeds = np.asarray(
            [c.speed for c in population], dtype=float
        )
        sensors = np.asarray(
            [c.sensor_power for c in population], dtype=float
        )

        waypoint_diversity = np.mean(
            np.std(
                waypoints / max(1, population[0].grid_size - 1),
                axis=0
            )
        )

        speed_diversity = np.mean(
            np.std((speeds - 1.0) / 9.0, axis=0)
        )

        sensor_diversity = np.mean(
            np.std(sensors, axis=0)
        )

        diversity = (
            waypoint_diversity
            + speed_diversity
            + sensor_diversity
        ) / 3.0

        return float(np.clip(diversity, 0.0, 1.0))

    def record_statistics(self, population, scores):
        """Record fitness, diversity, and global best."""
        best_index = int(np.argmax(scores))
        generation_best = float(scores[best_index])
        average = float(np.mean(scores))
        diversity = self.calculate_diversity(population)

        self.history["best_fitness"].append(generation_best)
        self.history["average_fitness"].append(average)
        self.history["diversity"].append(diversity)

        if generation_best > self.best_fitness:
            self.best_fitness = generation_best
            self.best_chromosome = population[best_index].copy()

        return generation_best, average, diversity

    def create_next_generation(
        self, population, scores, generation, stagnation
    ):
        """Select, crossover, mutate, repair, and replace."""
        next_population = []

        elite_indices = np.argsort(scores)[-self.elite_size:]

        for index in elite_indices[::-1]:
            next_population.append(population[index].copy())

        mutation_rate = self.operators.adaptive_mutation_rate(
            generation=generation,
            total_generations=self.generations,
            stagnation=stagnation,
            stagnation_limit=max(1, self.patience // 2)
        )

        while len(next_population) < self.population_size:
            parent1 = self.operators.tournament_selection(
                population, scores
            )
            parent2 = self.operators.tournament_selection(
                population, scores
            )

            if self.rng.random() < 0.5:
                child1, child2 = (
                    self.operators.segment_swap_crossover(
                        parent1, parent2
                    )
                )
            else:
                child1, child2 = (
                    self.operators.geometric_crossover(
                        parent1, parent2
                    )
                )

            for child in (child1, child2):
                if self.rng.random() < 0.9:
                    child = self.operators.gaussian_mutation(
                        child,
                        mutation_rate=mutation_rate
                    )

                child = self.operators.repair_no_fly(child)
                next_population.append(child)

                if len(next_population) >= self.population_size:
                    break

        return next_population

    def run(self):
        """Run the GA and return the best chromosome and history."""
        population = self.initialize_population()

        previous_best = -float("inf")
        stagnation = 0

        print("\nStarting UAV Genetic Algorithm")
        print("-" * 72)
        print(
            f"{'Gen':>5} {'Best':>10} "
            f"{'Average':>10} {'Diversity':>12}"
        )
        print("-" * 72)

        for generation in range(self.generations):
            scores = self.evaluate_population(population)

            generation_best, average, diversity = (
                self.record_statistics(population, scores)
            )

            # Generation numbers in the report start at 1.
            completed_generation = generation + 1

            # Save the best chromosome at selected generations.
            if completed_generation in (1, 25, 100):
                best_index = int(np.argmax(scores))
                self.generation_chromosomes[completed_generation - 1] = (
                    population[best_index].copy()
                )

            print(
                f"{completed_generation:5d} "
                f"{generation_best:10.4f} "
                f"{average:10.4f} "
                f"{diversity:12.4f}"
            )

            if generation_best > previous_best + 1e-8:
                previous_best = generation_best
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.patience:
                print(
                    f"\nEarly stopping: no improvement for "
                    f"{self.patience} generations."
                )
                break

            if generation == self.generations - 1:
                break

            population = self.create_next_generation(
                population,
                scores,
                generation + 1,
                stagnation
            )

        # Always save the best chromosome found across all generations.
        self.generation_chromosomes["final"] = (
            self.best_chromosome.copy()
        )

        print("-" * 72)
        print(f"Best fitness found: {self.best_fitness:.4f}")

        return self.best_chromosome, self.history


if __name__ == "__main__":
    ga = GeneticAlgorithm(
        population_size=10,
        generations=5,
        mutation_rate=0.05,
        tournament_size=3,
        elite_size=2,
        patience=10,
        random_seed=42
    )

    best_solution, history = ga.run()

    assert best_solution is not None
    assert len(history["best_fitness"]) > 0

    assert (
        len(history["best_fitness"])
        == len(history["average_fitness"])
        == len(history["diversity"])
    )

    assert all(
        0.0 <= value <= 1.0
        for value in history["diversity"]
    )

    print("\nTest passed.")
    print("Best chromosome shape:", best_solution.waypoints.shape)
    print("Recorded generations:", len(history["best_fitness"]))
