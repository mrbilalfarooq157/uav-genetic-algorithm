
# src/operators.py

import numpy as np


class GeneticOperators:
    def __init__(
        self,
        environment=None,
        tournament_size=3,
        mutation_rate=0.05,
        min_mutation_rate=0.01,
        max_mutation_rate=0.30,
        random_seed=42
    ):
        self.environment = environment
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.rng = np.random.default_rng(random_seed)

    # --------------------------------------------------
    # 1. TOURNAMENT SELECTION
    # --------------------------------------------------
    def tournament_selection(self, population, fitness_scores):
        """Select the chromosome with the highest fitness."""

        if len(population) == 0:
            raise ValueError("Population cannot be empty.")

        if len(population) != len(fitness_scores):
            raise ValueError("Population and fitness sizes must match.")

        size = min(self.tournament_size, len(population))

        candidates = self.rng.choice(
            len(population),
            size=size,
            replace=False
        )

        winner_index = max(
            candidates,
            key=lambda i: fitness_scores[i]
        )

        return population[winner_index]

    # --------------------------------------------------
    # 2. SEGMENT-SWAP CROSSOVER
    # --------------------------------------------------
    def segment_swap_crossover(self, parent1, parent2):
        """Swap a route segment to create two children."""

        child1 = parent1.copy()
        child2 = parent2.copy()

        num_waypoints = parent1.waypoints_per_drone

        if num_waypoints < 2:
            return child1, child2

        # Select one drone and a segment of its route.
        drone = self.rng.integers(parent1.num_drones)

        start = self.rng.integers(0, num_waypoints - 1)
        end = self.rng.integers(start + 1, num_waypoints + 1)

        # Exchange waypoint coordinates.
        child1.waypoints[drone, start:end] = (
            parent2.waypoints[drone, start:end].copy()
        )
        child2.waypoints[drone, start:end] = (
            parent1.waypoints[drone, start:end].copy()
        )

        # Exchange the corresponding speed and sensor settings.
        child1.speed[drone, start:end] = (
            parent2.speed[drone, start:end].copy()
        )
        child2.speed[drone, start:end] = (
            parent1.speed[drone, start:end].copy()
        )

        child1.sensor_power[drone, start:end] = (
            parent2.sensor_power[drone, start:end].copy()
        )
        child2.sensor_power[drone, start:end] = (
            parent1.sensor_power[drone, start:end].copy()
        )

        return child1, child2

    # --------------------------------------------------
    # 3. GEOMETRIC CROSSOVER
    # --------------------------------------------------
    def geometric_crossover(self, parent1, parent2):
        """Blend parent genes using weighted interpolation."""

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Each gene receives a random interpolation weight.
        alpha = self.rng.random(parent1.waypoints.shape)

        blended1 = (
            alpha * parent1.waypoints
            + (1.0 - alpha) * parent2.waypoints
        )

        blended2 = (
            alpha * parent2.waypoints
            + (1.0 - alpha) * parent1.waypoints
        )

        child1.waypoints = np.rint(blended1).astype(int)
        child2.waypoints = np.rint(blended2).astype(int)

        # Blend speeds and sensor powers.
        alpha_speed = self.rng.random(parent1.speed.shape)

        child1.speed = (
            alpha_speed * parent1.speed
            + (1.0 - alpha_speed) * parent2.speed
        )
        child2.speed = (
            alpha_speed * parent2.speed
            + (1.0 - alpha_speed) * parent1.speed
        )

        alpha_sensor = self.rng.random(parent1.sensor_power.shape)

        child1.sensor_power = (
            alpha_sensor * parent1.sensor_power
            + (1.0 - alpha_sensor) * parent2.sensor_power
        )
        child2.sensor_power = (
            alpha_sensor * parent2.sensor_power
            + (1.0 - alpha_sensor) * parent1.sensor_power
        )

        # Keep genes inside their valid bounds.
        self._clip_genes(child1)
        self._clip_genes(child2)

        return child1, child2

    # --------------------------------------------------
    # 4. GAUSSIAN MUTATION
    # --------------------------------------------------
    def gaussian_mutation(
        self,
        chromosome,
        mutation_rate=None,
        waypoint_sigma=3.0,
        speed_sigma=0.5,
        sensor_sigma=0.1
    ):
        """Randomly perturb route, speed, and sensor genes."""

        child = chromosome.copy()

        rate = (
            self.mutation_rate
            if mutation_rate is None
            else mutation_rate
        )

        rate = float(np.clip(rate, 0.0, 1.0))

        # Mutate waypoint coordinates.
        mask = self.rng.random(child.waypoints.shape[:2]) < rate
        noise = self.rng.normal(
            0.0, waypoint_sigma, size=child.waypoints.shape
        )

        child.waypoints = np.rint(
            child.waypoints + noise * mask[:, :, None]
        ).astype(int)

        # Mutate speed values.
        speed_mask = self.rng.random(child.speed.shape) < rate
        child.speed += (
            self.rng.normal(0.0, speed_sigma, child.speed.shape)
            * speed_mask
        )

        # Mutate sensor power values.
        sensor_mask = self.rng.random(child.sensor_power.shape) < rate
        child.sensor_power += (
            self.rng.normal(
                0.0, sensor_sigma, child.sensor_power.shape
            )
            * sensor_mask
        )

        self._clip_genes(child)

        return child

    # --------------------------------------------------
    # 5. REPAIR NO-FLY WAYPOINTS
    # --------------------------------------------------
    def repair_no_fly(self, chromosome):
        """Move invalid or no-fly waypoints to the nearest safe cell."""

        child = chromosome.copy()

        if self.environment is None:
            raise ValueError(
                "Pass an environment to repair no-fly waypoints."
            )

        mask = self.environment.no_fly_mask
        grid_size = self.environment.grid_size

        # Safe coordinates are stored as (row, column).
        safe_cells = np.argwhere(~mask)

        if len(safe_cells) == 0:
            raise ValueError("The environment has no safe cells.")

        for drone in range(child.num_drones):
            for i in range(child.waypoints_per_drone):
                x, y = child.waypoints[drone, i]

                # Clamp coordinates to the grid.
                x = int(np.clip(x, 0, grid_size - 1))
                y = int(np.clip(y, 0, grid_size - 1))

                # True in the mask means this cell is forbidden.
                if mask[y, x]:
                    distances_sq = (
                        (safe_cells[:, 1] - x) ** 2
                        + (safe_cells[:, 0] - y) ** 2
                    )

                    nearest_index = np.argmin(distances_sq)
                    nearest_row, nearest_col = safe_cells[nearest_index]

                    x = int(nearest_col)
                    y = int(nearest_row)

                child.waypoints[drone, i] = [x, y]

        # A valid route must preserve its base station.
        # This operator repairs waypoints, not entire flight paths.
        return child

    # --------------------------------------------------
    # 6. ADAPTIVE MUTATION
    # --------------------------------------------------
    def adaptive_mutation_rate(
        self,
        generation,
        total_generations,
        stagnation=0,
        stagnation_limit=15
    ):
        """
        Increase mutation during stagnation.
        Gradually reduce it as evolution progresses.
        """

        progress = np.clip(
            generation / max(1, total_generations),
            0.0,
            1.0
        )

        # Decay from the configured maximum towards the minimum.
        rate = (
            self.max_mutation_rate
            - progress
            * (self.max_mutation_rate - self.min_mutation_rate)
        )

        # Increase exploration when fitness stops improving.
        if stagnation_limit > 0:
            stagnation_factor = min(
                1.0,
                max(0, stagnation) / stagnation_limit
            )
            rate += stagnation_factor * (
                self.max_mutation_rate - rate
            )

        self.mutation_rate = float(np.clip(
            rate,
            self.min_mutation_rate,
            self.max_mutation_rate
        ))

        return self.mutation_rate

    # --------------------------------------------------
    # HELPER: KEEP GENES VALID
    # --------------------------------------------------
    @staticmethod
    def _clip_genes(chromosome):
        """Enforce grid, speed, and sensor bounds."""

        chromosome.waypoints = np.clip(
            chromosome.waypoints,
            0,
            chromosome.grid_size - 1
        ).astype(int)

        chromosome.speed = np.clip(
            chromosome.speed, 1.0, 10.0
        )

        chromosome.sensor_power = np.clip(
            chromosome.sensor_power, 0.0, 1.0
        )


# ------------------------------------------------------
# TEST ALL OPERATORS
# ------------------------------------------------------
if __name__ == "__main__":
    from src.environment import UAVEnvironment
    from src.chromosome import Chromosome
    from src.fitness import FitnessEvaluator

    env = UAVEnvironment(grid_size=50, random_seed=42)

    operators = GeneticOperators(
        environment=env,
        tournament_size=3,
        mutation_rate=0.05,
        random_seed=42
    )

    # Create a small test population.
    population = [
        Chromosome(
            num_drones=4,
            waypoints_per_drone=10,
            grid_size=50,
            random_seed=i
        ).random_init()
        for i in range(10)
    ]

    evaluator = FitnessEvaluator(env)

    scores = [
        evaluator.evaluate(chromosome)["score"]
        for chromosome in population
    ]

    # Tournament selection.
    selected = operators.tournament_selection(population, scores)

    # Segment-swap crossover.
    child1, child2 = operators.segment_swap_crossover(
        population[0], population[1]
    )

    # Geometric crossover.
    child3, child4 = operators.geometric_crossover(
        population[0], population[1]
    )

    # Gaussian mutation.
    mutated = operators.gaussian_mutation(child3)

    # No-fly repair.
    repaired = operators.repair_no_fly(mutated)

    # Adaptive mutation.
    rate = operators.adaptive_mutation_rate(
        generation=100,
        total_generations=200,
        stagnation=10
    )

    # Validation checks.
    for candidate in [selected, child1, child2, child3,
                      child4, mutated, repaired]:
        assert candidate.waypoints.shape == (4, 10, 2)
        assert candidate.speed.shape == (4, 10)
        assert candidate.sensor_power.shape == (4, 10)

        assert np.all(candidate.waypoints >= 0)
        assert np.all(candidate.waypoints < 50)
        assert np.all(candidate.speed >= 1.0)
        assert np.all(candidate.speed <= 10.0)
        assert np.all(candidate.sensor_power >= 0.0)
        assert np.all(candidate.sensor_power <= 1.0)

    # Verify repaired waypoints avoid no-fly cells.
    for route in repaired.waypoints:
        for x, y in route:
            assert not env.no_fly_mask[y, x]

    print("All operator tests passed.")
    print(f"Best initial fitness: {max(scores):.4f}")
    print(f"Adaptive mutation rate: {rate:.4f}")
    print("Repaired route has no waypoints inside no-fly cells.")
