
# src/fitness.py

import numpy as np


class FitnessEvaluator:
    def __init__(
        self,
        environment,
        max_range=1000.0,
        weights=None,
        sensor_radius=2.0,
        collision_distance=2.0
    ):
        self.environment = environment
        self.max_range = max_range
        self.sensor_radius = sensor_radius
        self.collision_distance = collision_distance

        # Reward and cost weights.
        self.weights = weights or {
            "coverage": 0.30,
            "detection": 0.20,
            "energy": 0.15,
            "time": 0.10,
            "risk": 0.15,
            "penalty": 0.10
        }

    def evaluate(self, chromosome):
        """Calculate normalized fitness terms and the final score."""

        priority = self.environment.priority_map
        hazard = self.environment.hazard_map
        no_fly = self.environment.no_fly_mask
        size = self.environment.grid_size

        points = chromosome.waypoints
        speeds = chromosome.speed
        powers = chromosome.sensor_power

        drones, count, _ = points.shape

        # Clamp values for safe evaluation.
        speeds = np.clip(speeds, 0.1, 10.0)
        powers = np.clip(powers, 0.0, 1.0)

        # ------------------------------------------------
        # 1. COVERAGE: fraction of priority captured
        # ------------------------------------------------
        rows, cols = np.mgrid[0:size, 0:size]
        total_priority = np.sum(priority)

        if total_priority <= 0:
            coverage = 0.0
        else:
            # Combine the sensing footprints of all waypoints.
            uncovered = np.ones((size, size), dtype=float)

            for d in range(drones):
                for i in range(count):
                    x, y = points[d, i]
                    distance_sq = (cols - x) ** 2 + (rows - y) ** 2
                    inside = distance_sq <= self.sensor_radius ** 2

                    # Sensor power controls detection strength.
                    uncovered[inside] *= (1.0 - powers[d, i])

            captured_priority = np.sum(priority * (1.0 - uncovered))
            coverage = captured_priority / total_priority

        # ------------------------------------------------
        # 2. DETECTION: priority-weighted sensor power
        # ------------------------------------------------
        detection_values = []

        for d in range(drones):
            for i in range(count):
                x, y = points[d, i]
                x = int(np.clip(x, 0, size - 1))
                y = int(np.clip(y, 0, size - 1))

                detection_values.append(
                    priority[y, x] * powers[d, i]
                )

        detection = float(np.mean(detection_values))

        # ------------------------------------------------
        # 3. ENERGY COST: movement + sensor energy
        # ------------------------------------------------
        distances = []
        movement_energy = 0.0

        for d in range(drones):
            route = points[d]

            # Include travel from the base to the first waypoint.
            base_row, base_col = self.environment.base_station
            previous = np.array([base_col, base_row])

            for i in range(count):
                current = route[i]
                distance = float(np.linalg.norm(current - previous))
                distances.append(distance)

                # Illustrative movement model: distance * speed^2.
                movement_energy += distance * speeds[d, i] ** 2
                previous = current

        total_distance = sum(distances)
        max_distance = max(
            1.0,
            drones * count * np.sqrt(2) * (size - 1)
        )

        # Normalize energy against a defined reference budget.
        energy_budget = max(1.0, max_distance * 100.0)
        sensor_energy = float(np.sum(powers)) * 10.0

        energy = np.clip(
            (movement_energy + sensor_energy) / energy_budget,
            0.0,
            1.0
        )

        # ------------------------------------------------
        # 4. TIME COST: estimated flight time
        # ------------------------------------------------
        total_time = 0.0

        for d in range(drones):
            previous = np.array([
                self.environment.base_station[1],
                self.environment.base_station[0]
            ])

            for i in range(count):
                current = points[d, i]
                distance = float(np.linalg.norm(current - previous))
                total_time += distance / speeds[d, i]
                previous = current

        # Reference: all drones travel the grid diagonal at speed 1.
        time_budget = max(
            1.0,
            drones * count * np.sqrt(2) * (size - 1)
        )

        time = float(np.clip(total_time / time_budget, 0.0, 1.0))

        # ------------------------------------------------
        # 5. RISK COST: hazard exposure at waypoints
        # ------------------------------------------------
        hazard_values = []

        for d in range(drones):
            for i in range(count):
                x, y = points[d, i]
                x = int(np.clip(x, 0, size - 1))
                y = int(np.clip(y, 0, size - 1))
                hazard_values.append(hazard[y, x])

        risk = float(np.mean(hazard_values))

        # ------------------------------------------------
        # 6. PENALTY: no-fly zones, collisions, range
        # ------------------------------------------------
        no_fly_count = 0
        for d in range(drones):
            for i in range(count):
                x, y = points[d, i]
                x = int(np.clip(x, 0, size - 1))
                y = int(np.clip(y, 0, size - 1))

                if no_fly[y, x]:
                    no_fly_count += 1

        no_fly_penalty = no_fly_count / max(1, drones * count)

        collision_count = 0
        comparisons = 0

        # Penalize drones whose waypoint positions are too close
        # at the same waypoint index.
        for i in range(count):
            for d1 in range(drones):
                for d2 in range(d1 + 1, drones):
                    comparisons += 1
                    distance = np.linalg.norm(
                        points[d1, i] - points[d2, i]
                    )

                    if distance < self.collision_distance:
                        collision_count += 1

        collision_penalty = collision_count / max(1, comparisons)

        # Range penalty uses each drone's total route distance.
        range_violations = 0

        for d in range(drones):
            route_distance = 0.0
            previous = np.array([
                self.environment.base_station[1],
                self.environment.base_station[0]
            ])

            for i in range(count):
                current = points[d, i]
                route_distance += np.linalg.norm(current - previous)
                previous = current

            if route_distance > self.max_range:
                range_violations += 1

        range_penalty = range_violations / max(1, drones)

        penalty = float(np.clip(
            0.60 * no_fly_penalty
            + 0.25 * collision_penalty
            + 0.15 * range_penalty,
            0.0,
            1.0
        ))

        # ------------------------------------------------
        # FINAL SCORE: higher is better
        # ------------------------------------------------
        score = (
            self.weights["coverage"] * coverage
            + self.weights["detection"] * detection
            + self.weights["energy"] * (1.0 - energy)
            + self.weights["time"] * (1.0 - time)
            + self.weights["risk"] * (1.0 - risk)
            + self.weights["penalty"] * (1.0 - penalty)
        )

        # The default weights sum to 1, so score stays in [0, 1].
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "coverage": float(coverage),
            "detection": float(detection),
            "energy": float(energy),
            "time": float(time),
            "risk": float(risk),
            "penalty": float(penalty),
            "score": score
        }


if __name__ == "__main__":
    from src.environment import UAVEnvironment
    from src.chromosome import Chromosome

    env = UAVEnvironment(grid_size=50, random_seed=42)
    chromosome = Chromosome(
        num_drones=4,
        waypoints_per_drone=10,
        grid_size=50,
        random_seed=42
    ).random_init()

    evaluator = FitnessEvaluator(env, max_range=1000.0)
    results = evaluator.evaluate(chromosome)

    print("\nFitness Evaluation Results")
    print("-" * 35)

    for name, value in results.items():
        print(f"{name.capitalize():12s}: {value:.4f}")

    # Check that all reported values are valid and normalized.
    assert all(0.0 <= value <= 1.0 for value in results.values())
    print("\nTest passed: every term is between 0 and 1.")
