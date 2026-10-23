
# src/chromosome.py

import numpy as np


class Chromosome:
    def __init__(
        self,
        num_drones=4,
        waypoints_per_drone=10,
        grid_size=50,
        random_seed=None
    ):
        self.num_drones = num_drones
        self.waypoints_per_drone = waypoints_per_drone
        self.grid_size = grid_size

        self.rng = np.random.default_rng(random_seed)

        # Waypoint coordinates: (x, y) for each drone.
        self.waypoints = np.zeros(
            (num_drones, waypoints_per_drone, 2),
            dtype=int
        )

        # Speed for each leg between consecutive waypoints.
        self.speed = np.zeros(
            (num_drones, waypoints_per_drone),
            dtype=float
        )

        # Sensor power at each waypoint.
        self.sensor_power = np.zeros(
            (num_drones, waypoints_per_drone),
            dtype=float
        )

    def random_init(self):
        """Generate random routes, speeds, and sensor power."""

        # Random waypoint coordinates within the grid.
        self.waypoints = self.rng.integers(
            0,
            self.grid_size,
            size=(self.num_drones, self.waypoints_per_drone, 2)
        )

        # Random speed for each leg (assumed range: 1 to 10).
        self.speed = self.rng.uniform(
            1.0,
            10.0,
            size=(self.num_drones, self.waypoints_per_drone)
        )

        # Random sensor power at each waypoint (range: 0 to 1).
        self.sensor_power = self.rng.uniform(
            0.0,
            1.0,
            size=(self.num_drones, self.waypoints_per_drone)
        )

        return self

    def decode(self):
        """Convert chromosome arrays into readable drone routes."""

        decoded = []

        for drone in range(self.num_drones):
            route = []

            for i in range(self.waypoints_per_drone):
                x, y = self.waypoints[drone, i]

                route.append({
                    "waypoint": (int(x), int(y)),
                    "speed": float(self.speed[drone, i]),
                    "sensor_power": float(
                        self.sensor_power[drone, i]
                    )
                })

            decoded.append({
                "drone_id": drone + 1,
                "route": route
            })

        return decoded

    def copy(self):
        """Create an independent copy of this chromosome."""

        new_chromosome = Chromosome(
            num_drones=self.num_drones,
            waypoints_per_drone=self.waypoints_per_drone,
            grid_size=self.grid_size
        )

        new_chromosome.waypoints = self.waypoints.copy()
        new_chromosome.speed = self.speed.copy()
        new_chromosome.sensor_power = self.sensor_power.copy()

        return new_chromosome


if __name__ == "__main__":
    chromosome = Chromosome(
        num_drones=4,
        waypoints_per_drone=10,
        grid_size=50,
        random_seed=42
    )

    chromosome.random_init()

    print("Number of drones:", chromosome.num_drones)
    print("Waypoints shape:", chromosome.waypoints.shape)
    print("Speed shape:", chromosome.speed.shape)
    print("Sensor power shape:", chromosome.sensor_power.shape)

    print("\nFirst drone's route:")
    for point in chromosome.decode()[0]["route"]:
        print(point)

    copied = chromosome.copy()
    print("\nCopy created successfully:", copied is not chromosome)
