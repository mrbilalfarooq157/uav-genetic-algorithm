"""
UAV Environment
---------------
Creates a 50x50 grid with:
- Priority map
- Hazard map
- No-fly mask
- Base station
- Matplotlib visualization

Run from the project root:
    python -m src.environment
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


@dataclass
class UAVEnvironment:
    grid_size: int = 50
    random_seed: int = 42

    def __post_init__(self):
        if self.grid_size < 10:
            raise ValueError("grid_size should be at least 10.")

        self.rng = np.random.default_rng(self.random_seed)

        # Priority values: 0 = low priority, 1 = high priority.
        self.priority_map = self._build_priority_map()

        # Hazard values: 0.0 = safe, 1.0 = highest hazard.
        self.hazard_map = self._build_hazard_map()

        # True means a drone is forbidden from entering this cell.
        self.no_fly_mask = self._build_no_fly_mask()

        # Base station at the bottom-left corner (row, column).
        self.base_station = (self.grid_size - 3, 2)

        # Ensure the base station is always flyable and not hazardous.
        br, bc = self.base_station
        self.no_fly_mask[br, bc] = False
        self.hazard_map[br, bc] = 0.0

    def _build_priority_map(self):
        """Create smooth high-priority zones around selected locations."""
        size = self.grid_size
        y, x = np.mgrid[0:size, 0:size]

        centers = [
            (size * 0.22, size * 0.25, 1.00, size * 0.10),
            (size * 0.70, size * 0.30, 0.85, size * 0.12),
            (size * 0.55, size * 0.75, 0.95, size * 0.11),
            (size * 0.82, size * 0.78, 0.75, size * 0.09),
        ]

        priority = np.zeros((size, size), dtype=float)
        for cy, cx, strength, spread in centers:
            distance_sq = (y - cy) ** 2 + (x - cx) ** 2
            priority += strength * np.exp(-distance_sq / (2 * spread**2))

        # Normalize values to the range [0, 1].
        if priority.max() > 0:
            priority /= priority.max()

        return priority

    def _build_hazard_map(self):
        """Create a continuous hazard field with several hazard hotspots."""
        size = self.grid_size
        y, x = np.mgrid[0:size, 0:size]

        centers = [
            (size * 0.38, size * 0.60, 1.00, size * 0.07),
            (size * 0.76, size * 0.52, 0.80, size * 0.06),
            (size * 0.30, size * 0.83, 0.65, size * 0.05),
        ]

        hazard = self.rng.uniform(0.0, 0.10, size=(size, size))
        for cy, cx, strength, spread in centers:
            distance_sq = (y - cy) ** 2 + (x - cx) ** 2
            hazard += strength * np.exp(-distance_sq / (2 * spread**2))

        return np.clip(hazard, 0.0, 1.0)

    def _build_no_fly_mask(self):
        """Create rectangular no-fly zones; True marks prohibited cells."""
        size = self.grid_size
        mask = np.zeros((size, size), dtype=bool)

        # Fixed zones, scaled to work for different grid sizes.
        zones = [
            (0.12, 0.20, 0.12, 0.28),
            (0.42, 0.52, 0.16, 0.22),
            (0.68, 0.78, 0.72, 0.84),
        ]

        for r0, r1, c0, c1 in zones:
            row_start = int(r0 * size)
            row_end = int(r1 * size)
            col_start = int(c0 * size)
            col_end = int(c1 * size)
            mask[row_start:row_end, col_start:col_end] = True

        return mask

    def plot(self, show=True):
        """Plot priority, hazard, no-fly areas, and base station."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

        priority_img = axes[0, 0].imshow(
            self.priority_map, cmap="YlGn", origin="upper", vmin=0, vmax=1
        )
        axes[0, 0].set_title("Priority Map (0 = low, 1 = high)")
        fig.colorbar(priority_img, ax=axes[0, 0], fraction=0.046, pad=0.04)

        hazard_img = axes[0, 1].imshow(
            self.hazard_map, cmap="YlOrRd", origin="upper", vmin=0, vmax=1
        )
        axes[0, 1].set_title("Hazard Map (0 = safe, 1 = high hazard)")
        fig.colorbar(hazard_img, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Green = flyable, dark gray = no-fly.
        no_fly_img = axes[1, 0].imshow(
            self.no_fly_mask,
            cmap=ListedColormap(["#d9ead3", "#333333"]),
            origin="upper",
            vmin=0,
            vmax=1,
        )
        axes[1, 0].set_title("No-Fly Mask (dark = forbidden)")
        axes[1, 0].set_xlabel("Column")
        axes[1, 0].set_ylabel("Row")

        # Combined view: priority underneath, hazards and no-fly zones overlaid.
        axes[1, 1].imshow(
            self.priority_map, cmap="YlGn", origin="upper", vmin=0, vmax=1
        )
        hazard_overlay = np.ma.masked_where(self.hazard_map < 0.35, self.hazard_map)
        axes[1, 1].imshow(
            hazard_overlay, cmap="Reds", alpha=0.55, origin="upper", vmin=0, vmax=1
        )
        no_fly_overlay = np.ma.masked_where(~self.no_fly_mask, self.no_fly_mask)
        axes[1, 1].imshow(
            no_fly_overlay,
            cmap=ListedColormap(["#333333"]),
            alpha=0.90,
            origin="upper",
            vmin=1,
            vmax=1,
        )

        br, bc = self.base_station
        for ax in axes.flat:
            ax.scatter(
                bc, br, marker="*", s=220, c="deepskyblue",
                edgecolors="black", linewidths=0.8, label="Base station"
            )
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(self.grid_size - 0.5, -0.5)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_aspect("equal")

        axes[1, 1].set_title("Combined Map (priority + hazards + no-fly)")
        axes[1, 1].legend(loc="upper right")
        fig.suptitle(f"UAV Environment — {self.grid_size}x{self.grid_size} Grid", fontsize=15)

        if show:
            plt.show()
        return fig, axes

    def validate(self):
        """Run basic sanity checks; raises AssertionError if something is wrong."""
        expected_shape = (self.grid_size, self.grid_size)
        assert self.priority_map.shape == expected_shape
        assert self.hazard_map.shape == expected_shape
        assert self.no_fly_mask.shape == expected_shape
        assert np.all((self.priority_map >= 0) & (self.priority_map <= 1))
        assert np.all((self.hazard_map >= 0) & (self.hazard_map <= 1))
        assert self.no_fly_mask.dtype == bool

        br, bc = self.base_station
        assert 0 <= br < self.grid_size and 0 <= bc < self.grid_size
        assert not self.no_fly_mask[br, bc]
        assert self.hazard_map[br, bc] == 0.0

        return True


if __name__ == "__main__":
    env = UAVEnvironment(grid_size=50, random_seed=42)
    env.validate()
    print("Environment validation passed.")
    print(f"Grid size: {env.grid_size} x {env.grid_size}")
    print(f"Priority map shape: {env.priority_map.shape}")
    print(f"Hazard map shape: {env.hazard_map.shape}")
    print(f"No-fly cells: {int(env.no_fly_mask.sum())}")
    print(f"Base station (row, column): {env.base_station}")
    env.plot()
