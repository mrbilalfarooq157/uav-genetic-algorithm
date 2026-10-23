
"""Visualization utilities for the UAV Genetic Algorithm project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_save_path(save_path):
    """Create the output directory when a save path is provided."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return None


def plot_environment(environment, show=True, save_path=None):
    """Display the priority map, hazard map, and no-fly zones."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(environment.priority_map, cmap="YlOrRd", origin="upper")
    axes[0].set_title("Priority Map")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].imshow(environment.hazard_map, cmap="Reds", origin="upper")
    axes[1].set_title("Hazard Map")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    axes[2].imshow(environment.no_fly_mask, cmap="gray_r", origin="upper")
    axes[2].set_title("No-Fly Zones")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    row, col = environment.base_station
    axes[2].scatter(col, row, color="blue", marker="*", s=180,
                    label="Base Station")
    axes[2].legend()

    fig.tight_layout()

    path = _prepare_save_path(save_path)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_routes(environment, chromosome, show=True, save_path=None):
    """Plot all drone routes over the priority map."""

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.imshow(environment.priority_map, cmap="YlOrRd", origin="upper")
    ax.imshow(
        np.ma.masked_where(~environment.no_fly_mask,
                           environment.no_fly_mask),
        cmap="gray",
        alpha=0.65,
        origin="upper"
    )

    row, col = environment.base_station
    base = (col, row)

    ax.scatter(
        base[0], base[1], color="blue", marker="*",
        s=250, label="Base Station", zorder=5
    )

    colors = plt.get_cmap("tab10")

    for drone_index, waypoints in enumerate(chromosome.waypoints):
        color = colors(drone_index % 10)

        # Add the base station as the route's starting point.
        x_values = [base[0]] + [int(point[0]) for point in waypoints]
        y_values = [base[1]] + [int(point[1]) for point in waypoints]

        ax.plot(
            x_values, y_values,
            color=color, marker="o", markersize=3,
            linewidth=1.5, label=f"Drone {drone_index + 1}"
        )

        ax.scatter(
            x_values[1:], y_values[1:],
            color=[color], s=20
        )

    ax.set_title("UAV Drone Routes")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xlim(-0.5, environment.grid_size - 0.5)
    ax.set_ylim(environment.grid_size - 0.5, -0.5)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    path = _prepare_save_path(save_path)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_convergence(
    ga_history,
    baseline_history=None,
    ga_population_size=100,
    show=True,
    save_path=None
):
    """Compare best fitness over the evaluation budget."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ga_best = ga_history["best_fitness"]
    ga_evaluations = (
        np.arange(1, len(ga_best) + 1) * ga_population_size
    )

    ax.plot(
        ga_evaluations, ga_best,
        label="Genetic Algorithm",
        linewidth=2
    )

    if baseline_history is not None:
        baseline_best = baseline_history["best_fitness"]

        if "evaluations" in baseline_history:
            baseline_evaluations = baseline_history["evaluations"]
        else:
            baseline_evaluations = np.arange(
                1, len(baseline_best) + 1
            )

        ax.plot(
            baseline_evaluations, baseline_best,
            label="Random Search",
            linewidth=2,
            alpha=0.85
        )

    ax.set_title("Optimization Performance")
    ax.set_xlabel("Fitness Evaluations")
    ax.set_ylabel("Best Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _prepare_save_path(save_path)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_fitness_history(history, show=True, save_path=None):
    """Plot best and average fitness across GA generations."""

    fig, ax = plt.subplots(figsize=(10, 6))

    generations = np.arange(1, len(history["best_fitness"]) + 1)

    ax.plot(
        generations, history["best_fitness"],
        label="Best Fitness", linewidth=2
    )
    ax.plot(
        generations, history["average_fitness"],
        label="Average Fitness", linewidth=2
    )

    ax.set_title("GA Fitness Across Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _prepare_save_path(save_path)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
