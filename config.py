import config

print(config.GRID_SIZE)
print(config.NUMBER_OF_DRONES)
print(config.POPULATION_SIZE)
print(config.GENERATIONS)
# config.py
# Configuration for UAV Genetic Algorithm

# ==============================
# ENVIRONMENT SETTINGS
# ==============================

GRID_SIZE = 50
NUMBER_OF_DRONES = 4
WAYPOINTS_PER_DRONE = 10

# ==============================
# GENETIC ALGORITHM SETTINGS
# ==============================

POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 3

# ==============================
# FITNESS FUNCTION WEIGHTS
# ==============================

FITNESS_WEIGHTS = {
    "distance": 1.0,
    "collision": 10.0,
    "obstacle": 10.0,
    "coverage": 2.0,
    "smoothness": 1.0
}

# ==============================
# DRONE SETTINGS
# ==============================

MAX_RANGE = 1000.0

# ==============================
# REPRODUCIBILITY
# ==============================

RANDOM_SEED = 42
