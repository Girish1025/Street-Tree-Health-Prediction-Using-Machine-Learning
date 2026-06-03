"""Project configuration values."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Tree_census.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

RANDOM_STATE = 42
TARGET = "health"
TEST_SIZE = 0.30

SELECTED_COLUMNS = [
    "tree_id", "tree_dbh", "stump_diam", "curb_loc", "status", "health",
    "spc_latin", "steward", "sidewalk", "problems", "root_stone",
    "root_grate", "root_other", "trunk_wire", "trnk_light", "trnk_other",
    "brch_light", "brch_shoe", "brch_other", "borough"
]

NUMERIC_COLUMNS = ["tree_dbh", "stump_diam"]
