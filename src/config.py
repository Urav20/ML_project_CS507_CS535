"""Central settings: paths, seed, split size, column groups."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "dataset"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

SEED = 42            # one seed everywhere -> reproducible
VAL_SIZE = 0.20      # 80% train / 20% validation
CV_FOLDS = 3         # CV folds used only for hyperparameter tuning
TUNE_SAMPLE = int(os.environ.get("TUNE_SAMPLE", 40000))  # rows used while tuning (speed)
N_ITER = int(os.environ.get("N_ITER", 8))                # random-search candidates

ID_COL = "building_id"
TARGET_COL = "damage_grade"
CLASSES = [1, 2, 3]

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
NUM_COLS = ["count_floors_pre_eq", "age", "area_percentage",
            "height_percentage", "count_families"]
CAT_COLS = ["land_surface_condition", "foundation_type", "roof_type",
            "ground_floor_type", "other_floor_type", "position",
            "plan_configuration", "legal_ownership_status"]
BIN_COLS = [
    "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber", "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
    "has_superstructure_other",
    "has_secondary_use", "has_secondary_use_agriculture", "has_secondary_use_hotel",
    "has_secondary_use_rental", "has_secondary_use_institution",
    "has_secondary_use_school", "has_secondary_use_industry",
    "has_secondary_use_health_post", "has_secondary_use_gov_office",
    "has_secondary_use_use_police", "has_secondary_use_other",
]
FEATURE_COLS = GEO_COLS + NUM_COLS + CAT_COLS + BIN_COLS  # 3+5+8+22 = 38
