"""Load the CSVs and make the (identical) stratified train/validation split."""
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def load_train():
    """Return X (38 features, indexed by building_id) and y (damage_grade)."""
    values = pd.read_csv(config.DATA_DIR / "train_values.csv")
    labels = pd.read_csv(config.DATA_DIR / "train_labels.csv")
    df = values.merge(labels, on=config.ID_COL, how="inner", validate="one_to_one")
    df = df.set_index(config.ID_COL)           # building_id kept as index, NOT a feature
    return df[config.FEATURE_COLS], df[config.TARGET_COL]


def load_test():
    """Test features only (labels are not provided). building_id is the index."""
    test = pd.read_csv(config.DATA_DIR / "test_values.csv").set_index(config.ID_COL)
    return test[config.FEATURE_COLS]


def split_train_val(X, y):
    """80/20 stratified split with a fixed seed - same split for all 3 models."""
    return train_test_split(X, y, test_size=config.VAL_SIZE,
                            stratify=y, random_state=config.SEED)
