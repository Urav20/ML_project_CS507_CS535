from sklearn.ensemble import RandomForestClassifier
from src import config
from src.preprocessing import build_tree_preprocessor
from src.experiment import run_experiment

GRID = {
    "model__n_estimators": [100, 200],          # more trees = more stable, slower
    "model__max_depth": [None, 25],             # limits tree complexity
    "model__min_samples_leaf": [1, 3, 5],       # bigger leaf = less over-fitting
    "model__min_samples_split": [2, 10],
    "model__class_weight": [None, "balanced"],  # up-weight rare Grade 1
}
if __name__ == "__main__":
    run_experiment("Random Forest", build_tree_preprocessor(),
                   RandomForestClassifier(n_jobs=-1, random_state=config.SEED), GRID)
