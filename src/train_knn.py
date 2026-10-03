from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing import build_distance_preprocessor
from src.experiment import run_experiment

GRID = {
    "model__n_neighbors": [5, 15, 31],        # small k = flexible/noisy, large k = smooth
    "model__weights": ["uniform", "distance"],  # closer neighbours vote more?
    "model__p": [1, 2],                       # 1 = Manhattan, 2 = Euclidean distance
}
if __name__ == "__main__":
    run_experiment("KNN", build_distance_preprocessor(),
                   KNeighborsClassifier(n_jobs=-1), GRID)
