from sklearn.neural_network import MLPClassifier
from src import config
from src.preprocessing import build_distance_preprocessor
from src.experiment import run_experiment

GRID = {
    "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
    "model__alpha": [1e-4, 1e-3, 1e-2],         # L2 regularisation strength
    "model__learning_rate_init": [1e-3, 3e-3],
}
if __name__ == "__main__":
    mlp = MLPClassifier(activation="relu", solver="adam", batch_size=256,
                        max_iter=200, early_stopping=True, n_iter_no_change=10,
                        random_state=config.SEED)
    run_experiment("Neural Network", build_distance_preprocessor(), mlp, GRID)
