import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter_code.load_data import load_split
from starter_code.evaluate_validation import evaluate_predictions
from starter_code.create_submission import build_submission

def fit_per_sensor_models(X_train, Y_train, valid_mask):
    num_sensors = X_train.shape[2]
    models = []

    for s in range(num_sensors):
        rows = valid_mask[:, s]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train[rows, :, s],
            Y_train[rows, :, s]
        )

        models.append(model)

    return models

    
