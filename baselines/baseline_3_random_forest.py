import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Add the repository root to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter_code.preprocess import preprocess_data
from starter_code.evaluate_validation import evaluate_predictions
from starter_code.create_submission import build_submission

def fit_per_sensor_models(X_train, Y_train, valid_mask):
    # Get the number of sensors and create a list for the models.
    num_sensors = X_train.shape[2]
    models = []

    # Train one Random Forest model for each sensor.
    for s in range(num_sensors):

        # Select valid training samples for this sensor.
        rows = valid_mask[:, s]

        # Create the Random Forest model.
        model = RandomForestRegressor(
            n_estimators=10,   # Number of decision trees.
            max_depth=10,       # Limit how much each tree can grow.
            min_samples_leaf=5, # Require at least 5 samples per leaf.
            random_state=42,    # Keep results consistent.
            n_jobs=1            # Use one CPU worker to limit resource usage
        )

        print(f"Training sensor {s + 1}/{num_sensors}...", flush=True)

        # Train the model using past readings to predict future readings.
        model.fit(
            X_train[rows, :, s],
            Y_train[rows, :, s]
        )

        # Save the trained model.
        models.append(model)

    return models

def predict_per_sensor(models, X):
    # Get the number of samples and sensors.
    num_samples, _, num_sensors = X.shape

    # Get the number of future time steps each model predicts.
    forecast_horizon = models[0].n_outputs_

    # Create an array to store all predictions.
    preds = np.empty(
        (num_samples, forecast_horizon, num_sensors),
        dtype=np.float32
    )

    # Replace missing input values with zero before prediction.
    X_safe = np.nan_to_num(X, nan=0.0)

    # Generate predictions for each sensor.
    for s, model in enumerate(models):
        preds[:, :, s] = model.predict(X_safe[:, :, s])

    return preds

def main():
    print("Baseline 3: Per-Sensor Random Forest\n")

    # Load and preprocess the datasets.
    train, val, test = preprocess_data()

    # Train one Random Forest model per sensor.
    models = fit_per_sensor_models(
        train["X"],
        train["Y"],
        train["valid_mask"]
    )

    print(f"Fitted {len(models)} Random Forest models.")

    # Generate predictions using the validation data.
    val_pred = predict_per_sensor(models, val["X"])

    # Evaluate the model's validation performance.
    print("\nValidation performance:")
    evaluate_predictions(val_pred)

    # Generate predictions for the test data.
    test_pred = predict_per_sensor(models, test["X"])

    # Create the submission CSV file.
    build_submission(
        test_pred,
        out_path="baseline_3_submission.csv"
    )

if __name__ == "__main__":
    main()

