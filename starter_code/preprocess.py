import numpy as np
from starter_code.load_data import load_split


def preprocess_data():
    # Load training data and get the input speeds 
    # and validity mask
    train = load_split('train')
    X_train = train['X']
    valid_mask = train['valid_mask']
    # Move the sensor axis before time for matching
    # with valid_mask
    X_transpose = X_train.transpose(0, 2, 1)
    # Get a valid speeds from the training set
    Valid_speed = X_transpose[valid_mask]

    # Calculate the mean and the standard deviation
    # of valid training speed 
    mean = Valid_speed.mean()
    std = Valid_speed.std()

    # Standardize speeds in the training set by using the 
    # training mean and standard deviation
    X_train_standardized = (X_train - mean) / std

    # Find missing speeds in the training set and set them
    # to zero
    missing_mask = np.isnan(X_train_standardized)
    X_train_standardized[missing_mask] = 0

    # Load validiation data and get the input speed
    validation = load_split('validation')
    X_val = validation['X']

    # Standardize speeds in the validation by using the 
    # training mean and standard deviation
    X_val_standardize = (X_val - mean) / std

    # Find missing speeds in the validation and set them
    # to zero
    missing_Val_mask = np.isnan(X_val_standardize)
    X_val_standardize[missing_Val_mask] = 0

    # Load test_features data and get the input speed
    test = load_split('test_features')
    X_test = test['X']

    # Standardize speeds in the test_features by using the 
    # training mean and standard deviation
    X_Test_standardize = (X_test - mean) / std

    # Find missing speeds in the test_features and set them
    # to zero
    missing_test_mask = np.isnan(X_Test_standardize)
    X_Test_standardize[missing_test_mask] = 0

    # Update 'X' in each dictionary with preprocessed speeds
    train['X'] = X_train_standardized
    validation['X'] = X_val_standardize
    test['X'] = X_Test_standardize

    return train, validation, test