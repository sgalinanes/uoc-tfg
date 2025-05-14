#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm
from pathlib import Path
import tsfel
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, log_loss, confusion_matrix, classification_report)
from sklearn.base import clone
import numpy as np
from tabulate import tabulate
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
import os
from pathlib import Path
import logging, json, datetime

N_THREADS = int(os.getenv("N_THREADS", os.cpu_count()))   # 32
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)            # XGBoost/LightGBM

# Añade esto justo después de los imports:
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)
    
DATA_ROOT = Path(os.getenv('DATA_DIR', './data'))
BASE_DIR  = DATA_ROOT / 'preprocessed'

results_dir = Path('results'); results_dir.mkdir(exist_ok=True)

logging.basicConfig(filename=results_dir/'run.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

logging.info(f"*** DATA_DIR = {os.getenv('DATA_DIR')} ***")

# ## Utilidades

# In[2]:


def check_robustness(metric_name, metric_values, threshold_percent=10):
    """
    Check the robustness of a model based on k-fold metric values.

    A model is considered robust if the coefficient of variation (CV),
    defined as (standard deviation / mean) * 100, is less than threshold_percent.

    Parameters:
        metric_values (list or np.array): List/array of metric values from each fold.
        threshold_percent (float): The maximum allowed CV percentage (default: 10).

    Returns:
        bool: True if the model is robust (CV < threshold_percent), False otherwise.
    """
    metric_values = np.array(metric_values)
    mean_val = np.mean(metric_values)
    std_val = np.std(metric_values)

    # Avoid division by zero:
    if mean_val == 0:
        cv = float('inf')
    else:
        cv = (std_val / mean_val) * 100

    logging.info(f"[{metric_name}] Metric Mean: {mean_val:.4f}")
    logging.info(f"[{metric_name}] Metric Standard Deviation: {std_val:.4f}")
    logging.info(f"[{metric_name}] Coefficient of Variation (CV): {cv:.2f}%")

    if cv < threshold_percent:
        logging.info(f"[{metric_name}] Model is robust (CV < {threshold_percent}%)")
        return True
    else:
        logging.info(f"[{metric_name}] Model is not robust (CV >= {threshold_percent}%)")
        return False

def get_movement_data(subject_id, path, sampling_rate):
    # File path handling
    if not path.exists():
        raise FileNotFoundError(f"Movement data not found for subject {subject_id}")

    # Load raw data
    raw_data = np.fromfile(path, dtype=np.float32)

    # Channel configuration
    tasks = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", 
             "StretchHold", "HoldWeight", "DrinkGlas", "CrossArms", 
             "TouchNose", "Entrainment1", "Entrainment2"]
    wrists = ["Left", "Right"]
    sensors = ["Accelerometer", "Gyroscope"]
    axes = ["X", "Y", "Z"]

    # Calculate expected parameters
    n_channels = len(tasks) * len(wrists) * len(sensors) * len(axes)

    # Even though the duration was 10.24 per assessment, the first ~0.5 seconds are not used and thus
    # the preprocessed data has only 9.76 seconds of data
    expected_duration = 9.76  # seconds per assessment
    expected_timepoints = int(expected_duration * sampling_rate)  # 976

    # Validate data size
    expected_size = n_channels * expected_timepoints
    #logging.info(f"Expected size: {expected_size} elements")
    if len(raw_data) != expected_size:
        raise ValueError(f"Unexpected data size for {subject_id}: "
                         f"Got {len(raw_data)} elements, expected {expected_size}")

    return {
        'tasks': tasks,
        'wrists': wrists,
        'sensors': sensors,
        'axes': axes,
        'raw_data': raw_data,
        'timepoints': expected_timepoints,
    }

def load_movement_features(subject_id, path, sampling_rate=100):
    """
    Load and structure movement data from binary files with proper validation

    Args:
        subject_id (str): Subject identifier (e.g., '001')
        base_path (str): Base directory for movement data
        sampling_rate (int): Sampling rate in Hz (used for validation)

    Returns:
        dict: Structured data with keys as channel names and values as time series arrays
    """
    movement_data = get_movement_data(subject_id, path, sampling_rate)
    tasks = movement_data['tasks']
    wrists = movement_data['wrists']
    sensors = movement_data['sensors']
    axes = movement_data['axes']
    raw_data = movement_data['raw_data']
    timepoints = movement_data['timepoints']

    # Reshape and structure the data
    structured_data = {}
    channel_idx = 0

    for task in tasks:
        for wrist in wrists:
            for sensor in sensors:
                for axis in axes:
                    # Extract channel data
                    start = channel_idx * timepoints
                    end = (channel_idx + 1) * timepoints

                    # Create channel name
                    channel_name = f"{task}_{wrist}_{sensor.split(' ')[0]}_{axis}"

                    structured_data[channel_name] = raw_data[start:end]

                    channel_idx += 1

    return structured_data

def load_movement_features_continuous(subject_id, path, sampling_rate=100):
    """
    Load and structure movement data by concatenating across tasks for each (wrist, sensor, axis).

    Instead of one series per task, produces one long series per channel (wrist/sensor/axis)
    by appending each task’s trimmed data end-to-end.

    Args:
        subject_id (str): Subject identifier (e.g., '001')
        base_path (str): Base directory for movement data
        sampling_rate (int): Sampling rate in Hz (used for validation)

    Returns:
        dict: Keys are "{wrist}_{sensor}_{axis}", values are 1D numpy arrays of length
              (expected_timepoints - skip) * n_tasks
    """
    movement_data = get_movement_data(subject_id, path, sampling_rate)
    tasks = movement_data['tasks']
    wrists = movement_data['wrists']
    sensors = movement_data['sensors']
    axes = movement_data['axes']
    raw_data = movement_data['raw_data']
    timepoints = movement_data['timepoints']

    # Prepare container: one list per (wrist, sensor, axis)
    cont = {
        f"{wrist}_{sensor}_{axis}": []
        for wrist in wrists
        for sensor in sensors
        for axis in axes
    }

    # Slice raw per (task, wrist, sensor, axis), trim, and append
    idx = 0
    for task in tasks:
        for wrist in wrists:
            for sensor in sensors:
                for axis in axes:
                    start = idx * timepoints
                    end = start + timepoints
                    segment = raw_data[start:end]
                    key = f"{wrist}_{sensor}_{axis}"
                    cont[key].append(segment)
                    idx += 1

    # Concatenate per key
    for key, chunks in cont.items():
        cont[key] = np.concatenate(chunks)

    return cont

def extract_tsfel_ts_features(channel_data, domain=None, fs=100):
    """
    Extract TSFEL features from a single-channel time series.

    Args:
        channel_data (np.ndarray): 1D array of time series values.
        domain: (str): Domain of features to extract (default: 'all').
            - 'statistical', 'temporal', 'spectral', 'fractal': Includes the corresponding feature domain.
            - 'all': Includes all available feature domains.
            - list of str: A combination of the above strings, e.g., ['statistical', 'temporal'].
            - None: By default, includes the 'statistical', 'temporal', and 'spectral' domains.
        fs (int): Sampling frequency (default: 100).

    Returns:
        np.ndarray: Array of TSFEL features.
    """
    # Obtain a default configuration covering all domains.
    cfg = tsfel.get_features_by_domain(domain)

    # Extract features; the result is a DataFrame with one row.
    features_df = tsfel.time_series_features_extractor(cfg, channel_data, fs=fs, verbose=0)

    # Flatten to 1D numpy array and return.
    return features_df.values.flatten()

def extract_ts_features(label, channel_data, domain=None, fs=None):
    if label == 'basic':
        features = [
            np.mean(channel_data), # Mean of the signal
            np.std(channel_data), # Standard deviation
            np.min(channel_data), # Minimum value
            np.max(channel_data), # Maximum value
            np.percentile(channel_data, 25), # 25th percentile
            np.percentile(channel_data, 75), # 75th percentile
            np.var(channel_data), # Variance
            len(np.where(np.diff(np.sign(channel_data)))[0]) / len(channel_data),  # Zero-crossing rate
            np.sqrt(np.mean(channel_data**2))  # Root Mean Square (RMS)
        ]

        # Return features as a numpy array
        features = np.array(features)
        return features
    else:
        return extract_tsfel_ts_features(channel_data, domain=domain, fs=fs)


# ## Carga de datos
# 
# Requisite: Python 3.12 (por catboost). brew install libomp para xgboost
# 
# tsfel explicaciones: https://github.com/fraunhoferportugal/tsfel/blob/v0.1.9/tsfel/feature_extraction/features.json
# 
# En base a los datos preprocesados, entrenaremos modelos de clasificacion bajo distintos escenarios:
# 

# In[5]:


# Cargar metadatos
file_list = pd.read_csv(BASE_DIR / 'file_list.csv', dtype={'id': str})

movement_data_per_subject = {}
sampling_rate = 100

# In the normal case, we have 132 channels and every channel has 976 time points (giving 128832 which is correct)

# Create multiple cases so we can check each model independently:
# 1. Basic statistical features (Reduce 128832 features [976 features p/c] to 1188 features [9 features p/c]) + questionnaire data
# 2. TSFEL only temporal features (Reduce 128832 features [976 features p/c] to 1848 features [14 features p/c]  + questionnaire data
# 3. [NOT DOING - TOO BIG] TSFEL only statistical features (Reduce 128832 features [976 features p/c] to 4092 features [31 features p/c]  + questionnaire data
# 4. [NOT DOING - TOO BIG] TSFEL only spectral features (Reduce 128832 features [976 features p/c] to 14652 features [111 features p/c]  + questionnaire data
# 5. Only questionnaire data (Uses only the questionnaire data, no movement data), used as a baseline

# In the continuous case, we have 12 channels but every channel has 10736 time points (giving 128832 which is correct)

# 1. [NOT DOING - DONE ON NORMAL] Basic statistical features (Reduce 128832 features [10736 features p/c] to 108 features [9 features p/c]) + questionnaire data
# 2. [NOT DOING - DONE ON NORMAL] TSFEL only temporal features (Reduce 128832 features [10736 features p/c] to 168 features [14 features p/c]  + questionnaire data
# 3. TSFEL only statistical features (Reduce 128832 features [10736 features p/c] to 372 features [31 features p/c]  + questionnaire data
# 4. TSFEL only spectral features (Reduce 128832 features [10736 features p/c] to 1332 features [111 features p/c]  + questionnaire data
# 5. Only questionnaire data (Uses only the questionnaire data, no movement data), used as a baseline

# For efficiency purposes, we do
# 1. (Normal) Basic statistical features (Reduce 128832 features [976 features p/c] to 1188 features [9 features p/c]) + questionnaire data
# 2. (Normal) TSFEL only temporal features (Reduce 128832 features [976 features p/c] to 1848 features [14 features p/c]  + questionnaire data
# 3. (Cont)   TSFEL only statistical features (Reduce 128832 features [10736 features p/c] to 372 features [31 features p/c]  + questionnaire data
# 4. (Cont)   TSFEL only spectral features (Reduce 128832 features [10736 features p/c] to 1332 features [111 features p/c]  + questionnaire data
# 5. (N/A)    Only questionnaire data (Uses only the questionnaire data, no movement data), used as a baseline

pipeline_labels = ['TSFEL-spectral', 'questionnaire-only'] #['basic', 'TSFEL-temporal', 

pipeline_args = [
    #{'domain': None, 'fs': None, 'ts_mode': None},  # Basic statistical features
    #{'domain': 'temporal', 'fs': sampling_rate, 'ts_mode': None},  # TSFEL-temporal
    #{'domain': 'statistical', 'fs': sampling_rate, 'ts_mode': 'continuous'},  # TSFEL-statistical
    {'domain': 'spectral', 'fs': sampling_rate, 'ts_mode': 'continuous'},  # TSFEL-spectral
    None,  # Questionnaire-only
]

# X must be 2D (samples, features)
X_vals = {label: [] for label in pipeline_labels}
y_vals = {label: [] for label in pipeline_labels}

# First, load timeseries data and extract features
for _, row in tqdm(file_list.iterrows(), total=len(file_list)):
    # Cargar cuestionario
    qpath = BASE_DIR / 'questionnaire' / f'{row["id"]}_ml.bin'
    quest_data = np.fromfile(qpath, dtype=np.float32)

    mpath = BASE_DIR / 'movement' / f'{row["id"]}_ml.bin'
    # Load structured movement data
    movement_data = load_movement_features(row["id"], mpath)

    # Load continuous movement data
    continuous_movement_data = load_movement_features_continuous(row["id"], mpath)

    """
    movement_data:
    {
        'Relaxed1_Left_Accelerometer_X': np.array([...]),
        'Relaxed1_Left_Accelerometer_Y': np.array([...]),
        ...
    }
    Each key corresponds to a channel, and the value is a 1D numpy array of time series data.
    We have a total of 132 keys and each time series has 976 time points.
    """


    for label, args in zip(pipeline_labels, pipeline_args):
        # Extract features for each channel unless label is 'questionnaire-only'
        features = quest_data
        ts_features = {}
        if label != 'questionnaire-only':
            pipeline_movement_data = movement_data if args['ts_mode'] is None else continuous_movement_data
            for channel_name, channel_data in pipeline_movement_data.items():
                # Extract features using the specified domain and sampling rate
                ts_features[channel_name] = extract_ts_features(label, channel_data, domain=args['domain'], fs=args['fs'])
                #logging.info(f"Extracted {ts_features[channel_name].shape} features from {channel_name} for {label}")

            # Concatenate features from all channels
            concat_ts_features = np.concatenate(list(ts_features.values()), axis=0)

            # Combine questionnaire data with time series features
            features = np.concatenate((features, concat_ts_features), axis=0)

        X_vals[label].append(features.reshape(1, -1))
        y_vals[label].append(row['label'])


# Combine all samples for each pipeline
for label in pipeline_labels:
    if X_vals[label]:
        X_vals[label] = np.concatenate(X_vals[label], axis=0)
        y_vals[label] = np.array(y_vals[label])
    else:
        X_vals[label] = np.array([])
        y_vals[label] = np.array([])

for x in X_vals:
    logging.info(f"{x} shape: {X_vals[x].shape}")


# ## Modelos de clasificacion

# In[6]:


# -----------------------------------------------------------------------------
# Labels mapping for reference:
# 0: HC
# 1: PD
# 2: DD
# -----------------------------------------------------------------------------

def tune_models(clf, param_grids, X_inner, y_inner, model_name, fit_params=None):
    """
    Tune hyperparameters for each model using GridSearchCV.

    Parameters:
        clf (sklearn classifier): Classifier instance to tune.
        param_grids (dict): Dictionary mapping model names to their hyperparameter grids.
        X_inner (np.array or pd.DataFrame): Feature matrix for inner CV.
        y_inner (np.array): Labels for inner CV.
        model_name (str): Name of the model being tuned.
        fit_params (dict): Additional parameters for fitting the model.

    Returns:
        tuned_results (dict): Dictionary mapping model names to the fitted GridSearchCV object.
    """
    # Use GridSearchCV for hyperparameter tuning.
    scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grids[model_name],
        scoring=scoring,
        cv=3,       # inner CV folds
        n_jobs=N_THREADS,
        refit="accuracy",  # refit on accuracy
        #refit=refit_strategy  # custom refit to pick robust candidate
    )
    grid.fit(X_inner, y_inner, **fit_params)
    return (grid.best_estimator_, grid.best_params_)

# TODO: Analyze the refit_strategy function further.
#def refit_strategy(cv_results):
#    """
#    Custom refit strategy that uses both accuracy and weighted F1 metrics.
#    
#    For each hyperparameter candidate (as provided in GridSearchCV’s cv_results_),
#    we compute the mean and standard deviation for both 'accuracy' and 'f1_weighted'.
#    We then calculate the coefficient of variation (CV) for each metric in percentage.
#    The composite score is calculated as:
#    
#        composite = 0.5*(mean_accuracy + mean_f1_weighted) - lambda_val * 0.5*(CV_accuracy + CV_f1_weighted)
#    
#    A candidate with a high mean score and a low CV will be preferred.
#    
#    Parameters:
#        cv_results (dict): The cv_results_ dictionary from GridSearchCV. It must contain the keys:
#            "mean_test_accuracy", "std_test_accuracy", 
#            "mean_test_f1_weighted", "std_test_f1_weighted".
#    
#    Returns:
#        best_index (int): The index (as in cv_results) of the best candidate.
#    """
#    df = pd.DataFrame(cv_results)
#    required_keys = ["mean_test_accuracy", "std_test_accuracy", "mean_test_f1_weighted", "std_test_f1_weighted"]
#    if not all(key in df.columns for key in required_keys):
#        raise ValueError(f"cv_results must contain the keys: {required_keys}")
#    
#    # Means and standard deviations for accuracy and weighted F1.
#    mean_acc = df["mean_test_accuracy"]
#    std_acc  = df["std_test_accuracy"]
#    mean_f1  = df["mean_test_f1_weighted"]
#    std_f1   = df["std_test_f1_weighted"]
#    
#    # Compute coefficient of variation (CV) in percentages.
#    cv_acc = (std_acc / mean_acc) * 100
#    cv_f1  = (std_f1 / mean_f1) * 100
#    
#    # Lambda controls the weight given to robustness.
#    lambda_val = 0.01
#    
#    # Composite score: we want high mean and low CV.
#    composite = 0.5 * (mean_acc + mean_f1) - lambda_val * 0.5 * (cv_acc + cv_f1)
#
#    best_index = composite.idxmax()
#    return best_index

def run_cv(X, y, models, n_splits=5, mode="default", class_weights=None, tune_inner=False, param_grids=None):
    """
    Unified cross-validation runner with optional inner-loop hyperparameter tuning.

    Parameters:
        X (np.array or pd.DataFrame): Feature matrix.
        y (np.array): Labels.
        models (dict): Dictionary mapping model names to a tuple: (classifier instance, fit_params dict).
                       (If no additional fit parameters are needed, use an empty dict.)
        n_splits (int): Number of outer CV folds.
        mode (str): One of:
            - "default": standard CV.
            - "smote": apply SMOTE oversampling on the training data.
            - "weighted": for multi-class cost-sensitive learning.
            - "weighted_binary": for binary cost-sensitive learning (with special handling for XGBoost).
        class_weights (dict): Custom class weighting dictionary (e.g., {0: 1.0, 1: 2.0}).
        tune_inner (bool): If True and param_grids is provided, perform inner-loop GridSearchCV tuning.
        param_grids (dict): Dictionary mapping model names to their hyperparameter grids for tuning.
                           Only used if tune_inner is True.

    Returns:
        dict: For each model, a dictionary with keys:
            "accuracy": list of accuracies per outer fold,
            "f1_score": list of weighted F1 scores per outer fold,
            "log_loss": list of log losses per outer fold,
            "y_true": list of true label arrays per fold,
            "y_pred": list of predicted label arrays per fold,
            "y_pred_proba": list of predicted probability arrays per fold.
    """
    # Suppress warnings regarding use_label_encoder and feature names
    # Ensure X is a DataFrame with valid feature names.
    if not hasattr(X, "columns"):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Store results as an array of "folds" for each model.
    results = { name: [] for name in models.keys() }

    # Initialize SMOTE if selected.
    if mode == "smote":
        oversampler = SMOTE(random_state=42)

    # --- K-fold Outer-loop: Cross-validation ---
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        # Optionally apply SMOTE.
        if mode == "smote":
            X_train, y_train = oversampler.fit_resample(X_train, y_train)

        # For each model:
        for name, (model, fit_params) in models.items():
            clf = clone(model)
            fold_fit_params = fit_params.copy() if fit_params is not None else {}

            # For weighted modes, set class_weight if supported.
            if mode in ("weighted", "weighted_binary") and class_weights is not None:
                if 'class_weight' in clf.get_params():
                    clf.set_params(class_weight=class_weights)

            # Special handling for XGBoost in weighted_binary mode.
            if mode == "weighted_binary":
                try:
                    if name == "XGBoost" and isinstance(clf, XGBClassifier) and len(np.unique(y_train)) == 2:
                        ratio = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
                        clf.set_params(objective='binary:logistic', scale_pos_weight=ratio)
                except Exception as e:
                    raise ValueError(f"Error setting scale_pos_weight for XGBoost: {e}")

            # For LightGBM: set eval_set and eval_metric if not already set and verbose to -1.
            # This is to avoid printing too much information during training.
            if name == "LightGBM":
                if "eval_set" not in fold_fit_params:
                    fold_fit_params["eval_set"] = [(X_test, y_test)]
                if "eval_metric" not in fold_fit_params:
                    fold_fit_params["eval_metric"] = "logloss"
                clf.set_params(verbose=-1)

            # --- K-fold Inner-loop: Hyperparameter tuning ---
            hyperparameter_tuning_best_params = None
            if tune_inner and param_grids is not None and name in param_grids:
                # Perform hyperparameter tuning using GridSearchCV.
                logging.info(f"Tuning hyperparameters for {name}...")
                (best_estimator, best_params) = tune_models(clf, param_grids, X_train, y_train, name, fit_params=fold_fit_params)
                clf = best_estimator
                hyperparameter_tuning_best_params = best_params
            else:
                logging.info(f"Not tuning hyperparameters for {name}.")
                clf.fit(X_train, y_train, **fold_fit_params)

            # Evaluate on the outer test set.
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)

            # Store each model's results in the "results" array, where each outer fold is indexed by k_idx.
            # And each element in the results array is a metrics dictionary.
            model_metrics = {}

            model_metrics["params"] = hyperparameter_tuning_best_params if hyperparameter_tuning_best_params else "default",
            model_metrics["accuracy"] = accuracy_score(y_test, y_pred)
            model_metrics["f1_score"] = f1_score(y_test, y_pred, average='weighted')
            model_metrics["log_loss"] = log_loss(y_test, y_pred_proba)
            model_metrics["y_true"] = y_test
            model_metrics["y_pred"] = y_pred
            model_metrics["y_pred_proba"] = y_pred_proba

            logging.info(f"Model results for {name}:")
            logging.info(f"Parameters: {model_metrics['params']}")
            logging.info(f"Accuracy: {model_metrics['accuracy']:.4f}")
            logging.info(f"F1 Score: {model_metrics['f1_score']:.4f}")
            logging.info(f"Log Loss: {model_metrics['log_loss']:.4f}")

            results[name].append(model_metrics)

    # Check robustness for each model (using your check_robustness function)
    for name in results.keys():
        # The metrics stored for each model are in an array of dictionaries, where each dictionary corresponds to a fold.
        # We need to extract the metrics from each fold and check their robustness.
        acc_values = [results[name][k]["accuracy"] for k in range(n_splits)]
        f1_values = [results[name][k]["f1_score"] for k in range(n_splits)]
        ll_values = [results[name][k]["log_loss"] for k in range(n_splits)]

        logging.info(f"\nRobustness for model: {name}")
        acc_robust = check_robustness("accuracy", acc_values)
        f1_robust = check_robustness("f1_score", f1_values)
        ll_robust = check_robustness("log_loss", ll_values)
        if acc_robust and f1_robust and ll_robust:
            logging.info(f"{name} is robust across folds.\n")
        else:
            logging.info(f"[ERROR] {name} is not robust across folds.\n")

    return results

def evaluate_cv(results, target_names):
    """
    Aggregates per-fold metrics from a results dictionary and prints overall metrics,
    confusion matrix, and classification report using tabulate.

    Parameters:
        results (dict): Dictionary mapping model names to a list of per-fold metric dictionaries.
                        Each per-fold dictionary must include:
                            "accuracy": float,
                            "f1_score": float,
                            "log_loss": float,
                            "y_true": true labels for the fold,
                            "y_pred": predicted labels for the fold,
                            "y_pred_proba": predicted probability array for the fold,
                            "params": the model parameters used.
        target_names (list): List of class names (e.g., ['HC', 'PD', 'DD']).

    Returns:
        dict: Mapping of model names to overall metrics including:
            "accuracy_mean", "accuracy_std", "f1_mean", "f1_std", 
            "log_loss_mean", "log_loss_std", "confusion_matrix", and "classification_report".
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    from tabulate import tabulate

    overall_metrics = {}
    table_data = []

    # From target_names, print what we are evaluating
    logging.info(f"Evaluating models with target names: {target_names}")
    logging.info(f"Number of classes: {len(target_names)}")
    logging.info(f"Classes: {target_names}")
    logging.info(f"Number of models: {len(results)}")
    logging.info(f"Models: {list(results.keys())}")

    for name, fold_results in results.items():
        # Aggregate per-fold predictions.
        y_true_all = np.concatenate([fold["y_true"] for fold in fold_results])
        y_pred_all = np.concatenate([fold["y_pred"] for fold in fold_results])
        y_pred_proba_all = np.concatenate([fold["y_pred_proba"] for fold in fold_results])

        # Aggregate per-fold metric values.
        acc_values = np.array([fold["accuracy"] for fold in fold_results])
        f1_values  = np.array([fold["f1_score"] for fold in fold_results])
        ll_values  = np.array([fold["log_loss"] for fold in fold_results])

        # Retrieve parameters (assumed constant across folds).
        params_val = fold_results[0].get("params", "default")
        if isinstance(params_val, tuple) and len(params_val) == 1:
            params_val = params_val[0]

        # Compute mean and standard deviation for numeric metrics.
        acc_mean, acc_std = np.mean(acc_values), np.std(acc_values)
        f1_mean,  f1_std  = np.mean(f1_values),  np.std(f1_values)
        ll_mean,  ll_std  = np.mean(ll_values),  np.std(ll_values)

        # Compute the overall confusion matrix and classification report.
        cm = confusion_matrix(y_true_all, y_pred_all)
        clf_report = classification_report(y_true_all, y_pred_all, target_names=target_names, zero_division=0)

        overall_metrics[name] = {
            "params": params_val,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "log_loss_mean": ll_mean,
            "log_loss_std": ll_std,
            "confusion_matrix": cm,
            "classification_report": clf_report,
        }
        table_data.append([
            name,
            params_val,
            f"{acc_mean:.4f} ± {acc_std:.4f}",
            f"{f1_mean:.4f} ± {f1_std:.4f}",
            f"{ll_mean:.4f} ± {ll_std:.4f}"
        ])

        # Print detailed report for this model.
        logging.info(f"Model: {name}")
        logging.info(f"Parameters: {params_val}")
        logging.info("Confusion Matrix:")
        logging.info(cm)
        logging.info("\nClassification Report:")
        logging.info(clf_report)
        logging.info("\n")

    logging.info("Overall Metrics:")
    logging.info(tabulate(table_data, headers=["Model", "Params", "Accuracy", "Weighted F1", "Log Loss"], tablefmt="pipe"))

    return overall_metrics

param_grids = {
    "RandomForest": {
        # default: { "n_estimators": 100, "max_depth": None, "max_features": "sqrt" }
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"]
    },
    "XGBoost": {
        # default: { "learning_rate": 0.3, "max_depth": 6, "subsample": 1.0 }
        "learning_rate": [0.1, 0.3, 0.05],
        "max_depth": [3, 6],
        "subsample": [0.7, 1.0]
    },
    "CatBoost": {
        # default: { "depth": 6, "l2_leaf_reg": 3 }
        "depth": [6],
        "l2_leaf_reg": [3]
    },
    "LightGBM": {
        # default: { "num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 20 }
        "num_leaves": [20, 31],
        "learning_rate": [0.05, 0.1],
        "min_child_samples": [10, 20]
    }
}

models = {
     "RandomForest": [RandomForestClassifier(n_jobs=1, random_state=42), {}],
     "XGBoost": [XGBClassifier(n_jobs=1, tree_method="hist", predictor="auto", eval_metric='mlogloss', random_state=42), {}],
     "CatBoost": [CatBoostClassifier(thread_count=1, verbose=0, random_state=42, allow_writing_files=False), {}],
     "LightGBM": [LGBMClassifier(n_jobs=1, random_state=42), {'callbacks': [lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)]}],
#     "LightGBM": [LGBMClassifier(random_state=42), {}]

}


# ## Modelos multiclase (PD vs DD vs HC)

# In[7]:


for x in X_vals:
    X = X_vals[x]
    y = y_vals[x]
    logging.info(f"Running models for {x} pipeline...")
    logging.info(f"(X shape: {X.shape}, y shape: {y.shape})")
    logging.info(y[:10])

    # -----------------------------------------------------------------------------
    # Multi-Class Classification (no changes): PD vs DD vs HC
    # -----------------------------------------------------------------------------
    logging.info(len(np.unique(y)))
    if len(np.unique(y)) == 3:
        logging.info("Performing multi-class classification (PD vs DD vs HC) [Default Mode]...")
        results_default = run_cv(X, y, models, n_splits=5, mode="default", tune_inner=True, param_grids=param_grids)
        overall_default = evaluate_cv(results_default, target_names=['HC', 'PD', 'DD'])
    else:
        logging.info("Multi-class classification (PD vs DD vs HC) is not possible with the current labels.")


    # -----------------------------------------------------------------------------
    # Multi-Class Classification with SMOTE: PD vs DD vs HC
    # -----------------------------------------------------------------------------
    if len(np.unique(y)) == 3:
        logging.info("Performing multi-class classification (PD vs DD vs HC) with SMOTE...")
        results_smote = run_cv(X, y, models, n_splits=5, mode="smote", tune_inner=True, param_grids=param_grids)
        overall_smote = evaluate_cv(results_smote, target_names=['HC', 'PD', 'DD'])
    else:
        logging.info("Multi-class classification with SMOTE is not possible with the current labels.")

    # -----------------------------------------------------------------------------
    # Multi-Class Classification with Cost-Sensitive Learning (PD vs DD vs HC)
    # -----------------------------------------------------------------------------
    if len(np.unique(y)) == 3:
        logging.info("Performing multi-class classification (PD vs DD vs HC) with cost-sensitive learning...")
        classes = np.unique(y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weights_multi = dict(zip(classes, weights))
        logging.info("Computed class weights:", class_weights_multi)

        results_weighted = run_cv(
            X, y,
            models,
            n_splits=5,
            mode="weighted",
            class_weights=class_weights_multi,
            tune_inner=True,
            param_grids=param_grids
        )
        overall_weighted = evaluate_cv(results_weighted, target_names=['HC', 'PD', 'DD'])
    else:
        logging.info("Multi-class classification with cost-sensitive learning is not possible with the current labels.")

# guarda métricas y cierra
#with open(results_dir/'metrics.json', 'w') as f:
#    json.dump(overall_default, f, indent=2, cls=NumpyEncoder)

# Also store forr overall_smote and overall_weighted
#with open(results_dir/'metrics_smote.json', 'w') as f:
#    json.dump(overall_smote, f, indent=2, cls=NumpyEncoder)

#with open(results_dir/'metrics_weighted.json', 'w') as f:
#    json.dump(overall_weighted, f, indent=2, cls=NumpyEncoder)