
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_FILE = "banking_queue_dataset.csv"
RANDOM_SEED = 42

FEATURE_CANDIDATES = [
    "queue_capacity",
    "queue_length",
    "system_occupancy",
    "configured_arrival_rate",
    "service_multiplier",
    "arrivals_this_second",
    "served_this_second",
    "dropped_this_second",
    "abandoned_this_second",
    "average_queue_length",
    "average_system_occupancy",
    "maximum_queue_length",
    "maximum_system_occupancy",
    "average_waiting_time",
    "maximum_waiting_time",
    "average_service_time",
    "server_utilization",
    "throughput_per_second",
    "drop_rate",
    "abandonment_rate",
    "arrivals_UPI",
    "served_UPI",
    "dropped_UPI",
    "arrivals_CARD_PAYMENT",
    "served_CARD_PAYMENT",
    "dropped_CARD_PAYMENT",
    "arrivals_ATM",
    "served_ATM",
    "dropped_ATM",
    "arrivals_IMPS",
    "served_IMPS",
    "dropped_IMPS",
    "arrivals_NEFT",
    "served_NEFT",
    "dropped_NEFT",
    "arrivals_NET_BANKING",
    "served_NET_BANKING",
    "dropped_NET_BANKING"
]


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading banking queue dataset...")

try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    raise SystemExit(
        f"ERROR: {DATASET_FILE} not found. "
        "Keep the CSV and Python file in the same folder."
    )

required_columns = ["is_anomaly", "anomaly_type"]

for column in required_columns:
    if column not in df.columns:
        raise SystemExit(f"ERROR: Missing column: {column}")

df["is_anomaly"] = pd.to_numeric(
    df["is_anomaly"], errors="coerce"
)

df["anomaly_type"] = (
    df["anomaly_type"].astype(str).str.strip().str.upper()
)

if df["is_anomaly"].isna().any():
    raise SystemExit("ERROR: Invalid values found in is_anomaly.")

if not df["is_anomaly"].isin([0, 1]).all():
    raise SystemExit("ERROR: is_anomaly must contain only 0 and 1.")

df["is_anomaly"] = df["is_anomaly"].astype(int)

if not (
    (df["is_anomaly"] == 0) == (df["anomaly_type"] == "NONE")
).all():
    raise SystemExit(
        "ERROR: is_anomaly and anomaly_type labels do not match."
    )

if df.empty:
    raise SystemExit("ERROR: Dataset is empty.")

print("\nDATASET INFORMATION")
print("-" * 45)
print("Total records:", len(df))

print("\nANOMALY DISTRIBUTION:")
print(df["is_anomaly"].value_counts().sort_index())

print("\nANOMALY TYPES:")
print(df["anomaly_type"].value_counts())


# ==========================================
# 3. SELECT NUMERICAL FEATURES
# ==========================================

feature_columns = [
    column for column in FEATURE_CANDIDATES
    if column in df.columns
]

if len(feature_columns) < 4:
    raise SystemExit(
        "ERROR: Fewer than 4 candidate features were found."
    )

X = df[feature_columns].copy()

for column in feature_columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns that are completely empty or constant.

usable_features = [
    column for column in feature_columns
    if X[column].notna().any()
    and X[column].nunique(dropna=True) > 1
]

X = X[usable_features]
feature_columns = usable_features

if len(feature_columns) < 4:
    raise SystemExit(
        "ERROR: Not enough usable numerical features."
    )

y_binary = df["is_anomaly"]
y_type = df["anomaly_type"]

print("\nFEATURES USED FOR TRAINING:", len(feature_columns))

for index, feature in enumerate(feature_columns, start=1):
    print(f"{index}. {feature}")


# ==========================================
# 4. SPLIT DATASET: 70% / 15% / 15%
# ==========================================

try:
    (
        X_train,
        X_temp,
        y_train,
        y_binary_temp,
        y_type_train,
        y_type_temp
    ) = train_test_split(
        X,
        y_binary,
        y_type,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y_type
    )

    (
        X_validation,
        X_test,
        y_binary_validation,
        y_binary_test,
        y_type_validation,
        y_type_test
    ) = train_test_split(
        X_temp,
        y_binary_temp,
        y_type_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_type_temp
    )

except ValueError as error:
    raise SystemExit(f"ERROR during dataset splitting: {error}")


# ==========================================
# 5. BALANCE BINARY TRAINING DATA
# ==========================================

print("\nBALANCING BINARY TRAINING DATA")
print("-" * 45)

normal_indices = y_train[y_train == 0].index
anomaly_indices = y_train[y_train == 1].index

if len(normal_indices) == 0 or len(anomaly_indices) == 0:
    raise SystemExit("ERROR: Both classes must exist in training data.")

minority_count = min(
    len(normal_indices),
    len(anomaly_indices)
)

rng = np.random.default_rng(RANDOM_SEED)

selected_normal = rng.choice(
    normal_indices.to_numpy(),
    size=minority_count,
    replace=False
)

selected_anomaly = rng.choice(
    anomaly_indices.to_numpy(),
    size=minority_count,
    replace=False
)

balanced_indices = np.concatenate(
    [selected_normal, selected_anomaly]
)

rng.shuffle(balanced_indices)

X_train_balanced = X_train.loc[balanced_indices]
y_train_balanced = y_train.loc[balanced_indices]

print("Before balancing:")
print(y_train.value_counts().sort_index())

print("\nAfter balancing:")
print(y_train_balanced.value_counts().sort_index())


# ==========================================
# 6. TRAIN BINARY ANOMALY DETECTION MODEL
# ==========================================

print("\nTraining binary anomaly detection model...")

binary_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2
    ))
])

binary_model.fit(
    X_train_balanced,
    y_train_balanced
)

print("Binary model training completed.")


# ==========================================
# 7. TRAIN ANOMALY TYPE CLASSIFICATION MODEL
# ==========================================

print("\nTraining anomaly type classification model...")

anomaly_train_mask = y_train == 1

X_type_train = X_train.loc[anomaly_train_mask]
y_type_train_anomaly = y_type_train.loc[anomaly_train_mask]

if y_type_train_anomaly.nunique() < 2:
    raise SystemExit(
        "ERROR: Not enough anomaly types in training data."
    )

type_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2
    ))
])

type_model.fit(
    X_type_train,
    y_type_train_anomaly
)

print("Anomaly type model training completed.")

print("\nANOMALY TYPES LEARNED:")
for anomaly_type in type_model.classes_:
    print(anomaly_type)


# ==========================================
# 8. VALIDATION PERFORMANCE
# ==========================================

print("\n" + "=" * 55)
print("VALIDATION PERFORMANCE")
print("=" * 55)

validation_predictions = binary_model.predict(X_validation)

print(
    "Validation Accuracy:",
    f"{accuracy_score(y_binary_validation, validation_predictions) * 100:.2f}%"
)

print("\nValidation Classification Report:")
print(classification_report(
    y_binary_validation,
    validation_predictions,
    labels=[0, 1],
    target_names=["NORMAL", "ANOMALY"],
    zero_division=0
))


# ==========================================
# 9. FINAL TEST PERFORMANCE
# ==========================================

print("\n" + "=" * 55)
print("FINAL TEST PERFORMANCE")
print("=" * 55)

test_predictions = binary_model.predict(X_test)

print(
    "Test Accuracy:",
    f"{accuracy_score(y_binary_test, test_predictions) * 100:.2f}%"
)

print("\nBinary Classification Report:")
print(classification_report(
    y_binary_test,
    test_predictions,
    labels=[0, 1],
    target_names=["NORMAL", "ANOMALY"],
    zero_division=0
))

print("\nBinary Confusion Matrix:")
print(confusion_matrix(
    y_binary_test,
    test_predictions,
    labels=[0, 1]
))


# ==========================================
# 10. ANOMALY TYPE TEST PERFORMANCE
# ==========================================

print("\n" + "=" * 55)
print("ANOMALY TYPE CLASSIFICATION PERFORMANCE")
print("=" * 55)

test_anomaly_mask = y_binary_test == 1

X_type_test = X_test.loc[test_anomaly_mask]
y_type_test_anomaly = y_type_test.loc[test_anomaly_mask]

if len(X_type_test) > 0:

    type_test_predictions = type_model.predict(X_type_test)

    print(
        "Anomaly Type Accuracy:",
        f"{accuracy_score(y_type_test_anomaly, type_test_predictions) * 100:.2f}%"
    )

    print("\nAnomaly Type Classification Report:")
    print(classification_report(
        y_type_test_anomaly,
        type_test_predictions,
        labels=type_model.classes_,
        target_names=type_model.classes_,
        zero_division=0
    ))

    print("\nAnomaly Type Confusion Matrix:")
    print(confusion_matrix(
        y_type_test_anomaly,
        type_test_predictions,
        labels=type_model.classes_
    ))

else:
    print("No anomalous records found in the test set.")


# ==========================================
# 11. GET PARAMETERS FROM TERMINAL
# ==========================================

def get_user_input():

    input_values = []

    print("\n" + "=" * 55)
    print("ENTER QUEUE PARAMETERS")
    print("=" * 55)

    print("Enter one numerical value per parameter.")
    print("You can type NA for a missing value.")
    print("Type EXIT at any prompt to stop.")

    for index, feature in enumerate(feature_columns, start=1):

        while True:

            value = input(
                f"{index}. {feature} : "
            ).strip()

            if value.lower() in ["exit", "quit", "q"]:
                return None

            if value.upper() in ["NA", "NAN", ""]:
                input_values.append(np.nan)
                break

            try:
                numeric_value = float(value)

                if not np.isfinite(numeric_value):
                    print("Enter a finite numerical value or NA.")
                    continue

                input_values.append(numeric_value)
                break

            except ValueError:
                print("Invalid input. Enter a numerical value or NA.")

    return pd.DataFrame(
        [input_values],
        columns=feature_columns
    )


# ==========================================
# 12. PREDICT QUEUE CONDITION
# ==========================================

def predict_queue(input_data):

    anomaly_prediction = binary_model.predict(input_data)[0]

    anomaly_probabilities = binary_model.predict_proba(input_data)[0]

    binary_classes = list(binary_model.classes_)

    anomaly_probability = (
        anomaly_probabilities[binary_classes.index(1)] * 100
    )

    normal_probability = 100 - anomaly_probability

    print("\n" + "=" * 55)
    print("             MODEL PREDICTION RESULT")
    print("=" * 55)

    if anomaly_prediction == 0:

        print("QUEUE STATUS        : NORMAL")
        print("PREDICTED TYPE      : NORMAL")
        print(f"NORMAL PROBABILITY  : {normal_probability:.2f}%")
        print(f"ANOMALY PROBABILITY : {anomaly_probability:.2f}%")

    else:

        type_prediction = type_model.predict(input_data)[0]

        type_probabilities = type_model.predict_proba(input_data)[0]

        type_classes = list(type_model.classes_)

        type_confidence = max(type_probabilities) * 100

        print("QUEUE STATUS        : ANOMALY DETECTED")
        print(f"PREDICTED TYPE      : {type_prediction}")
        print(f"ANOMALY PROBABILITY : {anomaly_probability:.2f}%")
        print(f"TYPE CONFIDENCE     : {type_confidence:.2f}%")

    print("=" * 55)


# ==========================================
# 13. CONTINUOUS PREDICTION LOOP
# ==========================================

print("\n" + "=" * 55)
print("       BANKING QUEUE ANOMALY DETECTION")
print("=" * 55)

print("Model training and evaluation completed.")
print("You can now test different queue parameter values.")

while True:

    input_data = get_user_input()

    if input_data is None:
        print("\nPrediction program stopped.")
        break

    predict_queue(input_data)

    print("\nDo you want to test another set of parameters?")

    choice = input(
        "Press ENTER to continue or type EXIT to stop: "
    ).strip()

    if choice.lower() in ["exit", "quit", "q"]:
        print("\nPrediction program stopped.")
        break

print("\nThank you for using the banking queue anomaly detection model.")

