"""Closed request and predictor schemas for the deployed RUL regressor."""

FEATURE_COLUMNS = (
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *(f"sensor_{number:02d}" for number in range(1, 22)),
    *(
        name
        for sensor in ("sensor_03", "sensor_04", "sensor_11")
        for name in (f"{sensor}_mean10", f"{sensor}_slope10")
    ),
    *(f"setting_{number}_mean10" for number in range(1, 4)),
    "history_count",
)
SETTING_COLUMNS = ("setting_1", "setting_2", "setting_3")
SENSOR_COLUMNS = tuple(f"sensor_{number:02d}" for number in range(1, 22))
OBSERVATION_COLUMNS = ("cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SHA256 = "e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe"
ARTIFACT_MANIFEST_SHA256 = "c280467cc28509fb04e63ad6ba1c26c86b8e16a09636b8ec1ce9e952c2641e09"
TRAINING_MANIFEST_SHA256 = "2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd"
BEST_ITERATION = 35
SAVED_TREE_COUNT = BEST_ITERATION + 1
MAX_INSTANCES = 100
MAX_OBSERVATIONS_PER_INSTANCE = 10
MAX_INT32 = 2**31 - 1
DRIFT_MINIMUM_BATCH = 20
DRIFT_ALERT_RATE = 0.1
