from pathlib import Path

slash = "\\"

WORKDIR = str(Path.cwd())
SOURCEDIR = WORKDIR + slash + "source"
MODELDIR = WORKDIR + slash + "models"
CACHE_FOLDER = WORKDIR + slash + "mycache"

STATIC_DATA_FOLDER = "static_data"
TM_DATA_FOLDER =  "tm_data"

TEST_FOLDER = WORKDIR + slash + "test_dataset_1"
TEST_STATIC_DATA_FOLDER = TEST_FOLDER + slash + STATIC_DATA_FOLDER
TEST_STATIC_DATA = TEST_STATIC_DATA_FOLDER + slash + "train_test_static_data.xlsx"
TEST_TM_DATA_FOLDER = TEST_FOLDER + slash + TM_DATA_FOLDER
TEST_DATA_CACHE_FILE = CACHE_FOLDER + slash + "test.csv"

TRAIN_FOLDER = WORKDIR + slash + "train_dataset"
TRAIN_STATIC_DATA_FOLDER = TRAIN_FOLDER + slash + STATIC_DATA_FOLDER
TRAIN_STATIC_DATA = TRAIN_STATIC_DATA_FOLDER + slash + "train_test_static_data.xlsx"
TRAIN_TM_DATA_FOLDER = TRAIN_FOLDER + slash + TM_DATA_FOLDER
TRAIN_DATA_CACHE_FILE = CACHE_FOLDER + slash + "train.csv"

MODEL_NAME = MODELDIR + slash + "catboost_model.pkl"

ALLOWED_STATIC_DATA_FORMATS = { ".xlsx" }
ALLOWED_TM_DATA_FORMATS = { ".csv" }

PREPOCESSING_CONFIG = {
            'numeric_strategy': 'median',
            'categorical_strategy': 'most_frequent',
            'scaling': 'standard',
            'outlier_threshold': 3.0,
            'time_features': True,
            'batch_size': 1000,
            'verbose': True
        }

ID_TO_PARAM_NAME = {
    1: "debit",                         # "Дебит жидкости (объёмный), м3/сут",
    2: "water_cut_percent",             # "Обводненность (объёмная), %",
    3: "wellhead_pressure_atm",         # "Рбуф, атм",
    4: "line_pressure_atm",             # "Давление линейное, атм",
    5: "pump_intake_pressure_atm",      # "Давление на приеме насоса, атм",
    6: "esp_load_percent",              # "Загрузка ПЭД, %",
    7: "associated_gas",                # "Попутный газ",
    8: "current_frequency_hz",          # "Частота тока, Герц",
    9: "phase_a_current_a",             # "Ток фазы А, A (ампер)",
    10: "active_power_kw",              # "Мощность активная, кВт",
    11: "voltage_v",                    # "Напряжение, АВ Вольт",
    12: "casing_pressure_atm"           # "P затрубное, атм",
}