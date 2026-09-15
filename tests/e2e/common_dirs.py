import os


def get_test_model_dir() -> str:
    return os.environ.get("MILES_SCRIPT_MODEL_DIR", "/root/models")


def get_test_data_dir() -> str:
    return os.environ.get("MILES_SCRIPT_DATA_DIR", "/root/datasets")
