DEFAULT_TRAIN_SCRIPT: str = "train.py"
FULLY_ASYNC_TRAIN_SCRIPT: str = "train_async.py"


def get_train_script(*, fully_async: bool) -> str:
    return FULLY_ASYNC_TRAIN_SCRIPT if fully_async else DEFAULT_TRAIN_SCRIPT


def get_fully_async_args(*, fully_async: bool) -> str:
    if not fully_async:
        return ""
    return "--fully-async --pause-generation-mode in_place "
