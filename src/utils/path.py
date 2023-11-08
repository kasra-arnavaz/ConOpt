import os


def get_next_numbered_path(path: str):
    os.makedirs(path, exist_ok=True)
    number = len(os.listdir(path)) + 1
    return f"{path}/{number}"
