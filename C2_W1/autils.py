import os
import numpy as np
from pathlib import Path


def get_root_folder():
    """
    Locate the root project folder by searching for a marker file (e.g., .projectroot).
    If not found, assume the current working directory is the root.
    """
    # Start from the current working directory (for Jupyter Notebooks)
    current_path = Path(os.getcwd()).resolve()

    # If running as a script, start from the script's directory
    if "__file__" in globals():
        current_path = Path(__file__).resolve().parent

    # Traverse up the directory tree to find the root folder
    for parent in current_path.parents:
        if (parent / ".projectroot").exists():  # Check for a marker file
            return parent
    return current_path  # Fallback to the current directory


def get_folder_path(folder_name):
    """
    Get the absolute path of a folder in the root project folder.
    """
    root_folder = get_root_folder()
    folder_path = root_folder / folder_name
    return folder_path.resolve()


def get_data_set_path():
    resources_dir: Path = get_folder_path("_resources_ML_spec")
    dataset_path: Path = resources_dir / "C2_W1" / "data"
    return dataset_path


def load_data_code():
    data_path: Path = get_data_set_path()
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


def load_data():
    data_path: Path = get_data_set_path()
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "X.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


def load_weights():
    data_path: Path = get_data_set_path()
    w1 = np.load(data_path / "w1.npy")
    b1 = np.load(data_path / "b1.npy")
    w2 = np.load(data_path / "w2.npy")
    b2 = np.load(data_path / "b2.npy")
    return w1, b1, w2, b2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
