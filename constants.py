from pathlib import Path
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# project root directory path
PROJECT_ROOT_DIR_PATH = Path(__file__).parent

# subdirectories relative to project root directory
DATA_DIR_PATH = PROJECT_ROOT_DIR_PATH / "data"

# dataset paths
PARKING_DATA_DIR_PATH = DATA_DIR_PATH / "parking"
KITTI_DATA_DIR_PATH = DATA_DIR_PATH / "kitti"
MALAGA_DATA_DIR_PATH = DATA_DIR_PATH / "malaga-urban-dataset-extract-07"

REPORT_DIR_PATH = PROJECT_ROOT_DIR_PATH / "report"
