from pathlib import Path

# project root directory path
PROJECT_ROOT_DIR_PATH = Path(__file__).parent

# subdirectories relative to project root directory
DATA_DIR_PATH = PROJECT_ROOT_DIR_PATH / "data"

# its good practice to have a raw data folder that is kept as originally downloaded
RAW_DATA_DIR_PATH = DATA_DIR_PATH / "raw"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"

# dataset paths
PARKING_DATA_DIR_PATH = RAW_DATA_DIR_PATH / "parking"
KITTI_DATA_DIR_PATH = RAW_DATA_DIR_PATH / "kitti"
MALAGA_DATA_DIR_PATH = RAW_DATA_DIR_PATH / "malaga-urban-dataset-extract-07"


REPORT_DIR_PATH = PROJECT_ROOT_DIR_PATH / "report"
