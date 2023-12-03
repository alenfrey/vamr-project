from pathlib import Path

# project root directory path
PROJECT_ROOT_DIR_PATH = Path(__file__).parent

# subdirectories relative to project root directory
DATA_DIR_PATH = PROJECT_ROOT_DIR_PATH / "data"
# its good practice to have a raw data folder that is kept as originally downloaded
RAW_DATA_DIR_PATH = DATA_DIR_PATH / "raw"
# folder for processed data
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"

REPORT_DIR_PATH = PROJECT_ROOT_DIR_PATH / "reports"
