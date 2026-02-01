import os
from pathlib import Path

PROJECT_ROOT_DIR = Path(Path(__file__).parent.parent.parent.parent.as_posix())
WORK_DIR = Path(os.getcwd())
    
DEFAULT_MODEL_PATH = Path(os.path.join(PROJECT_ROOT_DIR, 'models'))
DEFAULT_MODEL = Path(os.path.join(DEFAULT_MODEL_PATH, 'sam2_b.pt'))

DEFAULT_OUTPUT_DIR = Path(os.path.join(WORK_DIR, 'results', 'runs'))

DEFAULT_RUN_NAME = 'default'

DEFAULT_COCO2BIIGLE_OUTPUT_DIR = Path(os.path.join(WORK_DIR, 'results', 'coco2biigle'))

BIIGLE_MODEL_FILES_DIR = Path(os.path.join(PROJECT_ROOT_DIR, 'data', 'biigle'))

# Segmentation default values
DEFAULT_HSV_LOWER_BOUND = [90,  40,  40]
DEFAULT_HSV_UPPER_BOUND = [145, 255, 255]

DEFAULT_TILE_ROWS = 5
DEFAULT_TILE_COLUMNS = 5