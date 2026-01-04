import os
from pathlib import Path

ROOT_DIR = Path(Path(__file__).parent.parent.parent.parent.as_posix())
    
DEFAULT_MODEL_PATH = Path(os.path.join(ROOT_DIR, 'models'))
DEFAULT_MODEL = Path(os.path.join(DEFAULT_MODEL_PATH, 'sam2_b.pt'))

DEFAULT_OUTPUT_DIR = Path(os.path.join(ROOT_DIR, 'results', 'runs'))

DEFAULT_RUN_NAME = 'default'