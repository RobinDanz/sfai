# soil-fauna-ai
UNINE Project

## Usage
1. Clone this repo
2. Create a venv
3. Install dependencies from requirements.txt

## Project organization
### `soilfauna`
Main project folder

### `models`
Models will be downloaded there

### `scripts`
Will contain utility scripts in the future

## Installation

1. Clone the project

2. Run `pip install .`

3. Install SAM support: `pip install ".[sam]" --no-deps`

4. Install PyTorch (required for SAM)
-  CPU-Only:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

- CUDA (GPU)
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`



