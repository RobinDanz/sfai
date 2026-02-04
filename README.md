# sfai
UNINE Master Thesis Project

Automatic segmentation tool using SAM models.

## Quick Start

### Install the sfai tool
1. Clone the [repo](https://github.com/RobinDanz/sfai):

```
git clone https://github.com/RobinDanz/sfai.git
```

2. Navigate into the `sfai` folder:

```
cd sfai
```

3. Install the tool as a pip package:

```
pip install .
```

### CUDA support
The tool supports CUDA if it is installed on your system. To run segmentation on the GPU follow the steps below.

1. Uninstall current torch & torchvison package:

```
pip uninstall torch torchvision
```

2. Install CUDA-compatible torch version:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The version depends on your GPU and installed CUDA version. Check the exact command on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Usage

sfai is a CLI tool split into different subtool:

- Automatic segmentation
- COCO to Biigle converter
- Recursive file copy utility



### Segmentation
```sh
sfai segment --help

usage: sfai segment [-h] -c CONFIG

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Config file
```

Example with a config file: 
```sh
sfai segment -c /path/to/config.yaml
```

Take a look at the `config.example.yaml` file to get started.

### COCO to Biigle converter

```sh
sfai coco2biigle --help

usage: sfai coco2biigle [-h] -c COCO -t LABEL_TREE_PATH [-o OUT_DIR] [-p PROJECT_NAME] [-v VOLUME_NAME]

options:
  -h, --help            show this help message and exit
  -c COCO, --coco COCO  Path to a coco file.
  -t LABEL_TREE_PATH, --label_tree_path LABEL_TREE_PATH
                        Path to a Biigle Label Tree zip file.
  -o OUT_DIR, --out_dir OUT_DIR
                        Output path. Default: Current directory (C:\DEV\sfai\results\coco2biigle)
  -p PROJECT_NAME, --project_name PROJECT_NAME
                        Output project name. Default: project01
  -v VOLUME_NAME, --volume_name VOLUME_NAME
                        Output volume name. Default: volume01
```

Example to create a BIIGLE volume in the current directory.

```sh
sfai coco2biigle -c annotaions.json -t label_tree.zip
```



### File copy utility

Allows to copy or move files from multiple subdirectories into a single directory.

```sh
sfai cpfiles --help

usage: sfai cpfiles [-h] -s SOURCE -d DEST [-m] [-e [EXTENSIONS ...]]

options:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        Source folder. Subfolders will be explored
  -d DEST, --dest DEST  Destination folder.
  -m, --move            Move files instead copy
  -e [EXTENSIONS ...], --extensions [EXTENSIONS ...]
                        File extensions
```

Example:

```sh
sfai cpfiles -s ./test/ -d ./newfolder/ -m
```

This command will move files from test folder to newfolder. Newfolder si created if it does not exist.



