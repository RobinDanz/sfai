# Quick Start

## Install the soil-fauna-ai tool
1. Clone the [repo](https://github.com/RobinDanz/soil-fauna-ai):

```
git clone https://github.com/RobinDanz/soil-fauna-ai.git
```

2. Navigate into the `soil-fauna-ai` folder:

```
cd soil-fauna-ai
```

3. Install the tool as a pip package:

```
pip install .
```

## CUDA support
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