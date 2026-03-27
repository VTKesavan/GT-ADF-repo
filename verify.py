import sys

print("=" * 45)
print("GT-ADF Environment Verification")
print("=" * 45)

# Python
print(f"\nPython:  {sys.version}")

# PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch: NOT FOUND - run Step 4")

# PyG
try:
    import torch_geometric
    print(f"PyG:     {torch_geometric.__version__}")
except ImportError:
    print("PyG:     NOT FOUND - run Step 5")

# NumPy
try:
    import numpy
    print(f"NumPy:   {numpy.__version__}")
except ImportError:
    print("NumPy:   NOT FOUND - run Step 6")

# Pandas
try:
    import pandas
    print(f"Pandas:  {pandas.__version__}")
except ImportError:
    print("Pandas:  NOT FOUND - run Step 6")

# Sklearn
try:
    import sklearn
    print(f"Sklearn: {sklearn.__version__}")
except ImportError:
    print("Sklearn: NOT FOUND - run Step 6")

# PyYAML
try:
    import yaml
    print(f"PyYAML:  {yaml.__version__}")
except ImportError:
    print("PyYAML:  NOT FOUND - run Step 6")

print("\n" + "=" * 45)
print("If all lines show version numbers: ALL OK")
print("=" * 45)
