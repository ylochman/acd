# Python code for ACD

## Installation
```bash
export EIGEN_DIR=./third_party/eigen-3.4.1 && pip install -e .
```

## Usage
```python
from acd import acd

R_est, status, solver_runtime, obj_value = acd(Rrel, H)
```

The inputs `Rrel` and `H` have the following structure:
```python
# Rrel: nxnx3x3 matrix s.t.:
#     - Rrel[i,j] is an estimate approximating R_j @ R_i.T
#     - Rrel[i,j] .== 0 if unobserved
# H: nxnx3x3 matrix s.t.:
#     - H[i,j] is a Hessian of Rrel[i,j] estimate
#     - H[i,j] .== 0 if unobserved
```

We provide a function that loads inputs and GT from the matlab file:
```python
from acd.data import read_matlab_data

R_true, Rrel, H = read_matlab_data("../data/lu_sphinx.mat")
```

A demo with synthetic and real example can be found [here](./demo.py).