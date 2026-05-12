# Julia code for ACD

## Installation

```julia
include("./requirements.jl")
```

The code is tested on Julia 1.11.5.


## Usage

```julia
include("./acd/acd.jl");

R_est, status, solver_runtime, obj_value = acd(Rrel, H);
```

Change paths if you have a different file organization.

The inputs `Rrel` and `H` have the following structure:
```julia
# Rrel: nxnx3x3 matrix s.t.:
#     - Rrel[i,j] is an estimate approximating R_j @ R_i.T
#     - Rrel[i,j] .== 0 if unobserved
# H: nxnx3x3 matrix s.t.:
#     - H[i,j] is a Hessian of Rrel[i,j] estimate
#     - H[i,j] .== 0 if unobserved
```

We provide a function that loads inputs and GT from the matlab file:
```julia
include("./acd/data.jl");

R_true, Rrel, H = read_matlab_data("../data/lu_sphinx.mat");
```

The solver code can be found [here](./acd/acd.jl).

A demo with synthetic and real example can be found [here](./demo.jl).