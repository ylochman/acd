# ACD: Fast and Robust Rotation Averaging with Anisotropic Coordinate Descent

Julia code for the paper [Fast and Robust Rotation Averaging with Anisotropic Coordinate Descent](https://ylochman.github.io/acd) that proposes a fast solver for [anisotropic rotation averaging](https://ylochman.github.io/anisotropic-ra) where the uncertainties of optimized two-view relative rotations are incorporatess into the optimization of absolute rotations.

A C++ implementation with python bindings will be released soon.

## Algorithm details
**Input**: relative rotations `Rrel` (`Rrel_ij ≈ R_j @ R_i.T`), corresponding Hessians `H`.

**Output**: absolute rotations `R`.

The **pseudo-code** of ACD algorithm is:
```python
R ← initialize_rotations(n)        # R is nx3x3
N ← construct_cost_matrix(Rrel, H) # N is nxnx3x3
                                   # N[i,j] = 
                                   #    (tr(H_ij)/2 I − H_ij) @ Rrel_ij
                                   #      if Rrel_ij exists
                                   #    0 otherwise
for iter in range(max_iter):
    for k in shuffle({1,...,n}):
        R[k] ← project_on_SO3(sum(bmm(N[k].transpose(2,3), R), axis=1))
    if converged:
        break
```

## Running the code
The code is tested on Julia 1.11.5.
The required julia packages are `LinearAlgebra`, `StatsBase`, `MAT`, and can be installed by running [`requirements.jl`](./requirements.jl).

The solver code is in [`acd.jl`](./acd/acd.jl).

A demo example is in [`demo.jl`](./demos/demo.jl).

## Citation
If you found this work useful, consider citing:
```bibtex
@article{lochman2025fast,
    author    = {Lochman, Yaroslava and Olsson, Carl and Zach, Christopher},
    title     = {Fast and Robust Rotation Averaging with Anisotropic Coordinate Descent},
    journal   = {arXiv preprint arXiv:2506.01940},
    year      = {2025},
}
```

