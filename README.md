# ACD: Fast and Robust Rotation Averaging with Anisotropic Coordinate Descent

[[project page](https://ylochman.github.io/acd)] | [[paper](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_516/paper.pdf)]

A fast solver for [anisotropic rotation averaging](https://ylochman.github.io/anisotropic-ra) where the uncertainties of optimized two-view relative rotations are incorporated into the optimization of absolute rotations.


The code is available in:
* [Julia](./julia)
* [python/C++](./python)

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

The anisotropic formulation used in this work was proposed in:
```bibtex
@article{olsson2025certifiably,
    author    = {Olsson, Carl and Lochman, Yaroslava and Malmport, Johan and Zach, Christopher},
    title     = {Certifiably Optimal Anisotropic Rotation Averaging},
    journal   = {arXiv preprint arXiv:2503.07353},
    year      = {2025},
}
```

