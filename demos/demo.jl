include("../acd/data.jl")
include("../acd/helpers.jl")
include("../acd/acd.jl")

function run_and_eval_acd(R_true, Rrel, H; anisotropic_cost=true)
    """
    Args:
        anisotropic_cost -- if true, uses the anisotropic cost
    """
    n = size(R_true, 1)
    R_est, stat, stime, obj_val = acd(Rrel, H; anisotropic_cost=anisotropic_cost)

    # Evaluate the solution
    R_est = align_rotations(R_est, R_true)
    fro_err = sqrt(sum((R_est - R_true).^2))
    angles = zeros(n)
    for i=1:n
        angles[i] = norm(R2w(R_true[i,:,:] * R_est[i,:,:]')) / pi * 180
    end 
    angular_err = rms(angles)
    display("Frobenius error: $fro_err, Angular error: $angular_err, Solver runtime: $stime")
end

display("Synthetic dataset:")
n = 2000 # number of cameras
p = 0.1  # proportion of observed relative measurements
min_H_eigval = 1  # set to Inf for noiseless data
max_H_eigval = 100 # set to Inf for noiseless data
R_true, Rrel, H = generate_data(n, p; min_H_eigval=min_H_eigval, max_H_eigval=max_H_eigval);
run_and_eval_acd(R_true, Rrel, H; anisotropic_cost=false)
run_and_eval_acd(R_true, Rrel, H; anisotropic_cost=true)

display("LU Sphinx dataset:")
R_true, Rrel, H = read_matlab_data("./data/lu_sphinx.mat");
run_and_eval_acd(R_true, Rrel, H; anisotropic_cost=false)
run_and_eval_acd(R_true, Rrel, H; anisotropic_cost=true)