using LinearAlgebra, StatsBase

function acd(Rrel, H; anisotropic_cost=true, init="zero", shuffle_k=true, max_iters=1_000, eps_abs=1.0e-12, eps_rel=1.0e-12, print_frequency=-1)
    """ Solve rotation averaging with an ACD solver.
    Args:
        Rrel -- nxnx3x3 matrix of relative rotations (unobserved blocks should be all zeros)
        H -- nxnx3x3 matrix of Hessians
        anisotropic_cost -- whether to use Hessians or not
        init -- type of initialization (recommended: zero)
        shuffle_k -- whether to shuffle indices at each iteration (recommended: true)
        max_iters -- maximum number of iterations
        eps_abs -- absoute tolerance
        eps_rel -- relative tolerance
    Returns:
        R_est -- nx3x3 matrix of estimated absolute rotations
        stat -- status of optimization
        runtime -- solver runtime
        obj_val -- objective value at the solution
    """
    cost_matrix, constant_term, observed_indices = construct_cost_matrix(Rrel, H; anisotropic=anisotropic_cost)
    n = Rrel.size[1]

    runtime = @elapsed begin

    R_est = zeros(n,3,3)
    R_est .= initialize_rotations(n, init)

    converged = false
    obj_val = constant_term - sum(cost_matrix .* RRT(R_est))
    if print_frequency!==-1
            display("iter #0: obj_val: $obj_val")
        end
    obj_val_next = Inf
    for iter=1:max_iters
        R_next = R_est
        for k=(shuffle_k ? sample(1:n, n, replace=false) : 1:n)
            obs = observed_indices[k]
            R_next[k,:,:] .= project_on_SO3(bmmWTR(cost_matrix[k,obs,:,:], R_next[obs,:,:]))
        end
        obj_val_next = constant_term - sum(cost_matrix .* RRT(R_est))
        converged = (abs(obj_val_next - obj_val) < eps_abs || abs(obj_val_next - obj_val) / max(abs(obj_val),1) < eps_rel)
        if print_frequency!==-1 && (iter % print_frequency == 0 || converged)
            display("iter #$iter: obj_val: $obj_val_next, delta_obj: $(obj_val_next - obj_val)")
        end
        R_est .= R_next
        obj_val = obj_val_next
        if converged
            break
        end
    end
    end

    stat = converged ? "converged" : "reached the maximum number of iterations"
    return R_est, stat, runtime, obj_val
end

function construct_cost_matrix(Rrel, H; anisotropic=true)
    """
    Args:
        Rrel -- nxnx3x3 matrix of relative rotations (unobserved blocks should be all zeros)
        H -- nxnx3x3 matrix of Hessians 
    Returns:
        cost_matrix -- nxnx3x3 symmetric cost matrix for rotation averaging
        constant_term -- constant term in the objective
        observed_indices -- indices of observed relative rotations
    """
    n = Rrel.size[1]
    I3x3 = Float64.(Matrix(I,3,3))
    k_observed = 0

    # Make isotropic/anisotropic cost matrix
    constant_term = 0
    cost_matrix = zeros(n,n,3,3)
    observed_indices = Vector{Vector{Int}}(undef, n)
    for i=1:n
        observed_indices[i] = filter(x -> ((x != i) && (sum(Rrel[x,i,:,:].^2) > 0)), 1:n)
        for j in observed_indices[i]
            if j > i
                k_observed += 1
                if anisotropic
                    H_ij = H[i,j,:,:]
                    M_ij = tr(H_ij) / 2 * I3x3 - H_ij
                    cost_matrix[i,j,:,:] .= M_ij * Rrel[i,j,:,:]
                    constant_term += tr(M_ij)
                else
                    cost_matrix[i,j,:,:] .= Rrel[i,j,:,:]
                    constant_term += 6
                end
                cost_matrix[j,i,:,:] = cost_matrix[i,j,:,:]'
            end
        end
    end
    k_observed *= 2
    return cost_matrix / k_observed, constant_term / k_observed, observed_indices
end

function initialize_rotations(n, init; max_axis_angle_norm=360)
    """ Initialize rotations
    Args:
        init -- one of: ["zero", "id", "randn", "svd", "axis_angle"]
    """
    R = zeros(n,3,3)
    if init=="id"
        R .= kron(ones(n,1),Float64.(Matrix(I,3,3)));
    elseif init in ["randn", "svd", "axis_angle"]
        R .= randn(n,3,3)
        if init in ["svd", "axis_angle"]
            max_axis_angle_norm *= 2 * pi / 360
            for i=1:n
                if init=="svd"
                    R[i,:,:] .= svd(R[i,:,:]).U
                else # init=="axis_angle"
                    R[i,:,:] .= exp(skewsym(normalize(rand(3)) * rand() * max_axis_angle_norm))
                end
            end
        end
    end
    return R
end

function bmmWTR(W, R)
    n = R.size[1]
    return reshape(permutedims(W, [3 2 1]),3,3*n) * reshape(permutedims(R, [3 2 1]),3,3*n)'
end

function RRT(R)
    n = R.size[1]
    RRT = zeros(n,n,3,3)
    for i=1:n, j=1:n
        RRT[i,j,:,:] = R[j,:,:] * R[i,:,:]'
    end
    return RRT
end