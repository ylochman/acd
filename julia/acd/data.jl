using Random, LinearAlgebra, MAT

include("helpers.jl")

function generate_data(n, p; min_H_eigval=10, max_H_eigval=100, p_outlier=0.5)
    """ Create synthetic camera graph with N cameras and K observed noisy relative rotations, where k = max(N-1, ⌈pN(N-1)/2⌉). Noise perturbations achieved by left-multiplying with exp([Δw]_x), where Δw ~ 𝒩(0, H^{-1}).
     Args:
        n -- number of cameras/poses
        p -- fraction of observed relative rotations, p in (0,1]
        min_H_eigval/max_H_eigval -- eigenvalue range of the precision matrices (== Hessians)
    Returns:
        R -- nx3x3 matrix of absolute rotations
        Rrel -- nxnx3x3 matrix of observed noisy relative rotations
        H -- nxnx3x3 matrix of Hessians
        p_outlier -- fraction of outlying relative rotations
    """
    k = n * (n-1) ÷ 2
    k_observed = max(n-1, Int(ceil(k * p) ÷ 1))
    k_outliers = Int(ceil(k_observed * p_outlier) ÷ 1)
    
    triu_indices = findall(!iszero, triu(ones(n,n),1))
    # Observation pattern
    W = Bool.(zeros(n,n))
    # Pattern of outliers
    O = Bool.(zeros(n,n))
    while true
        W .= Bool.(zeros(n,n))
        O .= Bool.(zeros(n,n))
        perm_idx = randperm(k)
        W[triu_indices[perm_idx[1:k_observed]]] .= true
        O[triu_indices[perm_idx[1:k_outliers]]] .= true
        W .= W + W'
        O .= O + O'
        if all(sum(W-O,dims=1).>0)
            break
        end
    end

    # Absolute rotations
    R = zeros(n,3,3)
    for i=1:n
        R[i,:,:] .= exp(skewsym(normalize(rand(3))*rand()*2*pi))
    end

    # Relative rotations and corresponding Hessians
    Rrel = zeros(n,n,3,3)
    H = zeros(n,n,3,3)
    for i=1:n
        for j=i+1:n
            if W[i,j] > 0 # observed
                if O[i,j] > 0 # outlier
                    Rrel[i,j,:,:] .= exp(skewsym(normalize(rand(3))*rand()*2*pi))
                    eigvecs = exp(skewsym(normalize(rand(3))*rand()*2*pi))
                    eigvals = rand(3) .* 0.9 .+ 0.1
                    H[i,j,:,:] .= eigvecs * diagm(eigvals) * eigvecs'
                else # inlier
                    Rrel_ij = R[j,:,:] * R[i,:,:]'
                    if min_H_eigval===Inf && max_H_eigval===Inf # noiseless
                        H[i,j,:,:] .= Float64.(Matrix(I,3,3))
                        Rrel[i,j,:,:] .= Rrel_ij
                    else # noisy
                        eigvecs = exp(skewsym(normalize(rand(3))*rand()*2*pi))
                        eigvals = rand(3) .* (max_H_eigval - min_H_eigval) .+ min_H_eigval
                        H[i,j,:,:] .= eigvecs * diagm(eigvals) * eigvecs'
                        dw = eigvecs * (randn(3,1) ./ sqrt.(eigvals))
                        Rrel[i,j,:,:] .= exp(skewsym(dw)) * Rrel_ij
                    end
                end
                Rrel[j,i,:,:] = Rrel[i,j,:,:]'
                H[j,i,:,:] = H[i,j,:,:]'
            end
        end
    end
    return R, Rrel, H
end

function read_matlab_data(matfile)
    data = matread(matfile)
    n = data["Rgt"].size[1] ÷ 3 # number of cameras
    
    # Absolute rotations
    R_true = zeros(n,3,3)
    for i=1:n
        R_true[i,:,:] = data["Rgt"][3*i-2:3*i,:]
    end

    # Relative rotations and corresponding Hessians
    Rrel = zeros(n,n,3,3)
    H = zeros(n,n,3,3)
    for i=1:n
        for j=i+1:n
            Rrel[i,j,:,:] = data["Rrel"][3*i-2:3*i, 3*j-2:3*j]
            Rrel[j,i,:,:] = Rrel[i,j,:,:]'
            H[i,j,:,:] = data["H"][3*i-2:3*i, 3*j-2:3*j]
            H[j,i,:,:] = H[i,j,:,:]'
        end
    end
    return R_true, Rrel, H
end