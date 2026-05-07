using Random, LinearAlgebra

function skewsym(t)
    return [  0  -t[3]  t[2];
             t[3]  0   -t[1];
            -t[2] t[1]   0  ]
end

function R2w(R)
    logR = log(R)
    return [logR[3,2], logR[1,3], logR[2,1]]
end

function project_on_SO3(M)
    U, _, V = svd(M)
    R = U * V'
    if det(R) < 0
        R .= U * Diagonal([1,1,-1]) * V'
    end
    return R
end

function rms(diff)
    return sqrt(mean((diff[:]).^2))
end

function normalize(x, p::Real=2)
    return x ./ (sum(x.^p)).^(1/p)
end

function align_rotations(R1, R2)
    n = size(R1,1)
    M_align = zeros(3, 3)
    for i = 1:n
        M_align .+= R1[i,:,:]' * R2[i,:,:]
    end
    R_align = project_on_SO3(M_align)
    for i = 1:n
        R1[i,:,:] .= R1[i,:,:] * R_align
    end
    return R1
end