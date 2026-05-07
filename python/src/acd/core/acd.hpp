#pragma once

#include <Eigen/Dense>
#include <vector>

namespace acd {
    using Matrix3 = Eigen::Matrix3d;
    using Tensor4 = std::vector<std::vector<Matrix3>>;
    using RotationList = std::vector<Matrix3>;
    using CostMatrix = std::vector<std::vector<Matrix3>>;

    // Main function
    void acd_cpp(RotationList& R_est,
                const CostMatrix& cost_matrix,
                double constant_term,
                const std::vector<std::vector<int>>& observed_indices,
                int max_iters,
                double eps_abs,
                double eps_rel,
                int print_frequency,
                bool shuffle_k,
                bool& converged,
                double& obj_val);

    // Helpers
    Matrix3 project_on_SO3(const Matrix3& M);
    Matrix3 bmmWTR(const std::vector<Matrix3>& W,
                const std::vector<Matrix3>& R);

    CostMatrix RRT(const RotationList& R);
}