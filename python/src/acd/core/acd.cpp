#include "acd.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>

namespace acd {
    // Helpers
    Tensor4 RRT(const std::vector<Matrix3>& R) {
        int n = R.size();
        Tensor4 result(n, std::vector<Matrix3>(n));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result[i][j] = R[j] * R[i].transpose();
            }
        }
        return result;
    }

    Matrix3 bmmWTR(const std::vector<Matrix3>& W,
                const std::vector<Matrix3>& R) {
        int n = R.size();

        Eigen::MatrixXd Wm(3, 3 * n);
        Eigen::MatrixXd Rm(3, 3 * n);

        for (int i = 0; i < n; ++i) {
            Wm.block<3,3>(0, 3*i) = W[i].transpose();
            Rm.block<3,3>(0, 3*i) = R[i].transpose();
        }

        return Wm * Rm.transpose();
    }

    Matrix3 project_on_SO3(const Matrix3& M) {
        Eigen::JacobiSVD<Matrix3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Matrix3 U = svd.matrixU();
        Matrix3 Vt = svd.matrixV().transpose();

        Matrix3 R = U * Vt;

        if (R.determinant() < 0) {
            Matrix3 D = Matrix3::Identity();
            D(2, 2) = -1;  // diag([1, 1, -1])
            R = U * D * Vt;
        }

        return R;
    }


    // Main function
    void acd_cpp(std::vector<Matrix3>& R_est,
                const Tensor4& cost_matrix,
                double constant_term,
                const std::vector<std::vector<int>>& observed_indices,
                int max_iters,
                double eps_abs,
                double eps_rel,
                int print_frequency,
                bool shuffle_k,
                bool& converged,
                double& obj_val) {

        int n = R_est.size();

        auto compute_obj = [&](const std::vector<Matrix3>& R) {
            Tensor4 rrt = RRT(R);
            double sum = 0.0;

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    sum += (cost_matrix[i][j].cwiseProduct(rrt[i][j])).sum();
                }
            }
            return constant_term - sum;
        };

        obj_val = compute_obj(R_est);

        if (print_frequency != -1) {
            std::cout << "iter #0: obj_val: " << obj_val << std::endl;
        }

        double obj_val_next = std::numeric_limits<double>::infinity();

        std::mt19937 rng(std::random_device{}());

        for (int it = 1; it <= max_iters; ++it) {

            std::vector<Matrix3> R_next = R_est;

            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);

            if (shuffle_k) {
                std::shuffle(order.begin(), order.end(), rng);
            }

            for (int k : order) {
                const auto& obs = observed_indices[k];
                if (obs.empty()) continue;

                std::vector<Matrix3> W_subset;
                std::vector<Matrix3> R_subset;

                for (int idx : obs) {
                    W_subset.push_back(cost_matrix[k][idx]);
                    R_subset.push_back(R_next[idx]);
                }

                Matrix3 M = bmmWTR(W_subset, R_subset);
                R_next[k] = project_on_SO3(M);
            }

            obj_val_next = compute_obj(R_next);

            double diff = std::abs(obj_val_next - obj_val);
            double rel_diff = diff / std::max(std::abs(obj_val), 1.0);

            converged = (diff < eps_abs) || (rel_diff < eps_rel);

            if (print_frequency != -1 &&
                (it % print_frequency == 0 || converged)) {
                std::cout << "iter #" << it
                        << ": obj_val: " << obj_val_next
                        << ", delta_obj: " << (obj_val_next - obj_val)
                        << std::endl;
            }

            R_est = R_next;
            obj_val = obj_val_next;

            if (converged) break;
        }
    }
}