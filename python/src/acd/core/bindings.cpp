#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "acd.hpp"

namespace py = pybind11;

// Forward declaration
void acd_cpp(std::vector<Eigen::Matrix3d> &R_est,
             const std::vector<std::vector<Eigen::Matrix3d>> &cost_matrix,
             double constant_term,
             const std::vector<std::vector<int>> &observed_indices,
             int max_iters,
             double eps_abs,
             double eps_rel,
             int print_frequency,
             bool shuffle_k);

// Wrapper
std::tuple<bool, py::array_t<double>, double>
acd_cpp_wrapper(py::array_t<double> R_est_np,
                py::array_t<double> cost_np,
                double constant_term,
                std::vector<std::vector<int>> observed_indices,
                int max_iters,
                double eps_abs,
                double eps_rel,
                int print_frequency,
                bool shuffle_k)
{

    auto R_buf = R_est_np.unchecked<3>(); // (n, 3, 3)
    auto C_buf = cost_np.unchecked<4>();  // (n, n, 3, 3)

    int n = R_buf.shape(0);

    std::vector<Eigen::Matrix3d> R_est(n);
    std::vector<std::vector<Eigen::Matrix3d>> cost_matrix(n, std::vector<Eigen::Matrix3d>(n));

    // Convert R_est
    for (int i = 0; i < n; ++i)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_est[i](r, c) = R_buf(i, r, c);

    // Convert cost_matrix
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    cost_matrix[i][j](r, c) = C_buf(i, j, r, c);

    bool converged = false;
    double obj_val = 100000.0;

    // Call ACD solver
    acd::acd_cpp(R_est, cost_matrix, constant_term,
                 observed_indices, max_iters,
                 eps_abs, eps_rel, print_frequency, shuffle_k, converged, obj_val);

    // Convert back to NumPy
    py::array_t<double> result({n, 3, 3});
    auto res_buf = result.mutable_unchecked<3>();

    for (int i = 0; i < n; ++i)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                res_buf(i, r, c) = R_est[i](r, c);

    return std::make_tuple(converged, result, obj_val);
}

PYBIND11_MODULE(_acd, m)
{
    m.def("acd_cpp", &acd_cpp_wrapper, "ACD solver (C++ backend)");
}