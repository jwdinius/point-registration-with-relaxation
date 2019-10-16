//! c/c++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cppad/ipopt/solve.hpp>
#include "register.hpp"

using Dvec = CPPAD_TESTVECTOR(double);

//! forward declare
class ConstrainedObjective;
class PointRegRelaxation;

double consistency(arma::vec3 const & si, arma::vec3 const & tj, arma::vec3 const & sk,
        arma::vec3 const & tl) noexcept {
    double const dist_si_to_sk = arma::norm(si - sk, 2);
    double const dist_tj_to_tl = arma::norm(tj - tl, 2);
    return std::abs(dist_si_to_sk - dist_tj_to_tl);
}

WeightTensor generate_weight_tensor(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config) noexcept {
    WeightTensor weight = {};
    auto const & eps = config.epsilon;
    size_t const & m = source_pts.n_cols;
    size_t const & n = target_pts.n_cols;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < m; ++k) {
                for (size_t l = 0; l < n; ++l) {
                    if (i != k && j != l) {
                        arma::colvec si = source_pts.col(i);
                        arma::colvec tj = target_pts.col(j);
                        arma::colvec sk = source_pts.col(k);
                        arma::colvec tl = target_pts.col(l);
                        if (consistency(si, tj, sk, tl) < eps) {
                            weight[std::make_tuple(i, j, k, l)] = -1.;
                        }
                    }
                }
            }
        }
    }
    return weight;
}

//! destructor (nothing to do)
ConstrainedObjective::~ConstrainedObjective() { }

void ConstrainedObjective::operator()(ConstrainedObjective::ADvector &fgrad,
        ConstrainedObjective::ADvector const & z) noexcept {
    size_t curr_idx = 0;
    //! objective value
    fgrad[curr_idx] = 0.;
    for (size_t i = 0; i < m_; ++i) {
        for (size_t j = 0; j < n_; ++j) {
            for (size_t k = 0; k < m_; ++k) {
                for (size_t l = 0; l < n_; ++l) {
                    if (i != k && j != l) {
                        auto key = std::make_tuple(i, j, k, l);
                        if (weights_.find(key) != weights_.end()) {
                            fgrad[curr_idx] += static_cast<CppAD::AD<double>>(weights_[key] * z[i*n_ + j] * z[k*n_ + l]);
                        }
                    }
                }
            }
        }
    }

    //! constraints:
    //! for all i, \sum_j z_{ij} = 2 - n_ (there are m_ of these)
    for (size_t i = 0; i < m_; ++i) {
        fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(n_ - 2);
        for (size_t j = 0; j < n_; ++j) {
            fgrad[curr_idx] += static_cast<CppAD::AD<double>>(z[i*n_ + j]);
        }
    }

    fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(2*m_ - n_);
    for (size_t j = 0; j < n_; ++j) {
        fgrad[curr_idx] += static_cast<CppAD::AD<double>>(z[m_*n_ + j]);
    }

    for (size_t j = 0; j < n_; ++j) {
        fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(m_ - 1);
        for (size_t i = 0; i <= m_; ++i) {
            fgrad[curr_idx] += static_cast<CppAD::AD<double>>(z[i*n_ + j]);
        }
    }
}

PointRegRelaxation::~PointRegRelaxation() { }

PointRegRelaxation::status_t PointRegRelaxation::find_optimum(arma::colvec & sol) const noexcept {
    auto const & n_vars = ptr_obj_->state_length();
    auto const & n_constraints = ptr_obj_->num_constraints();
    sol.resize(n_vars);

    Dvec z(n_vars);
    Dvec z_lb(n_vars);
    Dvec z_ub(n_vars);
    for (size_t i = 0; i < n_vars; ++i) {
        z[i] = 0.;
        z_lb[i] = -1.;
        z_ub[i] = 1.;
    }

    Dvec constraints_lb(n_constraints);
    Dvec constraints_ub(n_constraints);
    for (size_t i = 0; i < n_constraints; ++i) {
        constraints_lb[i] = 0.;
        constraints_ub[i] = 0.;
    }

    // options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          5.0\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvec> solution;

    // solve the problem
    CppAD::ipopt::solve<Dvec, ConstrainedObjective>(
            options, z, z_lb, z_ub, constraints_lb,
            constraints_ub, *ptr_obj_, solution);

    if (solution.status != CppAD::ipopt::solve_result<Dvec>::success) {
        return solution.status;
    } else {
        // otherwise, get the solution
        // Cost
        // auto cost = solution.obj_value;
        // std::cout << "Cost " << cost << std::endl;

        for (size_t i = 0; i < n_vars; ++i) {
            sol[i] = solution.x[i];
        }
        return CppAD::ipopt::solve_result<Dvec>::success;
    }
};

Status registration(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config, arma::mat44 &) noexcept {
    if (source_pts.n_cols >= target_pts.n_cols) {
        std::cout << "no. of source pts must be less than no. of target pts" << std::endl;
        return Status::BadInput;
    }

    auto ptr_optmzr_obj = std::make_unique<PointRegRelaxation>(source_pts, target_pts, config);

    arma::colvec optimum( (source_pts.n_cols+1) * target_pts.n_cols );

    if (ptr_optmzr_obj->find_optimum(optimum) != CppAD::ipopt::solve_result<Dvec>::success) {
        return Status::SolverFailed;
    }
    
    //! find transformation
    return Status::Success;
    
}
