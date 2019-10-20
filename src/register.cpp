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
    auto const & pw_thresh = config.pairwise_dist_threshold;
    size_t const & m = source_pts.n_cols;
    size_t const & n = target_pts.n_cols;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < m; ++k) {
                for (size_t l = 0; l < n; ++l) {
                    if (i != k && j != l) {
                        arma::vec3 si = arma::vec3(source_pts.col(i));
                        arma::vec3 tj = arma::vec3(target_pts.col(j));
                        arma::vec3 sk = arma::vec3(source_pts.col(k));
                        arma::vec3 tl = arma::vec3(target_pts.col(l));
                        /*double c = consistency(si, tj, sk, tl);
                        double d = arma::norm(si - sk, 2);*/
                        if (consistency(si, tj, sk, tl) < eps
                                && arma::norm(si - sk, 2) > pw_thresh) {
                            /*std::cout << "i: " << i << ", j: " << j << ", k: " << k << ", l: " << l << std::endl;
                            std::cout << "  c(i, j, k, l): " << c << std::endl;
                            std::cout << "  d(i, k): " << d << std::endl;*/
                            weight[std::make_tuple(i, j, k, l)] = -1.;
                        }
                    }
                }
            }
        }
    }
    //std::cout << "num nonzero weights: " << weight.size() << std::endl;
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
                        WeightKey_t const key = std::make_tuple(i, j, k, l);
                        if (weights_.find(key) != weights_.end()) {
                            fgrad[curr_idx] -= z[i*n_ + j] * z[k*n_ + l];
                        }
                    }
                }
            }
        }
    }

    //! constraints:
    //! for all i, \sum_j z_{ij} = 2 - n_ (there are m_ of these)
    for (size_t i = 0; i < m_; ++i) {
        fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(n_) - 2.;
        for (size_t j = 0; j < n_; ++j) {
            fgrad[curr_idx] += z[i*n_ + j];
        }
    }

    fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(2*m_) - static_cast<CppAD::AD<double>>(n_);
    for (size_t j = 0; j < n_; ++j) {
        fgrad[curr_idx] += z[m_*n_ + j];
    }

    for (size_t j = 0; j < n_; ++j) {
        fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(m_) - 1.;
        for (size_t i = 0; i <= m_; ++i) {
            fgrad[curr_idx] += z[i*n_ + j];
        }
    }
}

//! destructor (nothing to do)
PointRegRelaxation::~PointRegRelaxation() { }

PointRegRelaxation::status_t PointRegRelaxation::find_optimum(arma::colvec & sol) const noexcept {
    auto const & m = ptr_obj_->num_source_pts();
    auto const & n = ptr_obj_->num_target_pts();
    auto const & n_vars = ptr_obj_->state_length();
    auto const & n_constraints = ptr_obj_->num_constraints();
    sol.resize(n_vars);
    //std::cout << "m: " << m << ", n: " << n << std::endl;
    Dvec z(n_vars);
    // assumes index i in source set maps to index i in target set
    // could randomize this
    /*
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                z[i*n + j] = 1.;
            } else {
                z[i*n + j] = -1.;
            }
        }
    }

    for (size_t j = 0; j < n; ++j) {
        if (j < m) {
            z[m*n + j] = -1.;
        } else {
            z[m*n + j] = 1.;
        }
    }

    ConstrainedObjective::ADvector _z(n_vars);
    for (size_t i = 0; i < n_vars; ++i) {
        _z[i] = z[i];
    }
    */

    for (size_t i = 0; i <= m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            z[i*n + j] = 0.;
        }
    }
    /*
    //! DEBUG
    ConstrainedObjective::ADvector f(n_constraints+1);
    (*ptr_obj_)(f, _z);
    for (size_t i = 0; i < n_constraints; ++i) {
        std::cout << "f[" << i << "]: " << f[i] << std::endl;
    }
    //! end DEBUG
    */

    Dvec z_lb(n_vars);
    Dvec z_ub(n_vars);
    for (size_t i = 0; i < n_vars; ++i) {
        z_lb[i] = -1.;
        z_ub[i] = 1.;
    }

    Dvec constraints_lb(n_constraints);
    Dvec constraints_ub(n_constraints);
    for (size_t i = 0; i < n_constraints; ++i) {
        constraints_lb[i] = 0.;
        constraints_ub[i] = 0.;
    }

    //! options for IPOPT solver
    std::string options;
    //options += "Integer print_level  5\n";
    options += "Integer print_level  0\n";
    /**
     * NOTE: Setting sparse to true allows the solver to take advantage
     * of sparse routines, this makes the computation MUCH FASTER. If you
     * can uncomment 1 of these and see if it makes a difference or not but
     * if you uncomment both the computation time should go up in orders of
     * magnitude.
     */
    //options += "Integer max_iter     1\n";
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    //options += "String  mu_strategy adaptive\n";
    //! timeout period (sec).
    options += "Numeric max_cpu_time          1000.0\n";

    //! solve the problem
    CppAD::ipopt::solve_result<Dvec> solution;
    CppAD::ipopt::solve<Dvec, ConstrainedObjective>(
            options, z, z_lb, z_ub, constraints_lb,
            constraints_ub, *ptr_obj_, solution);

    for (size_t i = 0; i < n_vars; ++i) {
        sol(i) = solution.x[i];
    }

    if (solution.status != status_t::success) {
        return solution.status;
    }
    //! solution is valid
    //! DEBUG - Cost
    //! auto cost = solution.obj_value;

    for (size_t i = 0; i < n_vars; ++i) {
        sol(i) = solution.x[i];
    }

    return status_t::success;
};

Status registration(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config, arma::mat44 &) noexcept {
    if (source_pts.n_cols >= target_pts.n_cols) {
        std::cout << "no. of source pts must be less than no. of target pts" << std::endl;
        return Status::BadInput;
    }

    auto ptr_optmzr_obj = std::make_unique<PointRegRelaxation>(source_pts, target_pts, config);

    arma::colvec optimum( (source_pts.n_cols+1) * target_pts.n_cols );

    if (ptr_optmzr_obj->find_optimum(optimum) != PointRegRelaxation::status_t::success) {
        return Status::SolverFailed;
    }
    
    //! find transformation
    return Status::Success;
    
}
