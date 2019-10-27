//! c/c++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
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
                        arma::vec3 const si = arma::vec3(source_pts.col(i));
                        arma::vec3 const tj = arma::vec3(target_pts.col(j));
                        arma::vec3 const sk = arma::vec3(source_pts.col(k));
                        arma::vec3 const tl = arma::vec3(target_pts.col(l));
                        double const c = consistency(si, tj, sk, tl);
                        if (c < eps && arma::norm(si - sk, 2) > pw_thresh) {
                            weight[std::make_tuple(i, j, k, l)] = -std::exp(-c);
                        }
                    }
                }
            }
        }
    }

    return weight;
}

/**
 * @brief Identify best transformation between two sets of points 
 *
 * @param [in] src_pts points to transform
 * @param [in] dst_pts target points
 * @param [in][out] H_optimal best-fit transformation to align points in homogeneous coordinates
 * @return
 *
 * @note src_pts and dst_pts must have the same number of columns
 */
void best_fit_transform(arma::mat src_pts, arma::mat dst_pts, arma::mat44 & H_optimal) noexcept {
    arma::mat33 optimal_rot;
    arma::vec3 optimal_trans;

    //! make const ref to size
    size_t const & n_pts = src_pts.n_cols;

    //! compute weighted centroids
    arma::vec3 const src_centroid = arma::vec3(arma::mean(src_pts, 1));
    arma::vec3 const dst_centroid = arma::vec3(arma::mean(dst_pts, 1));

    //! translate centroid to origin
    //! - Note the use of the decrement operator: this is why the function signature uses pass by value, not ref
    arma::mat const src_crep = arma::repmat(src_centroid, 1, n_pts);
    arma::mat const dst_crep = arma::repmat(dst_centroid, 1, n_pts);
    src_pts -= src_crep;
    dst_pts -= dst_crep;

    //! compute tensor product
    arma::mat33 const C = arma::mat33(src_pts * dst_pts.t());

    //! compute the singular value decomposition
    arma::mat33 U, V;
    arma::vec3 s;
    arma::svd(U, s, V, C);

    //! compute optimal rotation and translation
    arma::mat33 I(arma::fill::eye);
    if (arma::det(U * V.t()) < 0)
        I(2, 2) *= -1;
    optimal_rot = V * I * U.t();
    optimal_trans = dst_centroid - optimal_rot * src_centroid;

    H_optimal.zeros();
    H_optimal(arma::span(0, 2), arma::span(0, 2)) = optimal_rot;
    H_optimal(arma::span(0, 2), 3) = optimal_trans;
    H_optimal(3, 3) = 1;
    return;
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
                            //! remove the 1/2 from each term, since minimum is invariant under scaling
                            fgrad[curr_idx] += static_cast<CppAD::AD<double>>(weights_[key]) * (z[i*n_ + j] + 1.) * (z[k*n_ + l] + 1.);
                        }
                    }
                }
            }
        }
    }

    //! constraints (see paper):
    //! 1) Each source point should be mapped to a target point
    for (size_t i = 0; i < m_; ++i) {
        fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(n_) - 2.;
        for (size_t j = 0; j < n_; ++j) {
            fgrad[curr_idx] += z[i*n_ + j];
        }
    }

    //! 2) Two source points cannot be mapped to the same target point
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

void PointRegRelaxation::warm_start(PointRegRelaxation::Dvec & z) const noexcept {
    //! look at weights and assign initial associations as those that are most likely
    auto const & weights = ptr_obj_->get_weight_tensor();
    std::map<double, WeightKey_t> swapped_map;
    auto const & m = ptr_obj_->num_source_pts();
    auto const & n = ptr_obj_->num_target_pts();
    auto const & state_length = ptr_obj_->state_length();

    for (auto const & pair : weights) {
        auto const & key = pair.first;
        auto const & val = pair.second;
        swapped_map[val] = key;
    }

    std::vector<std::pair<size_t, size_t>> assocs;
    for (auto const & pair : swapped_map) {
        auto const & tup = pair.second;
        size_t const i = std::get<0>(tup);
        size_t const j = std::get<1>(tup);
        size_t const k = std::get<2>(tup);
        size_t const l = std::get<3>(tup);
        auto const assoc_ij = std::make_pair(i, j);
        auto const assoc_kl = std::make_pair(k, l);
        bool append_ij = true;
        bool append_kl = true;
        for (auto const & a : assocs) {
            if (a.first == assoc_ij.first || a.second == assoc_ij.second) {
                append_ij = false;
            }
            if (a.first == assoc_kl.first || a.second == assoc_kl.second) {
                append_kl = false;
            }
        }
        if (append_ij) {
            assocs.emplace_back(assoc_ij);
        }
        if (append_kl) {
            assocs.emplace_back(assoc_kl);
        }
        // if all associations have been made
        if (assocs.size() == m) {
            break;
        }
    }

    for (size_t i = 0; i < state_length; ++i)
        // when i < m+1, assume no association
        if (i < (m-1)*n + n) {
            z[i] = -1.0;
        } else { 
            z[i] = 1.0;
        }

    for (auto const & a : assocs) {
        z[a.first * n + a.second] = 1.0;
        z[m*n + a.second] = -1.0;
    }

    return;
}

PointRegRelaxation::status_t PointRegRelaxation::find_optimum() noexcept {
    auto const & n_vars = ptr_obj_->state_length();
    auto const & n_constraints = ptr_obj_->num_constraints();
    
    Dvec z(n_vars);
    if (config_.do_warm_start) { 
        warm_start(z);
    }

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
    options += "Integer print_level  0\n";
    /**
     * NOTE: Setting sparse to true allows the solver to take advantage
     * of sparse routines, this makes the computation MUCH FASTER. If you
     * can uncomment 1 of these and see if it makes a difference or not but
     * if you uncomment both the computation time should go up in orders of
     * magnitude.
     */
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    options += "Numeric tol         0.1\n";
    options += "Numeric acceptable_tol 0.1\n";

    //! timeout period (sec).
    options += "Numeric max_cpu_time          1000.0\n";

    //! solve the problem
    CppAD::ipopt::solve_result<Dvec> solution;
    CppAD::ipopt::solve<Dvec, ConstrainedObjective>(
            options, z, z_lb, z_ub, constraints_lb,
            constraints_ub, *ptr_obj_, solution);

    if (solution.status != status_t::success) {
        return solution.status;
    }

    for (size_t i = 0; i < n_vars; ++i) {
        optimum_(i) = solution.x[i];
    }
    return status_t::success;
};

void PointRegRelaxation::find_correspondences() noexcept {
    auto const & n = ptr_obj_->num_target_pts();
    auto const & n_vars = ptr_obj_->state_length();
    
    arma::colvec sol_noswap = optimum_(arma::span(0, n_vars - n - 1)); 

    arma::uvec const ids = arma::find(sol_noswap > config_.corr_threshold);
    std::list<size_t> corr_ids;
    for (size_t i = 0; i < ids.n_elem; ++i) {
        corr_ids.emplace_back(static_cast<size_t>(ids(i)));
    }

    for (auto const & c : corr_ids) {
        auto key = std::make_pair<size_t, size_t>(c / n, c % n);
        auto value = sol_noswap(c);
        correspondences_[key] = value;
    }
    return;
}

Status registration(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config, arma::colvec & optimum, PointRegRelaxation::correspondences_t & correspondences,
    arma::mat44 & H_optimal) noexcept {
    if (source_pts.n_cols >= target_pts.n_cols) {
        std::cout << "no. of source pts must be less than no. of target pts" << std::endl;
        return Status::BadInput;
    }
    auto ptr_optmzr_obj = std::make_unique<PointRegRelaxation>(source_pts, target_pts, config);

    if (ptr_optmzr_obj->find_optimum() != PointRegRelaxation::status_t::success) {
        return Status::SolverFailed;
    }
    ptr_optmzr_obj->find_correspondences();

    correspondences = ptr_optmzr_obj->get_correspondences();
    arma::uvec src_inds(correspondences.size()), tgt_inds(correspondences.size());
    size_t counter = 0;
    for (const auto & c : correspondences) {
        src_inds(counter) = c.first.first;
        tgt_inds(counter++) = c.first.second;
    }

    //! find transformation
    arma::mat const source_pts_align = source_pts.cols(src_inds);
    arma::mat const target_pts_align = target_pts.cols(tgt_inds);
    best_fit_transform(source_pts_align, target_pts_align, H_optimal);

    optimum = ptr_optmzr_obj->get_optimum();
    correspondences = ptr_optmzr_obj->get_correspondences();

    return Status::Success;
}
