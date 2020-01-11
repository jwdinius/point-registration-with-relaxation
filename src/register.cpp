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

//! forward declarations to minimize compile time
class ConstrainedObjective;
class PointRegRelaxation;

/**
 * @brief compute pairwise consistency score; see Eqn. 49 from reference
 *
 * @param [in] si ith point in source distribution
 * @param [in] tj jth point in target distribution
 * @param [in] sk kth point in source distribution
 * @param [in] tl lth point in target distribution
 * @return pairwise consistency score for (i, j, k, l)
 */
double consistency(arma::vec3 const & si, arma::vec3 const & tj, arma::vec3 const & sk,
        arma::vec3 const & tl) noexcept {
    double const dist_si_to_sk = arma::norm(si - sk, 2);
    double const dist_tj_to_tl = arma::norm(tj - tl, 2);
    return std::abs(dist_si_to_sk - dist_tj_to_tl);
}

/**
 * @brief generate weight tensor for optimation objective; see Eqn. 48 from reference
 *
 * @param [in] src_pts points to transform
 * @param [in] dst_pts target points
 * @param [in] config `Config` instance with optimization parameters; see `Config` definition
 * @return weight tensor for optimization objective; `w_{ijkl}` from reference
 */
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
 * @brief Identify best transformation between two sets of points with known correspondences 
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

/** ConstrainedObjective::~ConstrainedObjective()
 * @brief destructor for constrained objective function
 *
 * @param[in]
 * @return
 *
 * @note nothing to do; resources are automatically deallocated
 */
ConstrainedObjective::~ConstrainedObjective() { }

/** ConstrainedObjective::operator()
 * @brief operator overload for IPOPT
 *
 * @param[in][out] fgrad objective function evaluation (including constraints) at point `z`
 * @param[in] z point for evaluation
 * return
 */
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
                            fgrad[curr_idx] += static_cast<CppAD::AD<double>>(weights_[key]) * z[i*n_ + j] * z[k*n_ + l];
                        }
                    }
                }
            }
        }
    }

    //! constraints (see paper):
    //! 1) Each source point should be mapped to a target point
    for (size_t i = 0; i < m_; ++i) {
        fgrad[++curr_idx] = -1.;
        for (size_t j = 0; j < n_; ++j) {
            fgrad[curr_idx] += z[i*n_ + j];
        }
    }

    //! 2) Two source points cannot be mapped to the same target point
    fgrad[++curr_idx] = static_cast<CppAD::AD<double>>(m_) - static_cast<CppAD::AD<double>>(n_);
    for (size_t j = 0; j < n_; ++j) {
        fgrad[curr_idx] += z[m_*n_ + j];
    }

    for (size_t j = 0; j < n_; ++j) {
        fgrad[++curr_idx] = -1.;
        for (size_t i = 0; i <= m_; ++i) {
            fgrad[curr_idx] += z[i*n_ + j];
        }
    }
}

/** PointRegRelaxation::~PointRegRelaxation()
 * @brief destructor for optimization wrapper class
 *
 * @param[in]
 * @return
 *
 * @note nothing to do; resources are automatically deallocated
 */
PointRegRelaxation::~PointRegRelaxation() { }

/** PointRegRelaxation::warm_start
 * @brief compute feasible initial starting point for optimization based on correspondence weights
 *
 * @param[in][out] initial guess for optimization algorithm
 * return
 *
 * @note see implementation for algorithm details
 */
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

    //! choose best matches based upon correspondence scores
    //! NOTE: this means using the reverse iterator because std::map
    //! objects are sorted smallest-to-largest
    std::vector<std::pair<size_t, size_t>> assocs;
    for (auto cit = swapped_map.crbegin(); cit != swapped_map.crend(); ++cit) {
        auto const & tup = cit->second;
        size_t const i = std::get<0>(tup);
        size_t const j = std::get<1>(tup);
        size_t const k = std::get<2>(tup);
        size_t const l = std::get<3>(tup);
        //! try to associate both i->j AND k->l, but only if a "better" correspondence hasn't yet
        //! been found
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
        //! if a correspondence for i, j, k, or l have not been found, add the correspondences
        //! i->j and/or k->l
        if (append_ij) {
            assocs.emplace_back(assoc_ij);
        }
        if (append_kl) {
            assocs.emplace_back(assoc_kl);
        }
        //! if all associations have been made, exit the loop
        if (assocs.size() == m) {
            break;
        }
    }
    //! initialize values with index in non-slack variables to -1.0 (no correspondence)
    //! and slack-variables to 1.0 (again, no correspondence)
    for (size_t i = 0; i < state_length; ++i)
        // when i < m+1, assume no association
        if (i < (m-1)*n + n) {
            z[i] = 0.0;
        } else { 
            z[i] = 1.0;
        }
    //! overwrite values at indices associated with best correspondence score
    //! NOTE: this obeys equality constraints
    for (auto const & a : assocs) {
        z[a.first * n + a.second] = 1.0;
        z[m*n + a.second] = 0.0;
    }

    return;
}

/** PointRegRelaxation::find_optimum()
 * @brief run IPOPT to find optimum for optimization objective:
 * argmin f(z) subject to constraints g_i(z) == 0, 0 <= i < num_constraints
 *
 * @param[in]
 * @return
 *
 * @note see Eqn. 48 in paper and subsequent section for details
 */
PointRegRelaxation::status_t PointRegRelaxation::find_optimum() noexcept {
    auto const & n_vars = ptr_obj_->state_length();
    auto const & n_constraints = ptr_obj_->num_constraints();

    //! do warm start, if configured to do so
    Dvec z(n_vars);
    if (config_.do_warm_start) { 
        warm_start(z);
    }

    //! setup inequality constraints on variables: 0 <= z_{ij} <= 1 for all i,j
    Dvec z_lb(n_vars);
    Dvec z_ub(n_vars);
    for (size_t i = 0; i < n_vars; ++i) {
        z_lb[i] = 0.;
        z_ub[i] = 1.;
    }

    //! setup equality constraints g_i(z) == 0
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
    //! if solver was not successful, return early
    if (solution.status != status_t::success) {
        return solution.status;
    }
    //! overwrite private member optimum_ with result and
    //! return success
    for (size_t i = 0; i < n_vars; ++i) {
        optimum_(i) = solution.x[i];
    }
    return status_t::success;
};

/** PointRegRelaxation::find_correspondences()
 * @brief identify pairwise correspondences between source and target set given optimum found
 * during optimization
 *
 * @param[in]
 * @return
 */
void PointRegRelaxation::find_correspondences() noexcept {
    auto const & n = ptr_obj_->num_target_pts();
    auto const & n_vars = ptr_obj_->state_length();

    //! remove slack variables
    arma::colvec sol_noslack = optimum_(arma::span(0, n_vars - n - 1)); 
    //! find indices of valid correspondences i->j
    arma::uvec const ids = arma::find(sol_noslack > config_.corr_threshold);
    std::list<size_t> corr_ids;
    for (size_t i = 0; i < ids.n_elem; ++i) {
        corr_ids.emplace_back(static_cast<size_t>(ids(i)));
    }
    //! i = divide(id, n), j = remainder(id, n) for correspondence i->j
    //! value is the strength of the correspondence (-1 <= z_{ij} <= 1, closer to 1
    //! is better)
    for (auto const & c : corr_ids) {
        auto key = std::make_pair<size_t, size_t>(c / n, c % n);
        auto value = sol_noslack(c);
        correspondences_[key] = value;
    }
    return;
}

/**
 * @brief run full registration pipeline 
 *
 * @param [in] src_pts points to transform
 * @param [in] dst_pts target points
 * @param [in] config `Config` instance with optimization parameters; see `Config` definition
 * @param [in][out] optimum optimum for constrained objective identified by IPOPT
 * @param [in][out] correspondences pairwise correspondences identified between source and target point sets
 * @param [in][out] H_optimal best-fit transformation to align points in homogeneous coordinates
 * @return solver status; see `Status` enum definition
 *
 * @note this is the main functional interface to this library
 * @note src_pts and dst_pts must have the same number of columns
 * @note see original reference, available at https://www.sciencedirect.com/science/article/pii/S1077314208000891
 */
Status registration(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config, size_t const & min_corr, arma::colvec & optimum,
    PointRegRelaxation::correspondences_t & correspondences, arma::mat44 & H_optimal) noexcept {
    if (source_pts.n_cols >= target_pts.n_cols) {
        std::cout << "no. of source pts must be less than no. of target pts" << std::endl;
        return Status::BadInput;
    } else if (min_corr > source_pts.n_cols) {
        std::cout << "no. of minimum correspondences must be less than or equal to no. of source pts" << std::endl;
        return Status::BadInput;
    }
    auto ptr_optmzr_obj = std::make_unique<PointRegRelaxation>(source_pts, target_pts, config, min_corr);

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
