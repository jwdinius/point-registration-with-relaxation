#pragma once
#include <armadillo>

#include "correspondences/correspondences.hpp"

namespace registration {
/** @enum Status
 * @brief optimizer termination status definitions
 * @var Status::BadInput
 * termination criteria when input is inconsistent with functional requirements; see `registration` definition
 * @var Status::SolverFailed
 * termination criteria when IPOPT failed to converge (for any reason) to an optimum
 * @var Status::Success
 * termination criteria when full registration pipeline executed successfully
 */
enum class Status {
    BadInput,
    SolverFailed,
    Success
};

/**
 * @brief run full registration pipeline 
 *
 * @param [in] src_pts points to transform
 * @param [in] dst_pts target points
 * @param [in] config `Config` instance with optimization parameters; see `Config` definition
 * @param [in] min_corr minimum number of correspondences to identify between source and target sets
 * @param [in][out] optimum optimum for constrained objective identified by IPOPT
 * @param [in][out] correspondences pairwise correspondences identified between source and target point sets
 * @param [in][out] H_optimal best-fit transformation to align points in homogeneous coordinates
 * @return solver status; see `Status` enum definition
 *
 * @note src_pts and dst_pts must have the same number of columns
 * @note see original reference, available at https://www.sciencedirect.com/science/article/pii/S1077314208000891
 */
Status registration(arma::mat const & source_pts, arma::mat const & target_pts,
    correspondences::Config const & config, size_t const & min_corr, arma::colvec & optimum,
    correspondences::PointRegRelaxation::correspondences_t & correspondences,
    arma::mat44 & H_optimal) noexcept;

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
void best_fit_transform(arma::mat src_pts, arma::mat dst_pts, arma::mat44 & H_optimal) noexcept;
};
