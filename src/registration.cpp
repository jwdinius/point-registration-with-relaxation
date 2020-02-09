//! c/c++ headers
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
//! dependency headers
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
//! project headers
#include "registration/registration.hpp"

namespace cor = correspondences;
namespace reg = registration;
namespace mn = mlpack::neighbor;

namespace registration {
using KDTreeSearcher = mn::NeighborSearch<mn::NearestNeighborSort,
      mlpack::metric::EuclideanDistance,
      arma::mat,
      mlpack::tree::KDTree>;
};

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
bool reg::best_fit_transform(arma::mat src_pts, arma::mat dst_pts, arma::mat44 & H_optimal) noexcept {
    //! input checking
    if (src_pts.n_rows != 3) {
        std::cout << static_cast<std::string>(__func__) << ": First argument must be a matrix with 3 rows" << std::endl;
        return false;
    } else if (dst_pts.n_rows != 3) {
        std::cout << static_cast<std::string>(__func__) << ": Second argument must be a matrix with 3 rows" << std::endl;
        return false;
    }

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
    return true;
}

/**
 * @brief Iterative closest point algorithm: Perform point-set alignment on two sets of points with outlier rejection.
 *
 * @param [in] src_pts points to transform
 * @param [in] dst_pts target points
 * @param [in] H_init initial guess for best-fit homogeneous transformation
 * @param [in] max_its maximum number of iterations
 * @param [in] tolerance criteria for convergence, in terms of mean error between iterations
 * @param [in] reject_ratio ratio of worst-matches to reject in fit
 * @param [in][out] H_optimal best-fit transformation to align points in homogeneous coordinates
 * @return
 */
bool reg::iterative_closest_point(arma::mat const & src_pts, arma::mat const & dst_pts,
        arma::mat44 & H_init, size_t const & max_its, double const & tolerance, double const & reject_ratio,
        arma::mat44 & H_optimal) noexcept {
    //! input checking
    if (src_pts.n_rows != 3) {
        std::cout << static_cast<std::string>(__func__) << ": First argument must be a matrix with 3 rows" << std::endl;
        return false;
    } else if (dst_pts.n_rows != 3) {
        std::cout << static_cast<std::string>(__func__) << ": Second argument must be a matrix with 3 rows" << std::endl;
        return false;
    } else if (tolerance < std::numeric_limits<double>::epsilon()) {
        std::cout << static_cast<std::string>(__func__) << ": Fifth argument must be a positive scalar" << std::endl;
        return false;
    } else if (reject_ratio < std::numeric_limits<double>::epsilon() ||
            reject_ratio > static_cast<double>(1) - std::numeric_limits<double>::epsilon()) {
        std::cout << static_cast<std::string>(__func__) <<
            ": Sixth argument must be a positive scalar inside the interval (0, 1), non-inclusive" << std::endl;
        return false;
    }

    //! transform src points by initial homogeneous transformation
    size_t const & src_npts = src_pts.n_cols;
    arma::mat ones(1, src_npts, arma::fill::ones);
    arma::mat aug_src_pts = arma::join_vert(src_pts, ones);
    arma::mat src_xform_full = H_init * aug_src_pts;
    arma::mat src_xform = src_xform_full.rows(0, 2);

    //! identify first index to start discarding from sorted index list
    size_t const reject_idx = std::round( (static_cast<double>(1) - reject_ratio) * src_npts );

    //! setup nearest neighbor search
    KDTreeSearcher dst_searcher(dst_pts);
    arma::Mat<size_t> neighbors;
    arma::mat distances;

    //! loop until converged
    double error = 0;
    size_t counter = 0;
    while (counter++ < max_its) {
        //! find nearest neighbors and distances - neighbors come from searcher
        dst_searcher.Search(src_xform, 1, neighbors, distances);

        //! throw away worst matches
        arma::uvec const best_to_worst_idx = arma::sort_index(distances, "ascend");
        arma::uvec const nbr_idx = arma::conv_to<arma::uvec>::from(neighbors.row(0));
        arma::uvec const sorted_nbr_idx = nbr_idx.elem(best_to_worst_idx);
        arma::uvec const btw_idx_src_rej = best_to_worst_idx( arma::span(0, reject_idx-1) );
        arma::uvec const btw_idx_dst_rej = sorted_nbr_idx( arma::span(0, reject_idx-1) );

        //! compute mean error and check for convergence
        double const mean_error = arma::mean( distances.elem(btw_idx_src_rej) );

        //! if converged, break out of the loop
        if (std::abs(error - mean_error) < tolerance) {
            break;
        }

        //! reset error for next pass
        error = mean_error;

        //! compute best transformation between the current src and nearest dst points
        //! having rejected worst matches
        static_cast<void>( best_fit_transform(src_xform.cols( btw_idx_src_rej ), dst_pts.cols( btw_idx_dst_rej ),
                H_optimal) );

        //! transform src points by current best transformation
        aug_src_pts = arma::join_vert(src_xform, ones);
        src_xform_full = H_optimal * aug_src_pts;
        src_xform = src_xform_full.rows(0, 2);
    }

    if (counter > max_its)
        //! algorithm didn't converge
        return false;

    //! find best transform from source to icp transformed points
    return best_fit_transform(src_pts, src_xform, H_optimal);
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
 * @note src_pts must have less than or equal to the number of dst_pts (src cols <= dst cols)
 * @note see original reference, available at https://www.sciencedirect.com/science/article/pii/S1077314208000891
 */
reg::Status reg::registration(arma::mat const & source_pts, arma::mat const & target_pts,
    cor::Config const & config, size_t const & min_corr, arma::colvec & optimum,
    cor::PointRegRelaxation::correspondences_t & correspondences,
    arma::mat44 & H_optimal) noexcept {
    if (source_pts.n_cols > target_pts.n_cols) {
        std::cout << "no. of source pts must be AT MOST the no. of target pts" << std::endl;
        return Status::BadInput;
    } else if (min_corr > source_pts.n_cols) {
        std::cout << "no. of minimum correspondences must be less than or equal to no. of source pts" << std::endl;
        return Status::BadInput;
    }
    auto ptr_optmzr_obj = std::make_unique<cor::PointRegRelaxation>(source_pts, target_pts,
            config, min_corr);

    if (ptr_optmzr_obj->num_consistent_pairs() < config.n_pair_threshold) {
        return Status::TooFewPairs;
    }

    if (ptr_optmzr_obj->find_optimum() != cor::PointRegRelaxation::status_t::success) {
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
    static_cast<void>( best_fit_transform(source_pts_align, target_pts_align, H_optimal) );

    optimum = ptr_optmzr_obj->get_optimum();
    correspondences = ptr_optmzr_obj->get_correspondences();

    return Status::Success;
}
