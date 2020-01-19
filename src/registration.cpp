//! c/c++ headers
#include <cmath>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
//! dependency headers
//! project headers
#include "registration/registration.hpp"

namespace cor = correspondences;
namespace reg = registration;

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
void reg::best_fit_transform(arma::mat src_pts, arma::mat dst_pts, arma::mat44 & H_optimal) noexcept {
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
    best_fit_transform(source_pts_align, target_pts_align, H_optimal);

    optimum = ptr_optmzr_obj->get_optimum();
    correspondences = ptr_optmzr_obj->get_correspondences();

    return Status::Success;
}
