#pragma once
//! c/c++ headers
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
//! dependency headers
#include <armadillo>  // NOLINT [build/include_order]
#include <ensmallen.hpp>

namespace correspondences_sdp {
/** @struct Config
 * @brief configuration parameters for optimization algorithm 
 * @var Config::epsilon
 * threshold below which to declare a pairwise correspondence
 * @var Config::pairwise_dist_threshold
 * threshold above which to allow pairwise correspondence; this prevents considering correspondences
 * that may have high ambiguity
 * @var Config::corr_threshold
 * value in optimum solution to declare a correspondence as valid {\in (0, 1)}
 * @var Config::n_pair_threshold
 * minimum number of pairwise consistencies 
 */
struct Config {
    Config() :
        epsilon(1e-1), pairwise_dist_threshold(1e-1),
        corr_threshold(0.9), n_pair_threshold(std::numeric_limits<size_t>::signaling_NaN())
   { }
    double epsilon, pairwise_dist_threshold, corr_threshold;
    size_t n_pair_threshold;
};

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
        arma::vec3 const & tl) noexcept;

/** @class ConstrainedObjective
 * @brief class definition for point set registration relaxation objective
 */
class ConstrainedObjective {
 public:
     /** ConstrainedObjective::ConstrainedObjective(source_pts, target_pts, config)
      * @brief constructor for constrained objective function
      *
      * @param[in] source_pts distribution of (columnar) source points
      * @param[in] target_pts distribution of (columnar) target points
      * @param[in] config `Config` instance with optimization parameters; see `Config` definition
      * @param[in] min_corr minimum number of correspondences required
      * @return
      */
     ConstrainedObjective(arma::mat const & source_pts,
             arma::mat const & target_pts, Config const & config, size_t const & min_corr);

     /** ConstrainedObjective::~ConstrainedObjective()
      * @brief destructor for constrained objective function
      *
      * @param[in]
      * @return
      */
     ~ConstrainedObjective();

     /** ConstrainedObjective::num_constraints()
      * @brief return number of constraints for optimization objective
      *
      * @param[in]
      * @return copy of private member `n_constraints_`
      */
     size_t const num_constraints() const noexcept { return n_constraints_; }
 
     /** ConstrainedObjective::num_source_pts()
      * @brief return number of source points for optimization objective
      *
      * @param[in]
      * @return copy of private member `m_`
      */
     size_t const num_source_pts() const noexcept { return m_; }

     /** ConstrainedObjective::num_target_pts()
      * @brief return number of target points for optimization objective
      *
      * @param[in]
      * @return copy of private member `n_`
      */
     size_t const num_target_pts() const noexcept { return n_; }

     /** ConstrainedObjective::num_min_corr()
      * @brief return minimum number of correspondences for optimization objective
      *
      * @param[in]
      * @return copy of private member `min_corr_`
      */
     size_t const num_min_corr() const noexcept { return min_corr_; }

     /** ConstrainedObjective::state_length()
      * @brief get size of state vector for optimization objective
      *
      * @param[in]
      * @return (m_ + 1)*(n_ + 1)
      * @note includes slack variables
      */
     size_t const state_length() const noexcept { return (m_ + 1) * (n_ + 1); }

     /** ConstrainedObjective::get_sdp()
      * @brief get weight_tensor
      *
      * @param[in]
      */
     ens::SDP<arma::sp_mat> const get_sdp() const noexcept { return sdp_; }

     void find_optimum() const noexcept;

 private:
     const size_t m_, n_, min_corr_, n_constraints_;
     ens::SDP<arma::sp_mat> sdp_;
     /**
      * @brief populate weight tensor for optimization problem; see `w` in Eqn. 48 from paper
      *
      * @param[in] source_pts distribution of (columnar) source points
      * @param[in] target_pts distribution of (columnar) target points
      * @param[in] config `Config` instance with optimization parameters; see `Config` definition
      * @return  weight tensor with weights for pairwise correspondences in optimization objective
      */
     void generate_weight_tensor(arma::mat const & source_pts, arma::mat const & target_pts,
         Config const & config) noexcept;
};
}  // namespace correspondences
