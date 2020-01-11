#pragma once
#include <cmath>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <armadillo>
#include <boost/functional/hash.hpp>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>

/** @struct Config
 * @brief configuration parameters for optimization algorithm 
 * @var Config::epsilon
 * threshold below which to declare a pairwise correspondence
 * @var Config::pairwise_dist_threshold
 * threshold above which to allow pairwise correspondence; this prevents considering correspondences
 * that may have high ambiguity
 * @var Config::corr_threshold
 * value in optimum solution to declare a correspondence as valid {\in (-1, 1)}
 * @var Config::do_warm_start
 * flag indicating whether or not to do warm start before optimization
 */
struct Config {
    Config() :
        epsilon(1e-1), pairwise_dist_threshold(1e-1),
        corr_threshold(0.5), do_warm_start(true) { }
    double epsilon, pairwise_dist_threshold, corr_threshold;
    bool do_warm_start;
};

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

/** @typedef WeightKey_t
 * @brief key definition for weight associative container (an unordered map)
 * (i, j, k, l): i, k are indices for points in source distribution, and j, l
 * are points in target distribution
 */
using WeightKey_t = std::tuple<size_t, size_t, size_t, size_t>;

/** @struct key_hash : public std::unary_function<WeightKey_t, size_t>
 * @brief hash function for unordered_map with WeightKey_t as key
 * see https://stackoverflow.com/questions/3611951/building-an-unordered-map-with-tuples-as-keys
 */
struct key_hash : public std::unary_function<WeightKey_t, size_t> {
    /** key_hash::operator()
     * @brief operator overload for ()
     *
     * @param[in] k key to obtain hash for
     * @return seed hash value for key k
     */
    size_t operator()(const WeightKey_t& k) const {
        size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(k));
        boost::hash_combine(seed, std::get<1>(k));
        boost::hash_combine(seed, std::get<2>(k));
        boost::hash_combine(seed, std::get<3>(k));
        return seed;
    }
};

/** @struct key_equal : public std::binary_function<WeightKey_t, WeightKey_t, bool>
 * @brief check if two WeightKey_t instances are equal
 */
struct key_equal : public std::binary_function<WeightKey_t, WeightKey_t, bool> {
    /** key_hash::operator()
     * @brief operator overload for ()
     *
     * @param[in] v0 first key for equality check
     * @param[in] v1 second key for equality check
     * @return flag indicating whether two keys are equal (=true) or not (=false)
     */
   bool operator()(const WeightKey_t& v0, const WeightKey_t& v1) const {
      return std::get<0>(v0) == std::get<0>(v1) &&
             std::get<1>(v0) == std::get<1>(v1) &&
             std::get<2>(v0) == std::get<2>(v1) &&
             std::get<3>(v0) == std::get<3>(v1);
   }
};

/** @typedef WeightTensor
 * @brief unordered map for storing weights associated with pairwise correspondences
 */
using WeightTensor = std::unordered_map<WeightKey_t, double, key_hash, key_equal>;

/**
 * @brief populate weight tensor for optimization problem; see `w` in Eqn. 48 from paper
 *
 * @param[in] source_pts distribution of (columnar) source points
 * @param[in] target_pts distribution of (columnar) target points
 * @param[in] config `Config` instance with optimization parameters; see `Config` definition
 * @return  weight tensor with weights for pairwise correspondences in optimization objective
 */
WeightTensor generate_weight_tensor(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config) noexcept;

/** @class ConstrainedObjective
 * @brief class definition for point set registration relaxation objective
 */
class ConstrainedObjective {
 public:
     /** @typedef ConstrainedObjective::ADVector
      * @brief typedef for CppAD automatic differentiation during optimization execution
      */
     using ADvector = CPPAD_TESTVECTOR(CppAD::AD<double>);

     /** ConstrainedObjective::ConstrainedObjective(source_pts, target_pts, config)
      * @brief constructor for constrained objective function
      *
      * @param[in] source_pts distribution of (columnar) source points
      * @param[in] target_pts distribution of (columnar) target points
      * @param[in] config `Config` instance with optimization parameters; see `Config` definition
      * @param[in] min_corr minimum number of corresondences required
      * @return
      */
     explicit ConstrainedObjective(arma::mat const & source_pts,
             arma::mat const & target_pts, Config const & config, size_t const & min_corr) :
         m_(static_cast<size_t>(source_pts.n_cols)), n_(static_cast<size_t>(target_pts.n_cols)),
         min_corr_(min_corr) {
         weights_ = generate_weight_tensor(source_pts, target_pts, config);
         n_constraints_ = m_ + n_ + 1;
     }

     /** ConstrainedObjective::~ConstrainedObjective()
      * @brief destructor for constrained objective function
      *
      * @param[in]
      * @return
      */
     ~ConstrainedObjective();
     
     /** ConstrainedObjective::operator()
      * @brief operator overload for IPOPT
      *
      * @param[in][out] fgrad objective function evaluation (including constraints) at point `z`
      * @param[in] z point for evaluation
      * @return
      */
     void operator()(ADvector &fgrad, ADvector const & z) noexcept;

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
     
     size_t const num_min_corr() const noexcept { return min_corr_; }
     
     /** ConstrainedObjective::state_length()
      * @brief get size of state vector for optimization objective
      *
      * @param[in]
      * @return (m_ + 1)*n_
      * @note includes slack variables
      */
     size_t const state_length() const noexcept { return (m_ + 1) * n_; }

     /** ConstrainedObjective::get_weight_tensor()
      * @brief get weight_tensor
      *
      * @param[in]
      * @return copy of private member `weights_`
      */
     WeightTensor const get_weight_tensor() const noexcept { return weights_; }

 private:
     WeightTensor weights_;
     size_t m_, n_, min_corr_, n_constraints_;
};

/** @class PointRegRelaxation
 * @brief wrapper class definition for point set registration using convex relaxation
 */
class PointRegRelaxation {
 public:
     /** @typedef PointRegRelaxation::DVec
      * @brief typedef for CppAD test vector
      */
     using Dvec = CPPAD_TESTVECTOR(double);
     
     /** @typedef PointRegRelaxation::status_t
      * @brief typedef for IPOPT return code
      */
     using status_t = CppAD::ipopt::solve_result<Dvec>::status_type;

     /** @struct PointRegRelaxation::key_lthan : public std::binary_function<pair<size_t, size_t>, pair<size_t, size_t>, bool>
      * @brief less than comparator for correspondence map
      */
     struct key_lthan : public std::binary_function<std::pair<size_t, size_t>, std::pair<size_t, size_t>, bool> {
         /** key_lthan::operator()
          * @brief operator overload for ()
          *
          * @param[in] left left-hand key for "<" comparison
          * @param[in] right right-hand key for "<" comparison
          * @return left < right
          *
          * @note assumes row-major ordering:  if two keys have the same row index, column index is used for comparison, otherwise
          * row index is used for comparison
          */
         bool operator()(const std::pair<size_t, size_t> & left, const std::pair<size_t, size_t> & right) const {
             if (left.first == right.first) {
                 return left.second < right.second;
             } else {
                 return left.first < right.first;
             }
         }
     };

     /** @typedef PointRegRelaxation::correspondences_t
      * @brief typedef for IPOPT return code
      *
      * @note stores the strength of the correspondence as the value in the map container
      */
     using correspondences_t = std::map<std::pair<size_t, size_t>, double, key_lthan>;

     /** PointRegRelaxation::PointRegRelaxation(source_pts, target_pts, config)
      * @brief constructor for optimization wrapper class
      *
      * @param[in] source_pts distribution of (columnar) source points
      * @param[in] target_pts distribution of (columnar) target points
      * @param[in] config `Config` instance with optimization parameters; see `Config` definition
      * @return
      */
     explicit PointRegRelaxation(arma::mat const & source_pts,
             arma::mat const & target_pts, Config const & config,
             size_t const & min_corr) : config_(config) {
         ptr_obj_ = std::make_unique<ConstrainedObjective>(source_pts, target_pts, config, min_corr);
         optimum_.resize(ptr_obj_->state_length());
     }
     
     /** PointRegRelaxation::~PointRegRelaxation()
      * @brief destructor for optimization wrapper class
      *
      * @param[in]
      * @return
      */
     ~PointRegRelaxation();

     /** PointRegRelaxation::warm_start
      * @brief compute feasible initial starting point for optimization based on correspondence weights
      *
      * @param[in][out] initial guess for optimization algorithm
      * @return
      */
     void warm_start(Dvec & z_init) const noexcept;

     /** PointRegRelaxation::find_optimum()
      * @brief run IPOPT to find optimum for optimization objective
      *
      * @param[in]
      * @return
      */
     status_t find_optimum() noexcept;

     /** PointRegRelaxation::find_correspondences()
      * @brief identify pairwise correspondences between source and target set given optimum found
      * during optimization
      *
      * @param[in]
      * @return
      */
     void find_correspondences() noexcept;

     /** PointRegRelaxation::get_optimum()
      * @brief get optimization
      *
      * @param[in]
      * @return
      */
     arma::colvec const get_optimum() const noexcept { return optimum_; }

     /** PointRegRelaxation::get_correspondences()
      * @brief get identified correspondences
      *
      * @param[in]
      * @return copy of correspondences_
      */
     correspondences_t const get_correspondences() const noexcept { return correspondences_; }

 private:
     Config config_;
     std::unique_ptr<ConstrainedObjective> ptr_obj_;
     correspondences_t correspondences_;
     arma::colvec optimum_; 
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
    Config const & config, size_t const & min_corr, arma::colvec & optimum,
    PointRegRelaxation::correspondences_t & correspondences, arma::mat44 & H_optimal) noexcept;

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
