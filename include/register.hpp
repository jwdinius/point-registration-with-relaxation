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

struct Config {
    Config() :
        epsilon(1e-1), pairwise_dist_threshold(1e-1),
        corr_threshold(0.5), do_warm_start(true) { }
    double epsilon, pairwise_dist_threshold, corr_threshold;
    bool do_warm_start;
};

enum class Status {
    BadInput,
    SolverFailed,
    Success
};

using WeightKey_t = std::tuple<size_t, size_t, size_t, size_t>;

struct key_hash : public std::unary_function<WeightKey_t, size_t> {
    size_t operator()(const WeightKey_t& k) const {
        size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(k));
        boost::hash_combine(seed, std::get<1>(k));
        boost::hash_combine(seed, std::get<2>(k));
        boost::hash_combine(seed, std::get<3>(k));
        return seed;
    }
};

struct key_equal : public std::binary_function<WeightKey_t, WeightKey_t, bool> {
   bool operator()(const WeightKey_t& v0, const WeightKey_t& v1) const {
      return std::get<0>(v0) == std::get<0>(v1) &&
             std::get<1>(v0) == std::get<1>(v1) &&
             std::get<2>(v0) == std::get<2>(v1) &&
             std::get<3>(v0) == std::get<3>(v1);
   }
};

using WeightTensor = std::unordered_map<WeightKey_t, double, key_hash, key_equal>;

WeightTensor generate_weight_tensor(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config) noexcept;

class ConstrainedObjective {
 public:
     using ADvector = CPPAD_TESTVECTOR(CppAD::AD<double>);

     //! constructor
     explicit ConstrainedObjective(arma::mat const & source_pts,
             arma::mat const & target_pts, Config const & config) :
         m_(static_cast<size_t>(source_pts.n_cols)), n_(static_cast<size_t>(target_pts.n_cols)) {
         weights_ = generate_weight_tensor(source_pts, target_pts, config);
         n_constraints_ = m_ + n_ + 1;
     }

     ~ConstrainedObjective();
     
     void operator()(ADvector &fgrad, ADvector const & z) noexcept;

     size_t const num_constraints() const noexcept { return n_constraints_; }
     size_t const num_source_pts() const noexcept { return m_; }
     size_t const num_target_pts() const noexcept { return n_; }
     size_t const state_length() const noexcept { return (m_ + 1) * n_; }
     WeightTensor const get_weight_tensor() const noexcept { return weights_; }

 private:
     WeightTensor weights_;
     size_t m_, n_, n_constraints_;
};

class PointRegRelaxation {
 public:
     using Dvec = CPPAD_TESTVECTOR(double);
     using status_t = CppAD::ipopt::solve_result<Dvec>::status_type;

     struct key_lthan : public std::binary_function<std::pair<size_t, size_t>, std::pair<size_t, size_t>, bool> {
         bool operator()(const std::pair<size_t, size_t> & left, const std::pair<size_t, size_t> & right) const {
             if (left.first == right.first) {
                 return left.second < right.second;
             } else {
                 return left.first < right.first;
             }
         }
     };

     using correspondences_t = std::map<std::pair<size_t, size_t>, double, key_lthan>;

     explicit PointRegRelaxation(arma::mat const & source_pts,
             arma::mat const & target_pts, Config const & config) : config_(config) {
         ptr_obj_ = std::make_unique<ConstrainedObjective>(source_pts, target_pts, config);
         optimum_.resize(ptr_obj_->state_length());
     }
     
     ~PointRegRelaxation();

     void warm_start(Dvec & z_init) const noexcept;
     status_t find_optimum() noexcept;
     void find_correspondences() noexcept;
     correspondences_t const get_correspondences() const noexcept { return correspondences_; }
     arma::colvec const get_optimum() const noexcept { return optimum_; }

 private:
     std::unique_ptr<ConstrainedObjective> ptr_obj_;
     Config config_;
     correspondences_t correspondences_;
     arma::colvec optimum_; 
};

Status registration(arma::mat const & source_pts, arma::mat const & target_pts,
    Config const & config, arma::colvec & optimum, PointRegRelaxation::correspondences_t & correspondences,
    arma::mat44 & H_optimal) noexcept;
