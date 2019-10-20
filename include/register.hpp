#pragma once
#include <cmath>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <armadillo>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>

struct Config {
    Config() :
        epsilon(1e-1), pairwise_dist_threshold(1e-1) { }
    double epsilon, pairwise_dist_threshold;
};

enum class Status {
    BadInput,
    SolverFailed,
    Success
};

using WeightKey_t = std::tuple<size_t, size_t, size_t, size_t>;

/*
struct key_hash : public std::unary_function<WeightKey_t, size_t> {
   size_t operator()(const WeightKey_t& k) const {
      return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k) ^ std::get<3>(k);
   }
};*/

struct key_hash : public std::unary_function<WeightKey_t, size_t> {
    size_t operator()(const WeightKey_t& k) const {
        size_t h1 = std::hash<size_t>()(std::get<0>(k));
        size_t h2 = std::hash<size_t>()(std::get<1>(k));
        size_t h3 = std::hash<size_t>()(std::get<2>(k));
        size_t h4 = std::hash<size_t>()(std::get<3>(k));
        return (h1 << 32) + (h2 << 32) + (h3 << 32) + h4;
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

 private:
     WeightTensor weights_;
     size_t m_, n_, n_constraints_;
};

class PointRegRelaxation {
 public:
     using Dvec = CPPAD_TESTVECTOR(double);
     using status_t = CppAD::ipopt::solve_result<Dvec>::status_type;

     explicit PointRegRelaxation(arma::mat const & source_pts,
             arma::mat const & target_pts, Config const & config) {
         ptr_obj_ = std::make_unique<ConstrainedObjective>(source_pts, target_pts, config);
     }
     
     ~PointRegRelaxation();
     
     status_t find_optimum(arma::colvec & sol) const noexcept;

 private:
     std::unique_ptr<ConstrainedObjective> ptr_obj_;
};
