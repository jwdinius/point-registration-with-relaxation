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
//! dependency headers
//! project headers
#include "correspondences/correspondences-sdp.hpp"

//! namespaces
namespace cors = correspondences_sdp;

cors::ConstrainedObjective::ConstrainedObjective(arma::mat const & source_pts,
        arma::mat const & target_pts, cors::Config const & config, size_t const & min_corr) :
    m_(static_cast<size_t>(source_pts.n_cols)),
    n_(static_cast<size_t>(target_pts.n_cols)),
    min_corr_(min_corr),
    n_constraints_( m_ + 1 + n_ + 1 + 1 + 1 + 2*( (m_+1)*(n_+1) - 1 ) + ((m_+1)*(n_+1)+1)*(2*(m_+1)*(n_+1)-2) + ((m_+1)*(n_+1)-1)*(2*(m_+1)*(n_+1)-3) ),
    sdp_((m_+1)*(n_+1) + 1 + 2*( (m_+1)*(n_+1) - 1 ), n_constraints_, 0)
{
    size_t const state_len = (m_ + 1)*(n_ + 1);
    size_t const sdp_dim = 3 * state_len - 1;  // == state_len + 1 + 2*( state_len - 1 )
    sdp_.SparseB().zeros();

    std::cout << "+++ " << n_constraints_ << std::endl;
    size_t cons_cntr = 0;
    sdp_.SparseA() = {};

    {
      /**
       * \brief sum_{j=0}^n X_{i*(n+1)+j, state_len} = 1 for all i in [0, m) 
       */
      for (size_t i = 0; i < m_; i++) {
        arma::sp_mat A(sdp_dim, sdp_dim);
        A.zeros();
        for (size_t j = 0; j <= n_; j++) {
            A(i*(n_+1) + j, state_len) = A(state_len, i*(n_+1) + j) = 1;
        }
        sdp_.SparseA().emplace_back(A);
        sdp_.SparseB()(cons_cntr) = 2;
        ++cons_cntr;
      }
    }

    {
      /**
       * \brief sum_{j=0}^{n-1} X_{m*(n+1)+j, state_len} = n - k
       */
      arma::sp_mat A(sdp_dim, sdp_dim);
      A.zeros();
      for (size_t j = 0; j < n_; j++) {
          A(m_*(n_+1) + j, state_len) = A(state_len, m_*(n_+1) + j) = 1;
      }
      sdp_.SparseA().emplace_back(A);
      sdp_.SparseB()(cons_cntr) = 2 * (n_ - min_corr_);
      ++cons_cntr;
    }

    {
      /**
       * \brief sum_{i=0}^m X_{i*(n+1)+j, state_len} = 1 for all j in [0, n) 
       */
      for (size_t j = 0; j < n_; j++) {
        arma::sp_mat A(sdp_dim, sdp_dim);
        A.zeros();
        for (size_t i = 0; i <= m_; i++) {
            A(i*(n_+1) + j, state_len) = A(state_len, i*(n_+1) + j) = 1;
        }
        sdp_.SparseA().emplace_back(A);
        sdp_.SparseB()(cons_cntr) = 2;
        ++cons_cntr;
      }
    }
    
    {
      /**
       * \brief sum_{j=0}^{n-1} X_{m*(n+1)+j, state_len} = n - k
       */
      arma::sp_mat A(sdp_dim, sdp_dim);
      A.zeros();
      for (size_t i = 0; i < m_; i++) {
          A(i*(n_+1) + n_, state_len) = A(state_len, i*(n_+1) + n_) = 1;
      }
      sdp_.SparseA().emplace_back(A);
      sdp_.SparseB()(cons_cntr) = 2 * (m_ - min_corr_);
      ++cons_cntr;
    }

    {
      /**
       * \brief X_{state_len, state_len} = 1
       */
      arma::sp_mat A(sdp_dim, sdp_dim);
      A.zeros();
      A(state_len, state_len) = 1;
      sdp_.SparseA().emplace_back(A);
      sdp_.SparseB()(cons_cntr) = 1;
      ++cons_cntr;
    }
    
    {
      /**
       * \brief X_{state_len-1, state_len} = X_{state_len, state_len-1} = 0
       */
      arma::sp_mat A(sdp_dim, sdp_dim);
      A.zeros();
      A(state_len-1, state_len) = 1; A(state_len-1, state_len) = 1;
      sdp_.SparseA().emplace_back(A);
      sdp_.SparseB()(cons_cntr) = 0;
      ++cons_cntr;
    }

    {
      /**
       * \brief X_{j, state_len + k} = X_{state_len+k, j} = 0 for j = [0, state_len], k = [state_len, sdp_dim)
       */
       for (size_t j = 0; j <= state_len; ++j) {
           for (size_t k = state_len+1; k < sdp_dim; ++k) {
               arma::sp_mat A(sdp_dim, sdp_dim);
               A.zeros();
               A(j, k) = 1; A(k, j) = 1;
               sdp_.SparseA().emplace_back(A);
               sdp_.SparseB()(cons_cntr) = 0;
               ++cons_cntr;
           }
       }
    }

    {
      /**
       * \brief enforce zero terms in the off-diagonal of the slack block
       */
       for (size_t j = state_len+1; j < sdp_dim; ++j) {
           for (size_t k = j+1; k < sdp_dim; ++k) {
               arma::sp_mat A(sdp_dim, sdp_dim);
               A.zeros();
               A(j, k) = 1; A(k, j) = 1;
               sdp_.SparseA().emplace_back(A);
               sdp_.SparseB()(cons_cntr) = 0;
               ++cons_cntr;
           }
       }
    }

    {
      /**
       * \brief Y(state_len, j) = Y(j, state_len) >= 0
       */
       for (size_t j = 0; j < state_len-1; ++j) {
           arma::sp_mat A(sdp_dim, sdp_dim);
           A.zeros();
           A(j, state_len) = 1; A(state_len, j) = 1;
           A(state_len+1+j, state_len+1+j) = -2;
           sdp_.SparseA().emplace_back(A);
           sdp_.SparseB()(cons_cntr) = 0;
           ++cons_cntr;
       }

      /**
       * \brief Y(state_len, j) = Y(j, state_len) <= 1
       */
       for (size_t j = 0; j < state_len-1; ++j) {
           arma::sp_mat A(sdp_dim, sdp_dim);
           A.zeros();
           A(j, state_len) = 1; A(state_len, j) = 1;
           A(state_len+state_len-1+1+j, state_len+state_len-1+1+j) = -2;
           sdp_.SparseA().emplace_back(A);
           sdp_.SparseB()(cons_cntr) = 0;
           ++cons_cntr;
       }
    }
    std::cout << "#### final " << cons_cntr << std::endl;
    generate_weight_tensor(source_pts, target_pts, config);  // sets sdp_.C();
}

cors::ConstrainedObjective::~ConstrainedObjective(){ }

/**
 * @brief compute pairwise consistency score; see Eqn. 49 from reference
 *
 * @param [in] si ith point in source distribution
 * @param [in] tj jth point in target distribution
 * @param [in] sk kth point in source distribution
 * @param [in] tl lth point in target distribution
 * @return pairwise consistency score for (i, j, k, l)
 */
double cors::consistency(arma::vec3 const & si, arma::vec3 const & tj, arma::vec3 const & sk,
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
void cors::ConstrainedObjective::generate_weight_tensor(arma::mat const & source_pts, arma::mat const & target_pts,
    cors::Config const & config) noexcept {
    sdp_.C().zeros();
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
                        if (c <= eps && arma::norm(si - sk, 2) >= pw_thresh && arma::norm(tj - tl, 2) >= pw_thresh) {
                            sdp_.C()(i*(n+1)+j, k*(n+1) + l) = -std::exp(-c);
                        }
                    }
                }
            }
        }
    }
    return;
}

void cors::ConstrainedObjective::find_optimum() const noexcept {
    // That took a long time but we finally set up the problem right!  Now we can
    // use the PrimalDualSolver to solve it.
    // ens::PrimalDualSolver could be replaced with ens::LRSDP or other ensmallen
    // SDP solvers.
    ens::PrimalDualSolver<> solver;  // XXX(jwd) - the <> is a hack until ensmallen 2.12.0 is released
    arma::mat X, Z;
    arma::mat ysparse, ydense;
    sdp_.GetInitialPoints(X, ysparse, ydense, Z);
    // ysparse, ydense, and Z hold the primal and dual variables found during the
    // optimization.
    static_cast<void>(solver.Optimize(sdp_, X, ysparse, ydense, Z));
    return;
};
