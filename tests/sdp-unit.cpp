//! c/c++ headers
#include <cmath>
//! dependency headers
#include <ensmallen.hpp>
//! boost boilerplate
#define BOOST_TEST_MODULE TestSDP test
#include <boost/test/unit_test.hpp>
//! project headers
#include "correspondences/correspondences-sdp.hpp"

using namespace ens;
namespace cors = correspondences_sdp;

/**
 * constants for test
 */
static const double FLOAT_TOL(1e-3);  // XXX(jwd) - values on wikipedia only reported to 3 s.f.

template <typename SDPType>
static bool CheckKKT(const SDPType& sdp,
                     const arma::mat& X,
                     const arma::vec& ysparse,
                     const arma::vec& ydense,
                     const arma::mat& Z)
{
  // Require that the KKT optimality conditions for sdp are satisfied
  // by the primal-dual pair (X, y, Z).

  if (!X.is_sympd())
    return false;
  if (!Z.is_sympd())
    return false;

  bool success = true;
  const double normXz = arma::norm(X * Z, "fro");
  success &= (std::abs(normXz) < 1e-5);

  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
  {
    success &= (std::abs(
        arma::dot(sdp.SparseA()[i], X) - sdp.SparseB()[i]) < 1e-5);
  }

  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
  {
    success &= (std::abs(
        arma::dot(sdp.DenseA()[i], X) - sdp.DenseB()[i]) < 1e-5);
  }

  arma::mat dualCheck = Z - sdp.C();
  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
    dualCheck += ysparse(i) * sdp.SparseA()[i];
  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
    dualCheck += ydense(i) * sdp.DenseA()[i];
  const double dualInfeas = arma::norm(dualCheck, "fro");
  success &= (dualInfeas < 1e-5);

  return success;
}

//static inline SDP<arma::sp_mat> Ensmallen unit tests set this outside of function call??
BOOST_AUTO_TEST_CASE( ensmallen_sdp_example_minimum ) {
  /*!
   * @brief example is copied from Ensmallen docs
   * @see https://ensmallen.org/docs.html#semidefinite-programs
   * @see https://en.wikipedia.org/wiki/Semidefinite_programming#Example_1
   *
   */
  // We will build a toy semidefinite program and then use the PrimalDualSolver to find a solution

  // The semi-definite constraint looks like:
  //
  // [ 1  x_12  x_13  0  0  0  0 ]
  // [     1    x_23  0  0  0  0 ]
  // [            1   0  0  0  0 ]
  // [               s1  0  0  0 ]  >= 0
  // [                  s2  0  0 ]
  // [                     s3  0 ]
  // [                        s4 ]

  // x_11 == 0
  arma::sp_mat A0(7, 7); A0.zeros();
  A0(0, 0) = 1.;

  // x_22 == 0
  arma::sp_mat A1(7, 7); A1.zeros();
  A1(1, 1) = 1.;

  // x_33 == 0
  arma::sp_mat A2(7, 7); A2.zeros();
  A2(2, 2) = 1.;

  // x_12 <= -0.1  <==>  x_12 + s1 == -0.1, s1 >= 0
  arma::sp_mat A3(7, 7); A3.zeros();
  A3(1, 0) = A3(0, 1) = 1.; A3(3, 3) = 2.;

  // -0.2 <= x_12  <==>  x_12 - s2 == -0.2, s2 >= 0
  arma::sp_mat A4(7, 7); A4.zeros();
  A4(1, 0) = A4(0, 1) = 1.; A4(4, 4) = -2.;

  // x_23 <= 0.5  <==>  x_23 + s3 == 0.5, s3 >= 0
  arma::sp_mat A5(7, 7); A5.zeros();
  A5(2, 1) = A5(1, 2) = 1.; A5(5, 5) = 2.;

  // 0.4 <= x_23  <==>  x_23 - s4 == 0.4, s4 >= 0
  arma::sp_mat A6(7, 7); A6.zeros();
  A6(2, 1) = A6(1, 2) = 1.; A6(6, 6) = -2.;

  std::vector<arma::sp_mat> ais({A0, A1, A2, A3, A4, A5, A6});

  SDP<arma::sp_mat> sdp(7, 7 + 4 + 4 + 4 + 3 + 2 + 1, 0);

  for (size_t j = 0; j < 3; j++)
  {
    // x_j4 == x_j5 == x_j6 == x_j7 == 0
    for (size_t i = 0; i < 4; i++)
    {
      arma::sp_mat A(7, 7); A.zeros();
      A(i + 3, j) = A(j, i + 3) = 1;
      ais.emplace_back(A);
    }
  }

  // x_45 == x_46 == x_47 == 0
  for (size_t i = 0; i < 3; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 4, 3) = A(3, i + 4) = 1;
    ais.emplace_back(A);
  }

  // x_56 == x_57 == 0
  for (size_t i = 0; i < 2; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 5, 4) = A(4, i + 5) = 1;
    ais.emplace_back(A);
  }

  // x_67 == 0
  arma::sp_mat A(7, 7); A.zeros();
  A(6, 5) = A(5, 6) = 1;
  ais.emplace_back(A);

  std::swap(sdp.SparseA(), ais);

  sdp.SparseB().zeros();
  sdp.SparseB()[0] = sdp.SparseB()[1] = sdp.SparseB()[2] = 1.;
  sdp.SparseB()[3] = -0.2; sdp.SparseB()[4] = -0.4;
  sdp.SparseB()[5] = 1.; sdp.SparseB()[6] = 0.8;

  sdp.C().zeros();
  sdp.C()(0, 2) = sdp.C()(2, 0) = 1.;

  // That took a long time but we finally set up the problem right!  Now we can
  // use the PrimalDualSolver to solve it.
  // ens::PrimalDualSolver could be replaced with ens::LRSDP or other ensmallen
  // SDP solvers.
  PrimalDualSolver<> solver;  // XXX(jwd) - the <> is a hack until ensmallen 2.12.0 is released
  arma::mat X, Z;
  arma::mat ysparse, ydense;
  sdp.GetInitialPoints(X, ysparse, ydense, Z);
  // ysparse, ydense, and Z hold the primal and dual variables found during the
  // optimization.
  static_cast<void>(solver.Optimize(sdp, X, ysparse, ydense, Z));
  BOOST_CHECK( CheckKKT(sdp, X, ysparse, ydense, Z) );
  BOOST_CHECK_SMALL(std::abs(X(0, 2) + 0.978), FLOAT_TOL);  // expected value is -0.978
}

BOOST_AUTO_TEST_CASE( ensmallen_sdp_example_maximum ) {
  /*!
   * @brief example is copied from Ensmallen docs
   * @see https://ensmallen.org/docs.html#semidefinite-programs
   * @see https://en.wikipedia.org/wiki/Semidefinite_programming#Example_1
   *
   */
  // We will build a toy semidefinite program and then use the PrimalDualSolver to find a solution

  // The semi-definite constraint looks like:
  //
  // [ 1  x_12  x_13  0  0  0  0 ]
  // [     1    x_23  0  0  0  0 ]
  // [            1   0  0  0  0 ]
  // [               s1  0  0  0 ]  >= 0
  // [                  s2  0  0 ]
  // [                     s3  0 ]
  // [                        s4 ]

  // x_11 == 0
  arma::sp_mat A0(7, 7); A0.zeros();
  A0(0, 0) = 1.;

  // x_22 == 0
  arma::sp_mat A1(7, 7); A1.zeros();
  A1(1, 1) = 1.;

  // x_33 == 0
  arma::sp_mat A2(7, 7); A2.zeros();
  A2(2, 2) = 1.;

  // x_12 <= -0.1  <==>  x_12 + s1 == -0.1, s1 >= 0
  arma::sp_mat A3(7, 7); A3.zeros();
  A3(1, 0) = A3(0, 1) = 1.; A3(3, 3) = 2.;

  // -0.2 <= x_12  <==>  x_12 - s2 == -0.2, s2 >= 0
  arma::sp_mat A4(7, 7); A4.zeros();
  A4(1, 0) = A4(0, 1) = 1.; A4(4, 4) = -2.;

  // x_23 <= 0.5  <==>  x_23 + s3 == 0.5, s3 >= 0
  arma::sp_mat A5(7, 7); A5.zeros();
  A5(2, 1) = A5(1, 2) = 1.; A5(5, 5) = 2.;

  // 0.4 <= x_23  <==>  x_23 - s4 == 0.4, s4 >= 0
  arma::sp_mat A6(7, 7); A6.zeros();
  A6(2, 1) = A6(1, 2) = 1.; A6(6, 6) = -2.;

  std::vector<arma::sp_mat> ais({A0, A1, A2, A3, A4, A5, A6});

  SDP<arma::sp_mat> sdp(7, 7 + 4 + 4 + 4 + 3 + 2 + 1, 0);

  for (size_t j = 0; j < 3; j++)
  {
    // x_j4 == x_j5 == x_j6 == x_j7 == 0
    for (size_t i = 0; i < 4; i++)
    {
      arma::sp_mat A(7, 7); A.zeros();
      A(i + 3, j) = A(j, i + 3) = 1;
      ais.emplace_back(A);
    }
  }

  // x_45 == x_46 == x_47 == 0
  for (size_t i = 0; i < 3; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 4, 3) = A(3, i + 4) = 1;
    ais.emplace_back(A);
  }

  // x_56 == x_57 == 0
  for (size_t i = 0; i < 2; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 5, 4) = A(4, i + 5) = 1;
    ais.emplace_back(A);
  }

  // x_67 == 0
  arma::sp_mat A(7, 7); A.zeros();
  A(6, 5) = A(5, 6) = 1;
  ais.emplace_back(A);

  std::swap(sdp.SparseA(), ais);

  sdp.SparseB().zeros();
  sdp.SparseB()[0] = sdp.SparseB()[1] = sdp.SparseB()[2] = 1.;
  sdp.SparseB()[3] = -0.2; sdp.SparseB()[4] = -0.4;
  sdp.SparseB()[5] = 1.; sdp.SparseB()[6] = 0.8;

  sdp.C().zeros();
  sdp.C()(0, 2) = sdp.C()(2, 0) = -1.;

  // That took a long time but we finally set up the problem right!  Now we can
  // use the PrimalDualSolver to solve it.
  // ens::PrimalDualSolver could be replaced with ens::LRSDP or other ensmallen
  // SDP solvers.
  PrimalDualSolver<> solver;  // XXX(jwd) - the <> is a hack until ensmallen 2.12.0 is released
  arma::mat X, Z;
  arma::mat ysparse, ydense;
  sdp.GetInitialPoints(X, ysparse, ydense, Z);
  // ysparse, ydense, and Z hold the primal and dual variables found during the
  // optimization.
  static_cast<void>(solver.Optimize(sdp, X, ysparse, ydense, Z));
  BOOST_CHECK( CheckKKT(sdp, X, ysparse, ydense, Z) );
  BOOST_CHECK_SMALL(std::abs(X(0, 2) - 0.872), FLOAT_TOL);  // expected value is +0.872
}

BOOST_AUTO_TEST_CASE( test_constructor ) {
  cors::Config config;
  config.epsilon = 0.1;
  config.pairwise_dist_threshold = 0.1;
  config.corr_threshold = 0.9;
  config.n_pair_threshold = 100;

  arma::mat src(3, 4, arma::fill::randu);
  arma::mat tgt(src);
  cors::ConstrainedObjective opt(src, tgt, config, 3);

  auto sdp = opt.get_sdp();
  /*
  for (auto a : sdp.SparseA()) {
      arma::mat A(a);
      std::cout << a << std::endl;
  }*/
}
