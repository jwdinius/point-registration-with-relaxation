//! c/c++ headers
#include <string>
#include <fstream>
#include <streambuf>
//! dependency headers
#include <armadillo>
#include <nlohmann/json.hpp>
//! boost boilerplate
#define BOOST_TEST_MODULE TestTransformUtils test
#include <boost/test/unit_test.hpp>
//! project headers
#include "registration/registration.hpp"

namespace reg = registration;

using json = nlohmann::json;

/**
 * constants for test
 */
static const std::string data_path("../tests/testdata");  // NOLINT [runtime/string]
static const double FLOAT_TOL(1e-7);

/**
 * @brief Construct roll-pitch-yaw rotation matrix from angles
 *
 * @param [in] yaw rotation angle about z-axis
 * @param [in] pitch rotation angle about y-axis
 * @param [in] roll rotation angle about x-axis
 * @param [out] R rpy rotation matrix
 * @return
 */
void make_euler(double const & yaw, double const & pitch, double const & roll, arma::mat33 & R) {
    //! yaw
    auto const & psi = yaw;
    double const s_psi = std::sin(psi);
    double const c_psi = std::cos(psi);
    arma::mat33 const Ry = {{c_psi, -s_psi, 0}, {s_psi, c_psi, 0}, {0, 0, 1}};
    //! pitch
    auto const & theta = pitch;
    double const s_theta = std::sin(theta);
    double const c_theta = std::cos(theta);
    arma::mat33 const Rp = {{c_theta, 0, s_theta}, {0, 1, 0}, {-s_theta, 0, c_theta}};
    //! roll
    auto const & phi = roll;
    double const s_phi = std::sin(phi);
    double const c_phi = std::cos(phi);
    arma::mat33 const Rr = {{1, 0, 0}, {0, c_phi, -s_phi}, {0, s_phi, c_phi}};

    //! rpy sequence
    R = Ry * Rp * Rr;
}

/**
 * @brief Compute Euler angles from rotation matrix
 *
 * @param [in] R rpy rotation matrix
 * @param [out] angles column vector with (roll, pitch, yaw) angles of rpy rotation sequence
 * @param [in] flip_pitch (optional, default=false) use alternate convention for pitch angle
 * @return
 *
 * @note this routine assumes input matrix is a valid rotation matrix
 */
void find_euler_angles(arma::mat33 const & R, arma::vec3 & angles, bool flip_pitch = false) {
    auto const & r_31 = R(2, 0);
    if (std::abs(r_31 - 1.0) <= std::numeric_limits<double>::epsilon() ||
            std::abs(r_31 + 1.0) <= std::numeric_limits<double>::epsilon()) {
        //! gimbal lock condition
        angles(2) = static_cast<double>(0);
        if (r_31 < 0) {
            angles(1) = 0.5 * M_PI;
            angles(0) = angles(2) + std::atan2(R(0, 1), R(0, 2));
        } else {
            angles(1) = -0.5 * M_PI;
            angles(0) = -angles(2) + std::atan2(-R(0, 1), -R(0, 2));
        }
    } else {
        if (flip_pitch) {
            angles(1) = M_PI + std::asin(r_31);
        } else {
            angles(1) = -std::asin(r_31);
        }
        angles(0) = std::atan2(R(2, 1) / std::cos(angles(0)), R(2, 2) / std::cos(angles(0)));
        angles(2) = std::atan2(R(1, 0) / std::cos(angles(0)), R(0, 0) / std::cos(angles(0)));
    }
}

BOOST_AUTO_TEST_CASE( best_fit_transform_test ) {
    arma::arma_rng::set_seed(11011);
    double const & spread = 10;
    arma::mat const dst = spread * arma::randn(3, 20);
    arma::vec3 const dst_c = arma::vec3(arma::mean(dst, 1));
    arma::vec3 const src_c = {2, 3, 0};

    //! yaw, pitch, roll
    double const & psi = M_PI / 4;
    double const & theta = M_PI / 3;
    double const & phi = M_PI / 8;

    //! rpy sequence
    arma::mat33 R;
    make_euler(psi, theta, phi, R);

    //! create source points: R*(dst - dst_c) + src_c
    arma::mat const src( R * (dst - arma::repmat(dst_c, 1, dst.n_cols))
            + arma::repmat(src_c, 1, dst.n_cols) );

    //! allocate nominal output
    arma::mat44 H_opt;

    //! TEST CASE 1: nominal
    BOOST_CHECK( reg::best_fit_transform(src, dst, H_opt) );

    //! transform source points by the found transformation H = [R; t; 0 1] and check for (near) equality
    arma::mat33 R_opt = H_opt(arma::span(0, 2), arma::span(0, 2));
    arma::vec3 t_opt = H_opt(arma::span(0, 2), 3);
    arma::mat const src_xform( R_opt * src + arma::repmat(t_opt, 1, dst.n_cols) );
    BOOST_CHECK( arma::approx_equal(dst, src_xform, "absdiff", FLOAT_TOL) );
}

BOOST_AUTO_TEST_CASE( icp_test ) {
    //! set the random seed for repeatability
    arma::arma_rng::set_seed(11011);
    double const & spread = 10;
    arma::mat dst = spread * arma::randn(3, 20);
    arma::vec3 const dst_c = arma::mean(dst, 1);
    arma::vec3 const src_c = {2, 3, 0};

    //! yaw, pitch, roll
    double const & psi = M_PI / 4;
    double const & theta = M_PI / 3;
    double const & phi = M_PI / 8;

    //! rpy sequence
    arma::mat33 R;
    make_euler(psi, theta, phi, R);

    //! create source points: R*(dst - dst_c) + src_c
    arma::mat const src( R * (dst - arma::repmat(dst_c, 1, dst.n_cols))
            + arma::repmat(src_c, 1, dst.n_cols) );
    
    //! algorithm arguments
    size_t const & max_its = 20;
    double const & tol = FLOAT_TOL;
    double const & rej_ratio = 0.1;

    //! noise parameters
    double const & angle_noise = 0.05;  // ~3deg 1-sigma
    double const & pos_noise = 0.1;

    //! add noise to destination points
    //!  angle
    arma::vec3 angle_n(arma::fill::randn);
    angle_n *= angle_noise;
    arma::mat33 Rn;
    make_euler(angle_n(0), angle_n(1), angle_n(2), Rn);
    //!  position
    dst += pos_noise * arma::randn(dst.n_rows, dst.n_cols);

    //! initial guess for transformation - icp algorithm needs good initial guess to be successful
    arma::mat44 H_init(arma::fill::eye);
    H_init(arma::span(0, 2), arma::span(0, 2)) = Rn * R.t(); 
    H_init(arma::span(0, 2), 3) = dst_c - Rn * R.t() * src_c;
    
    //! allocate nominal output
    arma::mat44 H_opt;
    {
        //! TEST CASE 1: nominal (no reordering)
        BOOST_CHECK( reg::iterative_closest_point(src, dst, H_init, max_its, tol, rej_ratio, H_opt) );
        arma::vec3 angles;
        arma::mat33 R_opt = H_opt(arma::span(0, 2), arma::span(0, 2));
        arma::vec3 t_opt = H_opt(arma::span(0, 2), 3);
        find_euler_angles(R_opt.t(), angles);
        BOOST_CHECK(std::abs(angles(0) - phi) < 3 * angle_noise);
        BOOST_CHECK(std::abs(angles(1) - theta) < 3 * angle_noise);
        BOOST_CHECK(std::abs(angles(2) - psi) < 3 * angle_noise);
        BOOST_CHECK(arma::approx_equal(t_opt, dst_c - R.t() * src_c, "absdiff", 3 * pos_noise));
    }
    {
        //! TEST CASE 2: nominal test with random point ordering (removes implicit correspondence in construction)
        arma::uvec const ordering = arma::randperm(dst.n_cols, dst.n_cols);
        arma::mat const dst_shuffled = dst.cols( ordering );
        arma::vec3 angles;
        arma::mat33 R_opt = H_opt(arma::span(0, 2), arma::span(0, 2));
        arma::vec3 t_opt = H_opt(arma::span(0, 2), 3);
        BOOST_CHECK( reg::iterative_closest_point(src, dst_shuffled, H_init, max_its, tol, rej_ratio, H_opt) );
        find_euler_angles(R_opt.t(), angles);
        BOOST_CHECK(std::abs(angles(0) - phi) < 3 * angle_noise);
        BOOST_CHECK(std::abs(angles(1) - theta) < 3 * angle_noise);
        BOOST_CHECK(std::abs(angles(2) - psi) < 3 * angle_noise);
        BOOST_CHECK(arma::approx_equal(t_opt, dst_c - R.t() * src_c, "absdiff", 3 * pos_noise));
    }
}

BOOST_AUTO_TEST_CASE( icp_test_compare_to_matlab ) {
    //! load unit test data from json
    std::ifstream ifs(data_path + "/icp.json");
    std::string json_str = std::string((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
    json json_data = json::parse(json_str);

    //! TM - source points
    auto const & rows_M = json_data["TM"].size();
    auto const & cols_M = json_data["TM"][0].size();
    size_t i = 0;
    arma::mat  src_pts(rows_M, cols_M);
    for (auto const & it : json_data["TM"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            src_pts(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! B - destination points (to map src_pts onto)
    auto const & rows_B = json_data["B"].size();
    auto const & cols_B = json_data["B"][0].size();
    i = 0;
    arma::mat dst_pts(rows_B, cols_B);
    for (auto const & it : json_data["B"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            dst_pts(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! Ricp - optimal rotation
    i = 0;
    arma::mat33 R_opt_matlab;
    for (auto const & it : json_data["Ricp"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            R_opt_matlab(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! Ticp - optimal translation
    i = 0;
    arma::vec3 t_opt_matlab;
    for (auto const & it : json_data["Ticp"]) {
        t_opt_matlab(i) = static_cast<double>(it[0]);
        ++i;
    }

    //! combine above into homogeneous representation
    arma::mat44 H_opt_matlab(arma::fill::eye);
    H_opt_matlab(arma::span(0, 2), arma::span(0, 2)) = R_opt_matlab;
    H_opt_matlab(arma::span(0, 2), 3) = t_opt_matlab;

    //! algorithm arguments
    size_t const & max_its = 100;
    double const & tol = 1e-12;
    double const & rej_ratio = 0.1;

    //! TEST CASE 1: nominal call from matlab implementation
    //! - unit test data has initial guess included, so pass identity as H_init
    arma::mat44 H_init(arma::fill::eye);
    //! allocate container for output
    arma::mat44 H_opt;
    BOOST_CHECK( reg::iterative_closest_point(src_pts, dst_pts, H_init, max_its, tol, rej_ratio, H_opt) );
    BOOST_CHECK(arma::approx_equal(H_opt, H_opt_matlab, "absdiff", 5e-3));
}
