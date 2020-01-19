//! c/c++ headers
#include <string>
#include <fstream>
#include <streambuf>
//! dependency headers
#include <nlohmann/json.hpp>
//! boost boilerplate
#define BOOST_TEST_MODULE TestRegistration test
#include <boost/test/unit_test.hpp>
//! unit-under-test header
#include "correspondences/correspondences.hpp"
#include "registration/registration.hpp"

namespace cor = correspondences;
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
    arma::mat const _dst = spread * arma::randn(3, 20);
    arma::vec3 const dst_c = arma::vec3(arma::mean(_dst, 1));
    arma::vec3 const src_c = {2, 3, 0};

    //! yaw, pitch, roll
    double const & psi = M_PI / 4;
    double const & theta = M_PI / 3;
    double const & phi = M_PI / 8;

    //! rpy sequence
    arma::mat33 R;
    make_euler(psi, theta, phi, R);

    //! create source points: R*(dest - dest_centroid) + source_centroid
    arma::mat const src( R * (_dst - arma::repmat(dst_c, 1, _dst.n_cols))
            + arma::repmat(src_c, 1, _dst.n_cols) );

    //! allocate nominal output
    arma::mat44 H_opt;

    //! TEST CASE 1: nominal
    reg::best_fit_transform(src, _dst, H_opt);

    //! transform source points by the found transformation (R, t) and check for (near) equality
    arma::mat33 opt_R = H_opt(arma::span(0, 2), arma::span(0, 2));
    arma::vec3 opt_t = H_opt(arma::span(0, 2), 3);
    arma::mat const dst( opt_R * src + arma::repmat(opt_t, 1, _dst.n_cols) );
    BOOST_CHECK(arma::approx_equal(dst, _dst, "absdiff", FLOAT_TOL));
}

BOOST_AUTO_TEST_CASE( full_pipeline_test ) {
    //! load unit test data from json
    //! NOTE: this test data was generated without adding noise
    std::ifstream ifs(data_path + "/registration-data.json");
    std::string json_str = std::string((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
    json json_data = json::parse(json_str);

    //! setup configuration struct for test
    cor::Config config;
    config.epsilon = 0.1;
    config.pairwise_dist_threshold = 0.1;
    config.corr_threshold = 0.5;
    config.do_warm_start = true;

    //! source pts
    auto const rows_S = json_data["source_pts"].size();
    auto const cols_S = json_data["source_pts"][0].size();
    size_t i = 0;
    arma::mat src_pts(rows_S, cols_S);
    for (auto const & it : json_data["source_pts"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            src_pts(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! target pts
    auto const rows_T = json_data["target_pts"].size();
    auto const cols_T = json_data["target_pts"][0].size();
    i = 0;
    arma::mat tgt_pts(rows_T, cols_T);
    for (auto const & it : json_data["target_pts"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            tgt_pts(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! correspondences
    i = 0;
    cor::PointRegRelaxation::correspondences_t _corrs;
    for (auto const & it : json_data["correspondences"]) {
        auto key = std::make_pair(i, static_cast<size_t>(it));
        _corrs[key] = config.corr_threshold;  //! this is an unnecessary assignment that won't be used below
        ++i;
    }

    //! optimal transformation
    i = 0;
    arma::mat44 _H;
    for (auto const & it : json_data["src_to_tgt"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            _H(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! make the call
    arma::mat44 H;
    cor::PointRegRelaxation::correspondences_t corrs;
    arma::colvec optimum;  //! not-used; corrs has the desired scores
    BOOST_CHECK( reg::registration(src_pts, tgt_pts, config, src_pts.n_cols,
                optimum, corrs, H) == reg::Status::Success );

    //! checking that key is present in both correspondence sets is enough; see `find_correspondences` implementation
    for (auto const & c : _corrs) {
        auto key = c.first;
        BOOST_CHECK(corrs.find(key) != corrs.end());
    }
    //! check that transformation is close to truth
    BOOST_CHECK(arma::approx_equal(H, _H, "absdiff", FLOAT_TOL));
}

BOOST_AUTO_TEST_CASE( minimal_correspondences_full_pipeline_test ) {
    //! load unit test data from json
    //! NOTE: this test data was generated without adding noise
    std::ifstream ifs(data_path + "/registration-data-mincorr.json");
    std::string json_str = std::string((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
    json json_data = json::parse(json_str);

    //! setup configuration struct for test
    cor::Config config;
    config.epsilon = 0.1;
    config.pairwise_dist_threshold = 0.1;
    config.corr_threshold = 0.5;
    config.do_warm_start = true;

    //! source pts
    auto const rows_S = json_data["source_pts"].size();
    auto const cols_S = json_data["source_pts"][0].size();
    size_t i = 0;
    arma::mat src_pts(rows_S, cols_S);
    for (auto const & it : json_data["source_pts"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            src_pts(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! target pts
    auto const rows_T = json_data["target_pts"].size();
    auto const cols_T = json_data["target_pts"][0].size();
    i = 0;
    arma::mat tgt_pts(rows_T, cols_T);
    for (auto const & it : json_data["target_pts"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            tgt_pts(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! correspondences
    i = 0;
    cor::PointRegRelaxation::correspondences_t _corrs;
    for (auto const & it : json_data["correspondences"]) {
        auto key = std::make_pair(i, static_cast<size_t>(it));
        _corrs[key] = config.corr_threshold;  //! this is an unnecessary assignment that won't be used below
        ++i;
    }

    //! optimal transformation
    i = 0;
    arma::mat44 _H;
    for (auto const & it : json_data["src_to_tgt"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            _H(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    auto const min_corr = static_cast<size_t>(json_data["min_corr"]);

    //! make the call
    arma::mat44 H;
    cor::PointRegRelaxation::correspondences_t corrs;
    arma::colvec optimum;  //! not-used; corrs has the desired scores
    BOOST_CHECK( reg::registration(src_pts, tgt_pts, config, min_corr,
                optimum, corrs, H) == reg::Status::Success );
    //! checking that key is present in both correspondence sets is enough; see `find_correspondences` implementation
    for (auto const & c : _corrs) {
        auto key = c.first;
        BOOST_CHECK(corrs.find(key) != corrs.end());
    }
    //! check that transformation is close to truth
    BOOST_CHECK(arma::approx_equal(H, _H, "absdiff", FLOAT_TOL));
}
