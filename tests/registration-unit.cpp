//! c/c++ headers
#include <string>
#include <fstream>
#include <streambuf>
//! dependency headers
#include <armadillo>
#include <nlohmann/json.hpp>
//! boost boilerplate
#define BOOST_TEST_MODULE TestRegistration test
#include <boost/test/unit_test.hpp>
//! project headers
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
    config.corr_threshold = 0.9;
    config.n_pair_threshold = 100;
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
    config.n_pair_threshold = 100;
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
