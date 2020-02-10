//! c/c++ headers
#include <string>
#include <fstream>
#include <streambuf>
//! dependency headers
#include <nlohmann/json.hpp>
//! boost boilerplate
#define BOOST_TEST_MODULE TestCorrespondenceUtils test
#include <boost/test/unit_test.hpp>
//! project headers
#include "correspondences/correspondences.hpp"

namespace cor = correspondences;

using json = nlohmann::json;

/**
 * constants for test
 */
static const std::string data_path("../tests/testdata");  // NOLINT [runtime/string]
static const double FLOAT_TOL(1e-7);

BOOST_AUTO_TEST_CASE( linear_programming_test ) {
    //! load unit test data from json
    std::ifstream ifs(data_path + "/linear-programming.json");
    std::string json_str = std::string((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
    json json_data = json::parse(json_str);

    //! c
    auto const & rows_c = json_data["c"].size();
    size_t i = 0;
    arma::colvec c(rows_c);
    for (auto const & it : json_data["c"]) {
        c(i) = static_cast<double>(it[0]);
        ++i;
    }

    //! A
    auto const & rows_A = json_data["A"].size();
    auto const & cols_A = json_data["A"][0].size();
    i = 0;
    arma::mat A(rows_A, cols_A);
    for (auto const & it : json_data["A"]) {
        size_t j = 0;
        for (auto const & jt : it) {
            A(i, j) = static_cast<double>(jt);
            ++j;
        }
        ++i;
    }

    //! b
    auto const & rows_b = json_data["b"].size();
    i = 0;
    arma::colvec b(rows_b);
    for (auto const & it : json_data["b"]) {
        b(i) = static_cast<double>(it[0]);
        ++i;
    }

    //! lower- and upper-bounds
    auto const lb = static_cast<double>(json_data["lb"]);
    auto const ub = static_cast<double>(json_data["ub"]);

    //! TEST 1: nominal call
    arma::colvec x_opt(rows_c);
    BOOST_CHECK( cor::linear_programming(c, A, b, lb, ub, x_opt) );

    //! solution (from matlab)
    auto const & rows_x_matlab = json_data["Xlp"].size();
    i = 0;
    arma::colvec x_opt_matlab(rows_x_matlab);
    for (auto const & it : json_data["Xlp"]) {
        x_opt_matlab(i) = static_cast<double>(it[0]);
        ++i;
    }

    /**
     * the objective value found is a better measure of solution quality than
     * a value-for-value comparison due to truncated precision on unit test input
     */
    double const obj_val_matlab = arma::dot(c, x_opt_matlab);
    double const obj_val = arma::dot(c, x_opt);
    BOOST_CHECK_SMALL(obj_val - obj_val_matlab, FLOAT_TOL);
}
