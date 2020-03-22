#include <boost/make_shared.hpp>
#include <boost/python.hpp>

#include "correspondences/correspondences.hpp"
#include "registration/registration.hpp"

namespace cor = correspondences;
namespace reg = registration;
namespace boopy = boost::python;

/** @struct PyConfig
 * @brief wrapper `Config`; see "register.hpp" for `Config` definition
 * @var PyConfig::epsilon
 * @var PyConfig::pairwiseDistThreshold
 * @var PyConfig::corrThreshold
 *
 * @note follows PEP-8 naming convention
 */
struct PythonConfig {
    double epsilon, pairwiseDistThreshold, corrThreshold;
    size_t nPairThreshold;
};

/** @struct PythonRegistration
 * @brief python wrapper for wrapper class `PointRegRelaxation`
 */
struct PythonRegistration {
     /** PythonRegistration(sourcePts, targetPts, pyConfig)
      * @brief constructor for optimization wrapper class
      *
      * @param[in] sourcePts distribution of (columnar) source points, as python list
      * @param[in] targetPts distribution of (columnar) target points, as python list
      * @param[in] pyConfig `PyConfig` instance with desired optimization parameters;
      *                     see `PyConfig` and `Config` definitions
      * @param[in] min_corr minimum no. of correspondences to identify between source and
      *                     point sets
      * @return
      *
      * @note the optimization is performed when the constructor is called
      */
    PythonRegistration(boopy::list const & srcPts, boopy::list const & tgtPts,
            PythonConfig const & pyConfig, size_t const & minCorr) {
        arma::mat source_pts(boopy::len(srcPts), boopy::len(srcPts[0]));
        for (size_t i = 0; i < source_pts.n_rows; ++i) {
            for (size_t j = 0; j < source_pts.n_cols; ++j) {
                source_pts(i, j) = boopy::extract<double>(srcPts[i][j]);
            }
        }

        arma::mat target_pts(boopy::len(tgtPts), boopy::len(tgtPts[0]));
        for (size_t i = 0; i < target_pts.n_rows; ++i) {
            for (size_t j = 0; j < target_pts.n_cols; ++j) {
                target_pts(i, j) = boopy::extract<double>(tgtPts[i][j]);
            }
        }

        cor::Config config;
        config.epsilon = pyConfig.epsilon;
        config.pairwise_dist_threshold = pyConfig.pairwiseDistThreshold;
        config.corr_threshold = pyConfig.corrThreshold;
        config.n_pair_threshold = pyConfig.nPairThreshold;

        //! setup main call
        arma::colvec optimum;
        cor::PointRegRelaxation::correspondences_t correspondences;
        arma::mat44 H;
        //! and make the main call
        reg::registration(source_pts, target_pts, config, minCorr, optimum, correspondences, H);
        //! unpack and wrap output
        for (auto const & c : correspondences) {
            auto const key = boopy::make_tuple<int, int>(c.first.first, c.first.second);
            auto const value = c.second;
            correspondences_[key] = value;
        }

        for (size_t i = 0; i < optimum.n_elem; ++i) {
            optimum_.append(optimum(i));
        }

        for (size_t i = 0; i < 4; ++i) {
            boopy::list row;
            for (size_t j = 0; j < 4; ++j) {
                row.append(H(i, j));
            }
            H_.append(row);
        }
    }

    /** PythonRegistration::~PythonRegistration()
     * @brief destructor for python wrapper of optimization wrapper class
     *
     * @param[in]
     * @return
     *
     * @note nothing to do; resources are automatically deallocated
     */
    ~PythonRegistration() { }

    //! public members for consumption on the python-side
    boopy::dict correspondences_;
    boopy::list optimum_;
    boopy::list H_;
};

BOOST_PYTHON_MODULE(pyregistration) {
    PyEval_InitThreads();

    using namespace boost::python;
    /**
     * @brief python module for point set registration using convex relaxation
     *
     * @note Only configuration struct, constructor, and output data are exposed.
     * @note If you want more things exposed, submit a PR!
     */
    //! expose PyConfig
    class_<PythonConfig>("PythonConfig")
        .def_readwrite("epsilon", &PythonConfig::epsilon)
        .def_readwrite("pairwiseDistThreshold", &PythonConfig::pairwiseDistThreshold)
        .def_readwrite("corrThreshold", &PythonConfig::corrThreshold)
        .def_readwrite("nPairThreshold", &PythonConfig::nPairThreshold);

    //! expose PyRegistration - NOTE: no default constructor is defined/exposed
    class_<PythonRegistration>("PythonRegistration", init<boopy::list, boopy::list, PythonConfig, size_t>())
        .def_readonly("optimum", &PythonRegistration::optimum_)
        .def_readonly("correspondences", &PythonRegistration::correspondences_)
        .def_readonly("transform", &PythonRegistration::H_);
}
