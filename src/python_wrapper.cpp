#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <armadillo>

#include "register.hpp"

namespace boopy = boost::python;

struct PythonConfig {
    double epsilon, pairwise_dist_threshold;
};

struct PythonRegistration {
    PythonRegistration(boopy::list const & src_pts, boopy::list const & tgt_pts,
            PythonConfig const & py_config) {
        arma::mat source_pts(boopy::len(src_pts), boopy::len(src_pts[0]));
        for (size_t i = 0; i < source_pts.n_rows; ++i) {
            for (size_t j = 0; j < source_pts.n_cols; ++j) {
                source_pts(i, j) = boopy::extract<double>(src_pts[i][j]);
            }
        }

        arma::mat target_pts(boopy::len(tgt_pts), boopy::len(tgt_pts[0]));
        for (size_t i = 0; i < target_pts.n_rows; ++i) {
            for (size_t j = 0; j < target_pts.n_cols; ++j) {
                target_pts(i, j) = boopy::extract<double>(tgt_pts[i][j]);
            }
        }

        Config config;
        config.epsilon = py_config.epsilon;
        config.pairwise_dist_threshold = py_config.pairwise_dist_threshold;

        reg_ = boost::make_shared<PointRegRelaxation>(source_pts, target_pts, config);
    }

    ~PythonRegistration() { }

    boopy::list findOptimumVector() const noexcept {
        boopy::list output;
        arma::colvec solution;
        auto status = reg_->find_optimum(solution);
        std::cout << status << std::endl;
        for (size_t i = 0; i < solution.n_elem; ++i) {
            output.append(solution[i]);
        }
        return output;
    }

    boost::shared_ptr<PointRegRelaxation> reg_;
};

BOOST_PYTHON_MODULE(pyregistration) {
    PyEval_InitThreads();
    
    using namespace boost::python;

    class_<PythonConfig>("PythonConfig")
        .def_readwrite("epsilon", &PythonConfig::epsilon)
        .def_readwrite("pairwise_dist_threshold", &PythonConfig::pairwise_dist_threshold);
    
    class_<PythonRegistration>("PythonRegistration", init<boopy::list, boopy::list, PythonConfig>())
        .def("findOptimumVector", &PythonRegistration::findOptimumVector);
}
