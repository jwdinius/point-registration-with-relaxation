#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <armadillo>

#include "register.hpp"

namespace boopy = boost::python;

struct PythonConfig {
    double epsilon, pairwise_dist_threshold, corr_threshold;
    bool do_warm_start;
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
        config.corr_threshold = py_config.corr_threshold;
        config.do_warm_start = py_config.do_warm_start;

        arma::colvec optimum;
        PointRegRelaxation::correspondences_t correspondences;
        arma::mat44 H;
        registration(source_pts, target_pts, config, optimum, correspondences, H);

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

    ~PythonRegistration() { }
 
    boopy::dict correspondences_;
    boopy::list optimum_;
    boopy::list H_;
};

BOOST_PYTHON_MODULE(pyregistration) {
    PyEval_InitThreads();
    
    using namespace boost::python;

    class_<PythonConfig>("PythonConfig")
        .def_readwrite("epsilon", &PythonConfig::epsilon)
        .def_readwrite("pairwise_dist_threshold", &PythonConfig::pairwise_dist_threshold)
        .def_readwrite("corr_threshold", &PythonConfig::corr_threshold)
        .def_readwrite("do_warm_start", &PythonConfig::do_warm_start);
    
    class_<PythonRegistration>("PythonRegistration", init<boopy::list, boopy::list, PythonConfig>())
        .def_readonly("optimum", &PythonRegistration::optimum_)
        .def_readonly("correspondences", &PythonRegistration::correspondences_)
        .def_readonly("transform", &PythonRegistration::H_);
}
