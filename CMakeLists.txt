cmake_minimum_required(VERSION 2.8.10)
project(registration C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{ORTOOLS_ROOT})
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Armadillo REQUIRED)
find_package(MLPACK REQUIRED)
find_package(Boost COMPONENTS system filesystem unit_test_framework python REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
find_package(nlohmann_json 3.7.0 REQUIRED)
find_package(CppAD REQUIRED)
find_package(IPOPT REQUIRED)
find_package(ortools REQUIRED)

# For Boost testing framework (will have no effect on non-testing executables)
add_definitions(-DBOOST_TEST_DYN_LINK)
add_definitions(-DUSE_GLOP)

include_directories(
    ${ARMADILLO_INCLUDE_DIRS}
    ${MLPACK_INCLUDE_DIRS}
    ${CPPAD_INCLUDE_DIRS}
    ${IPOPT_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_INSTALL_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${ORTOOLS_INCLUDE_DIRS}
    "include"
)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

add_library(registration SHARED src/registration.cpp src/correspondences.cpp)
target_link_libraries(registration ${IPOPT_LIBRARIES} ${ARMADILLO_LIBRARIES} ${MLPACK_LIBRARIES}
    ${CPPAD_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${ORTOOLS_LIBRARIES})

add_library(pyregistration SHARED src/python-wrapper.cpp)
set_target_properties(pyregistration PROPERTIES PREFIX "")
target_link_libraries(pyregistration ${BOOST_LIBRARIES} ${PYTHON_LIBRARIES} registration)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "__all__ = ['pyregistration']")

enable_testing()

add_executable(registration-test tests/registration-unit.cpp)
target_link_libraries(registration-test ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    registration
)
add_test(RegistrationTest registration-test)

add_executable(transform-test tests/transform-unit.cpp)
target_link_libraries(transform-test ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    registration
)
add_test(TransformTest transform-test)
