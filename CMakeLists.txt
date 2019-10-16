cmake_minimum_required(VERSION 2.8.10)
project(registration C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
find_package(nlohmann_json 3.7.0 REQUIRED)
find_package(CppAD REQUIRED)
find_package(IPOPT REQUIRED)
# For Boost testing framework (will have no effect on non-testing executables)
add_definitions(-DBOOST_TEST_DYN_LINK)

include_directories(
    ${ARMADILLO_INCLUDE_DIRS}
    ${CPPAD_INCLUDE_DIRS}
    ${IPOPT_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_INSTALL_DIR}
    "include"
)

# compiler flags - NOTE(jwd) - ORTOOLS dependencies will throw pedantic warnings
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

add_library(registration src/register.cpp)

#enable_testing()

#add_executable(register-test tests/register_ut.cpp)
#target_link_libraries(sdrsac-test ${Boost_FILESYSTEM_LIBRARY}
#    ${Boost_SYSTEM_LIBRARY}
#    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
#    ${ARMADILLO_LIBRARIES}
#)
#add_test(SDRSACTest sdrsac-test)
