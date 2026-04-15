project(preconditioning)
cmake_minimum_required(VERSION 3.22)
message("PRECONDITIONING STUDY BUILDING\n")
set(CMAKE_CXX_STANDARD 23)

add_subdirectory(../../src src)
add_subdirectory(../../third_party trird_party)
add_subdirectory(../../visualisation visualisation)

#find_package(OpenMP)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(EXE_NAME analytical_solution)
add_executable(${EXE_NAME}
        plate/study_preconditioner.cpp)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP Found")
endif ()

find_package (Eigen3 REQUIRED NO_MODULE)
if (EIGEN_FOUND)
    message("Eigen Found")
endif ()

target_link_libraries (${EXE_NAME} eigen)
target_link_libraries(${EXE_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${EXE_NAME} ${PROJECT_NAME} visual)
target_include_directories(${EXE_NAME} PUBLIC )
target_include_directories(${EXE_NAME} PUBLIC "../../src")