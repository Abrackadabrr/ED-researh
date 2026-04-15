project(waveguide)
cmake_minimum_required(VERSION 3.22)
message("RUPOR STUDY BUILDING\n")
set(CMAKE_CXX_STANDARD 23)

add_subdirectory(../../src src)
add_subdirectory(../../third_party third_party)
add_subdirectory(../../visualisation visualisation)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP Found")
endif ()

find_package(Eigen3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
    message("Eigen Found")
endif ()

find_package(TBB REQUIRED COMPONENTS tbb)

set(TARGET_LIBS eigen visual tbb)  #  + OpenMP::OpenMP_CXX

list(APPEND TASKS infinite_waveguide.cpp mesh_convergence.cpp)

foreach (TASK IN LISTS TASKS)
    get_filename_component(FILE_NAME ${TASK} NAME_WE)
    set(EXE_NAME ${FILE_NAME})
    add_executable(${EXE_NAME} ${TASK})
    message("TASK: ${EXE_NAME}")
    target_link_libraries(${EXE_NAME} ${PROJECT_NAME} ${TARGET_LIBS})
    target_compile_options(${EXE_NAME} PRIVATE -Wconversion)
    target_include_directories(${EXE_NAME} PUBLIC "../../src")
endforeach ()
