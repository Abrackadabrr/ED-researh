cmake_minimum_required(VERSION 3.22)
project(vol_die)
message("VOLUME REGULAR DIELECTRICS")
set(CMAKE_CXX_STANDARD 23)

# Some flags related to visualization
set(EMW_USE_VTK_VISUALIZATION OFF)
set(EMW_WITH_TESTS OFF)
set(EMW_WITH_BENCHMARKS ON)
set(EMW_WITH_TASKS OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Electromagnetic-Waves-Scattering
        ${CMAKE_CURRENT_BINARY_DIR}/Electromagnetic-Waves-Scattering)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../scattnlay
        ${CMAKE_CURRENT_BINARY_DIR}/scattnlay)

# Setting some flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
message(${CMAKE_CXX_FLAGS_RELEASE})

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP Found")
endif ()

find_package(Eigen3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
    message("Eigen Found")
endif ()

set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/yaml-cpp
        ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp EXCLUDE_FROM_ALL)

add_library(ed_research_yaml_utility STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utility/YamlParser.cpp)
target_include_directories(ed_research_yaml_utility PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(ed_research_yaml_utility PUBLIC yaml-cpp::yaml-cpp)

list(APPEND TASKS
        cluster_scalability_tests/strong_scalability_test.cpp
        cluster_scalability_tests/strong_scalability_fourier_test.cpp
        cluster_sphere.cpp
        cluster_sphere_circulant.cpp
        cluster_sphere_schur.cpp
        cube_circulant_preconditioner.cpp
        cluster_sphere_extinction.cpp
        ../plasmon/sphere_plasmon_regime.cpp
)

foreach (TASK IN LISTS TASKS)
    get_filename_component(FILE_NAME ${TASK} NAME_WE)
    set(EXE_NAME ${PROJECT_NAME}_${FILE_NAME})
    add_executable(${EXE_NAME} ${TASK})
    message("TASK: ${EXE_NAME}")
    target_sources(${EXE_NAME} PRIVATE $<TARGET_OBJECTS:nmie_obj_dp>)
    target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../scattnlay/src)
    target_link_libraries(${EXE_NAME} PUBLIC emw) # other libs will simultaneously be linked as emw is linked
    target_compile_definitions(${EXE_NAME} PUBLIC "OMP_NESTED=ON")
endforeach ()

target_sources(vol_die_cluster_sphere_extinction PRIVATE
        ClusterSphereExtinctionConfig.cpp)
target_link_libraries(vol_die_cluster_sphere_extinction PRIVATE
        ed_research_yaml_utility)
target_compile_definitions(vol_die_cluster_sphere_extinction PRIVATE
        CLUSTER_SPHERE_EXTINCTION_SOURCE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/cluster_sphere_extinction.yaml")
configure_file(cluster_sphere_extinction.yaml cluster_sphere_extinction.yaml COPYONLY)

target_sources(vol_die_sphere_plasmon_regime PRIVATE
        analytical_solution/SphereMieAnalytical.cpp
        ../plasmon/SpherePlasmonRegimeConfig.cpp)
target_link_libraries(vol_die_sphere_plasmon_regime PRIVATE
        ed_research_yaml_utility
        OpenMP::OpenMP_CXX)
target_compile_definitions(vol_die_sphere_plasmon_regime PRIVATE
        SPHERE_PLASMON_REGIME_SOURCE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/../plasmon/sphere_plasmon_regime.yaml")
configure_file(../plasmon/sphere_plasmon_regime.yaml sphere_plasmon_regime.yaml COPYONLY)

# Both sphere executables use the same implementation of the analytical Mie solution.
target_sources(vol_die_cluster_sphere PRIVATE analytical_solution/SphereMieAnalytical.cpp)
target_sources(vol_die_cluster_sphere_circulant PRIVATE analytical_solution/SphereMieAnalytical.cpp)

# Eigen can delegate the dense complex Schur decomposition to the optimized
# LAPACKE zgees implementation. Keep Eigen's built-in implementation as a fallback.
find_library(LAPACKE_LIBRARY NAMES lapacke)
if (LAPACKE_LIBRARY)
    target_compile_definitions(vol_die_cluster_sphere_schur PRIVATE EIGEN_USE_LAPACKE)
    target_link_libraries(vol_die_cluster_sphere_schur PRIVATE ${LAPACKE_LIBRARY})
    target_compile_definitions(vol_die_cube_circulant_preconditioner PRIVATE EIGEN_USE_LAPACKE)
    target_link_libraries(vol_die_cube_circulant_preconditioner PRIVATE ${LAPACKE_LIBRARY})
else ()
    message(WARNING "LAPACKE was not found; Schur tasks will use Eigen's built-in implementation")
endif ()
