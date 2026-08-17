cmake_minimum_required(VERSION 3.22)
project(plasmon_jvie)
message("PLASMON JVIE")
set(CMAKE_CXX_STANDARD 23)

# Some flags related to visualization
set(EMW_USE_VTK_VISUALIZATION ON)
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

find_package(OpenMP REQUIRED)
message("OpenMP Found")

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
        sphere_plasmon_regime.cpp
        sphere_frequency_sweep.cpp
        long_plate_dipole_scattering.cpp
        pillar_array_scattering.cpp
)

set(TARGET_LIBS
        emw
        emw_visual_vtk
        ed_research_yaml_utility
        OpenMP::OpenMP_CXX
)

foreach (TASK IN LISTS TASKS)
    get_filename_component(FILE_NAME ${TASK} NAME_WE)
    set(EXE_NAME ${PROJECT_NAME}_${FILE_NAME})

    add_executable(${EXE_NAME} ${TASK})
    message("TASK: ${EXE_NAME}")

    if (FILE_NAME STREQUAL "sphere_plasmon_regime")
        target_sources(${EXE_NAME} PRIVATE
                configs/SpherePlasmonRegimeConfig.cpp
                ../cluster_dielectric/analytical_solution/SphereMieAnalytical.cpp
                $<TARGET_OBJECTS:nmie_obj_dp>
        )
        target_compile_definitions(${EXE_NAME} PRIVATE
                SPHERE_PLASMON_REGIME_SOURCE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/sphere_plasmon_regime.yaml"
        )
    elseif (FILE_NAME STREQUAL "sphere_frequency_sweep")
        target_sources(${EXE_NAME} PRIVATE configs/SphereFrequencySweepConfig.cpp)
        target_compile_definitions(${EXE_NAME} PRIVATE
                SPHERE_FREQUENCY_SWEEP_SOURCE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/sphere_frequency_sweep.yaml"
        )
    elseif (FILE_NAME STREQUAL "long_plate_dipole_scattering")
        target_sources(${EXE_NAME} PRIVATE configs/LongPlateDipoleScatteringConfig.cpp)
        target_compile_definitions(${EXE_NAME} PRIVATE
                LONG_PLATE_DIPOLE_SCATTERING_SOURCE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/long_plate_dipole_scattering.yaml"
        )
    elseif (FILE_NAME STREQUAL "pillar_array_scattering")
        target_sources(${EXE_NAME} PRIVATE
                configs/PillarArrayScatteringConfig.cpp
                PillarArrayGeometry.cpp
        )
        target_compile_definitions(${EXE_NAME} PRIVATE
                PILLAR_ARRAY_SCATTERING_SOURCE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/pillar_array_scattering.yaml"
        )
    endif ()

    target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../scattnlay/src)
    target_link_libraries(${EXE_NAME} PRIVATE ${TARGET_LIBS})
    target_compile_definitions(${EXE_NAME} PRIVATE OMP_NESTED=ON)
endforeach ()

configure_file(sphere_plasmon_regime.yaml sphere_plasmon_regime.yaml COPYONLY)
configure_file(sphere_frequency_sweep.yaml sphere_frequency_sweep.yaml COPYONLY)
configure_file(long_plate_dipole_scattering.yaml long_plate_dipole_scattering.yaml COPYONLY)
configure_file(pillar_array_scattering.yaml pillar_array_scattering.yaml COPYONLY)
configure_file(pillar_array_geometry.yaml pillar_array_geometry.yaml COPYONLY)
