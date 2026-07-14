cmake_minimum_required(VERSION 3.22)
project(vol_die)
message("VOLUME REGULAR DIELECTRICS")
set(CMAKE_CXX_STANDARD 23)

# Some flags related to visualization
set(EMW_USE_VTK_VISUALIZATION OFF)
set(EMW_WITH_TESTS OFF)
set(EMW_WITH_BENCHMARKS OFF)
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

list(APPEND TASKS
        strong_scalability_test.cpp
        cluster_sphere.cpp
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
