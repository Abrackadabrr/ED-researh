project(scattering)
cmake_minimum_required(VERSION 3.22)
message("Perfect conductor scattering BUILDING\n")
set(CMAKE_CXX_STANDARD 23)

# Some flags related to visualization
set(EMW_USE_VTK_VISUALIZATION ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Electromagnetic-Waves-Scattering
        ${CMAKE_CURRENT_BINARY_DIR}/Electromagnetic-Waves-Scattering)

# Setting some flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
message(${CMAKE_CXX_FLAGS_RELEASE})

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP Found")
endif ()

find_package (Eigen3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
    message("Eigen Found")
endif ()

set(TARGET_LIBS emw emw_visual_vtk eigen OpenMP::OpenMP_CXX)

list(APPEND TASKS  calculate_scattered_field.cpp calculate_esa.cpp)

foreach (TASK IN LISTS TASKS)
    get_filename_component(FILE_NAME ${TASK} NAME_WE)
    set(EXE_NAME scattering_${FILE_NAME})
    add_executable(${EXE_NAME} ${TASK})
    message("TASK: ${EXE_NAME}")
    target_link_libraries(${EXE_NAME} ${TARGET_LIBS})
    target_compile_options(${EXE_NAME} PRIVATE -Wconversion)
    target_include_directories(${EXE_NAME} PUBLIC )
    target_include_directories(${EXE_NAME} PUBLIC "../../src")
    target_compile_options(${EXE_NAME} PUBLIC -flto)
endforeach ()
