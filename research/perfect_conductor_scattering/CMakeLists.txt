project(scattering)
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

find_package (Eigen3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
    message("Eigen Found")
endif ()

set(TARGET_LIBS eigen OpenMP::OpenMP_CXX visual)

list(APPEND TASKS calculate_scattered_field.cpp calculate_esa.cpp)

foreach (TASK IN LISTS TASKS)
    get_filename_component(FILE_NAME ${TASK} NAME_WE)
    set(EXE_NAME scattering_${FILE_NAME})
    add_executable(${EXE_NAME} ${TASK})
    message("TASK: ${EXE_NAME}")
    target_link_libraries(${EXE_NAME} ${PROJECT_NAME} ${TARGET_LIBS})
    target_compile_options(${EXE_NAME} PRIVATE -Wconversion)
    target_include_directories(${EXE_NAME} PUBLIC )
    target_include_directories(${EXE_NAME} PUBLIC "../../src")
    target_compile_options(${EXE_NAME} PUBLIC -flto)
endforeach ()
