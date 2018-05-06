cmake_minimum_required(VERSION 3.11)

project(linear_regression)

find_file(Eigen_version Macros.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen/Eigen/src/Core/util)

if (Eigen_version)

	message(STATUS "Eigen found")

else()

	find_package(Git REQUIRED)

	set(eigen_url https://github.com/eigenteam/eigen-git-mirror.git)

	execute_process(COMMAND git clone ${eigen_url} external/eigen
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)

set (PROJECT_SOURCES
    main.cpp
)

add_executable(linear_regression ${PROJECT_SOURCES})

set_property(TARGET linear_regression PROPERTY CXX_STANDARD 14)
set_property(TARGET linear_regression PROPERTY CXX_STANDARD_REQUIRED ON)
