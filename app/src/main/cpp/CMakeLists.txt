cmake_minimum_required(VERSION 3.22.1)
project("testgame")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# Move these BEFORE add_library so they apply to the target
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")

add_library(${CMAKE_PROJECT_NAME} SHARED native-lib.cpp)

find_library(log-lib log)
find_library(android-lib android)
find_library(egl-lib EGL)
find_library(gles-lib GLESv3)

target_link_libraries(
        testgame
        ${log-lib}
        ${android-lib}
        ${egl-lib}
        ${gles-lib}
        c++_shared   # Use c++_shared instead of both c++abi + c++_static
)