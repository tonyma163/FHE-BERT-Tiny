cmake_minimum_required(VERSION 3.15)  # Updated to a more recent version for better features

project(FHE-BERT-Tiny)

# ------------------------------------------
# 1. Specify the C++ Standard
# ------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------
# 2. Option to Build Static Libraries
# ------------------------------------------
option(BUILD_STATIC "Set to ON to link OpenFHE statically" OFF)

# ------------------------------------------
# 3. Compiler Settings for Apple (Optional)
# ------------------------------------------
if(APPLE)
    set(CMAKE_C_COMPILER "/Library/Developer/CommandLineTools/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/Library/Developer/CommandLineTools/usr/bin/g++")
endif()

# ------------------------------------------
# 4. Specify OpenFHE Include and Library Directories
# ------------------------------------------
# OpenFHE include directory
set(OpenFHE_INCLUDE_DIR "/usr/local/include/openfhe")

# OpenFHE library directory
set(OpenFHE_LIB_DIR "/usr/local/lib")

# ------------------------------------------
# 5. Define the Executable
# ------------------------------------------

#add_executable(FHE-BERT-Tiny
#    src/main.cpp
#    src/FHEController.cpp
#    src/FHEController.h
#    src/Utils.h
#)

add_executable(FHE-BERT-Tiny
    newsrc/main.cpp
    newsrc/FHEController.cpp
    newsrc/FHEController.h
    newsrc/Utils.h
)
# ------------------------------------------
# 6. Include Directories Scoped to the Target
# ------------------------------------------
target_include_directories(FHE-BERT-Tiny PRIVATE
    ${OpenFHE_INCLUDE_DIR}
    ${OpenFHE_INCLUDE_DIR}/third-party/include
    ${OpenFHE_INCLUDE_DIR}/core
    ${OpenFHE_INCLUDE_DIR}/pke
    ${OpenFHE_INCLUDE_DIR}/binfhe
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Your project's include directory
)

# ------------------------------------------
# 7. Link Directories
# ------------------------------------------
# Inform CMake where to find the OpenFHE libraries
link_directories(${OpenFHE_LIB_DIR})

# ------------------------------------------
# 8. Find Required Packages
# ------------------------------------------
# OpenMP
find_package(OpenMP REQUIRED)

# GSL
find_package(GSL REQUIRED)

# ------------------------------------------
# 9. Link Libraries Based on BUILD_STATIC Option
# ------------------------------------------
if(BUILD_STATIC)
    set(OpenFHE_LIBS
        OPENFHEbinfhe
        OPENFHEcore
        OPENFHEpke
        # Add other OpenFHE static libraries as needed
    )
    # Link with static OpenFHE libraries
    target_link_libraries(FHE-BERT-Tiny PRIVATE
        ${OpenFHE_LIBS}
        GSL::gsl
        OpenMP::OpenMP_CXX
    )
    # Set linker flags for static linking if required
    target_compile_options(FHE-BERT-Tiny PRIVATE ${OpenFHE_CXX_FLAGS})
    target_link_options(FHE-BERT-Tiny PRIVATE ${OpenFHE_EXE_LINKER_FLAGS} -static)
else()
    set(OpenFHE_LIBS
        OPENFHEbinfhe
        OPENFHEcore
        OPENFHEpke
        # Add other OpenFHE shared libraries as needed
    )
    target_link_libraries(FHE-BERT-Tiny PRIVATE
        ${OpenFHE_LIBS}
        GSL::gsl
        OpenMP::OpenMP_CXX
    )
    set_target_properties(FHE-BERT-Tiny PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "/usr/local/lib"
    )
endif()

# ------------------------------------------
# 10. Link stdc++fs If Necessary
# ------------------------------------------
# Only needed for GCC versions less than 9
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Extract GCC version
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Compare versions
    if(GCC_VERSION VERSION_LESS "9.0")
        target_link_libraries(FHE-BERT-Tiny PRIVATE stdc++fs)
    endif()
endif()
