cmake_minimum_required(VERSION 3.23.1)

project(orchid_cagra_adapter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(cuvs REQUIRED)

pybind11_add_module(_cagra_native MODULE src/cagra_adapter.cu)
target_compile_definitions(
  _cagra_native
  PRIVATE
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
    ORCHID_CAGRA_ARTIFACT_REVISION="79c153ffc4ebf4363f522097958f83d7eec19ad4"
)
target_compile_options(
  _cagra_native
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>
)
target_link_libraries(
  _cagra_native
  PRIVATE
    cuvs::cuvs
    raft::raft
    rmm::rmm
    CUDA::cudart
)

set_target_properties(
  _cagra_native
  PROPERTIES
    BUILD_RPATH "$ENV{CONDA_PREFIX}/lib"
    INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
)

install(TARGETS _cagra_native LIBRARY DESTINATION .)
