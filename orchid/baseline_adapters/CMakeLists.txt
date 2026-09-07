cmake_minimum_required(VERSION 3.18)

project(orchid_baseline_adapters LANGUAGES CXX)

set(BASELINE "" CACHE STRING "Baseline adapter to build: acorn or navix")
set_property(CACHE BASELINE PROPERTY STRINGS acorn navix)
set(BASELINE_FAISS_SOURCE_DIR "" CACHE PATH "Root of the selected Faiss fork")
set(BASELINE_FAISS_LIBRARY "" CACHE FILEPATH "Path to that fork's libfaiss static or shared library")

if(NOT BASELINE STREQUAL "acorn" AND NOT BASELINE STREQUAL "navix")
  message(FATAL_ERROR "Set -DBASELINE=acorn or -DBASELINE=navix")
endif()
if(NOT EXISTS "${BASELINE_FAISS_SOURCE_DIR}/faiss/Index.h")
  message(FATAL_ERROR "BASELINE_FAISS_SOURCE_DIR must contain faiss/Index.h")
endif()
if(NOT EXISTS "${BASELINE_FAISS_LIBRARY}")
  message(FATAL_ERROR "BASELINE_FAISS_LIBRARY does not exist: ${BASELINE_FAISS_LIBRARY}")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Threads REQUIRED)

set(EXPECTED_ACORN_REVISION "3996dcf90ca2fb58abe46679a13d6a57792274c2")
set(EXPECTED_NAVIX_REVISION "25d563a2dec9087891f19055b8407d4383a26342")
string(TOUPPER "${BASELINE}" BASELINE_UPPER)
set(EXPECTED_REVISION "${EXPECTED_${BASELINE_UPPER}_REVISION}")
set(BASELINE_REVISION "unverified")
find_package(Git QUIET)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${BASELINE_FAISS_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE DETECTED_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE GIT_REVISION_RESULT
  )
  if(GIT_REVISION_RESULT EQUAL 0)
    set(BASELINE_REVISION "${DETECTED_REVISION}")
    if(NOT BASELINE_REVISION STREQUAL EXPECTED_REVISION)
      message(FATAL_ERROR
        "${BASELINE} source is at ${BASELINE_REVISION}; expected ${EXPECTED_REVISION}")
    endif()
  else()
    message(WARNING "Could not verify the baseline source revision")
  endif()
endif()

if(BASELINE STREQUAL "acorn")
  set(ADAPTER_SOURCE src/acorn_adapter.cpp)
  set(ADAPTER_MODULE acorn_adapter)
  find_package(ZLIB REQUIRED)
else()
  set(ADAPTER_SOURCE src/navix_adapter.cpp)
  set(ADAPTER_MODULE navix_adapter)
endif()

pybind11_add_module(${ADAPTER_MODULE} MODULE ${ADAPTER_SOURCE})
target_compile_features(${ADAPTER_MODULE} PRIVATE cxx_std_17)
target_include_directories(${ADAPTER_MODULE} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${BASELINE_FAISS_SOURCE_DIR}"
)
target_compile_definitions(${ADAPTER_MODULE} PRIVATE
  ORCHID_BASELINE_REVISION="${BASELINE_REVISION}"
)
target_link_libraries(${ADAPTER_MODULE} PRIVATE
  "${BASELINE_FAISS_LIBRARY}"
  OpenMP::OpenMP_CXX
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  Threads::Threads
  ${CMAKE_DL_LIBS}
)
if(BASELINE STREQUAL "acorn")
  target_link_libraries(${ADAPTER_MODULE} PRIVATE ZLIB::ZLIB)
endif()

# Keep each fork's Faiss symbols private to its extension. This is important if
# a benchmark runner happens to load both adapters in one Python process.
if(UNIX AND NOT APPLE)
  target_link_options(${ADAPTER_MODULE} PRIVATE "LINKER:--exclude-libs,ALL")
endif()

install(TARGETS ${ADAPTER_MODULE} LIBRARY DESTINATION .)
