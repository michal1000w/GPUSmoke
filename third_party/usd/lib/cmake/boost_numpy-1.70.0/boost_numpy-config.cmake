# Generated by Boost 1.70.0

if(TARGET Boost::numpy)
  return()
endif()

message(STATUS "Found boost_numpy ${boost_numpy_VERSION} at ${boost_numpy_DIR}")

# Compute the include and library directories relative to this file.
get_filename_component(_BOOST_CMAKEDIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)
get_filename_component(_BOOST_INCLUDEDIR "${_BOOST_CMAKEDIR}/../../include/boost-1_70/" ABSOLUTE)
get_filename_component(_BOOST_LIBDIR "${_BOOST_CMAKEDIR}/../" ABSOLUTE)

# Create imported target Boost::numpy
add_library(Boost::numpy UNKNOWN IMPORTED)

set_target_properties(Boost::numpy PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_BOOST_INCLUDEDIR}"
  INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB"
)

include(${CMAKE_CURRENT_LIST_DIR}/../BoostDetectToolset-1.70.0.cmake)

if(Boost_DEBUG)
  message(STATUS "Scanning ${CMAKE_CURRENT_LIST_DIR}/libboost_numpy-variant*.cmake")
endif()

file(GLOB __boost_variants "${CMAKE_CURRENT_LIST_DIR}/libboost_numpy-variant*.cmake")

macro(_BOOST_SKIPPED fname reason)
  if(Boost_DEBUG)
    message(STATUS "  ... skipped ${fname} (${reason})")
  endif()
  list(APPEND __boost_skipped "${fname} (${reason})")
endmacro()

foreach(f IN LISTS __boost_variants)
  if(Boost_DEBUG)
    message(STATUS "  Including ${f}")
  endif()
  include(${f})
endforeach()

unset(_BOOST_LIBDIR)
unset(_BOOST_INCLUDEDIR)
unset(_BOOST_CMAKEDIR)

get_target_property(__boost_configs Boost::numpy IMPORTED_CONFIGURATIONS)

if(__boost_variants AND NOT __boost_configs)
  message(STATUS "No suitable boost_numpy variant has been identified!")
  if(NOT Boost_DEBUG)
    foreach(s IN LISTS __boost_skipped)
      message(STATUS "  ${s}")
    endforeach()
  endif()
  set(boost_numpy_FOUND 0)
  set(boost_numpy_NOT_FOUND_MESSAGE "No suitable build variant has been found.")
  unset(__boost_skipped)
  unset(__boost_configs)
  unset(__boost_variants)
  unset(_BOOST_NUMPY_DEPS)
  return()
endif()

unset(__boost_skipped)
unset(__boost_configs)
unset(__boost_variants)

if(_BOOST_NUMPY_DEPS)
  list(REMOVE_DUPLICATES _BOOST_NUMPY_DEPS)
  message(STATUS "Adding boost_numpy dependencies: ${_BOOST_NUMPY_DEPS}")
endif()

foreach(dep_boost_numpy IN LISTS _BOOST_NUMPY_DEPS)
  set(_BOOST_QUIET)
  if(boost_numpy_FIND_QUIETLY)
    set(_BOOST_QUIET QUIET)
  endif()
  set(_BOOST_REQUIRED)
  if(boost_numpy_FIND_REQUIRED)
    set(_BOOST_REQUIRED REQUIRED)
  endif()
  get_filename_component(_BOOST_CMAKEDIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)
  find_package(boost_${dep_boost_numpy} 1.70.0 EXACT CONFIG ${_BOOST_REQUIRED} ${_BOOST_QUIET} HINTS ${_BOOST_CMAKEDIR})
  set_property(TARGET Boost::numpy APPEND PROPERTY INTERFACE_LINK_LIBRARIES Boost::${dep_boost_numpy})
  unset(_BOOST_QUIET)
  unset(_BOOST_REQUIRED)
  unset(_BOOST_CMAKEDIR)
  if(NOT boost_${dep_boost_numpy}_FOUND)
    set(boost_numpy_FOUND 0)
    set(boost_numpy_NOT_FOUND_MESSAGE "A required dependency, boost_${dep_boost_numpy}, has not been found.")
    unset(_BOOST_NUMPY_DEPS)
    return()
  endif()
endforeach()

unset(_BOOST_NUMPY_DEPS)
