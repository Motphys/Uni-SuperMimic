cmake_minimum_required(VERSION 3.16)
project(uni_g1_deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime: set ONNXRUNTIME_ROOT to the unpacked onnxruntime release
# (contains include/ and lib/). Default: .deps/onnxruntime under cpp/.
if(NOT DEFINED ONNXRUNTIME_ROOT)
  if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
  else()
    get_filename_component(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.deps/onnxruntime" ABSOLUTE)
  endif()
endif()

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)

add_library(uni_deploy STATIC
  src/uni_config.cpp
  src/uni_deploy_core.cpp
  src/uni_fsm.cpp
  src/uni_policy.cpp
)
target_include_directories(uni_deploy PUBLIC src "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(uni_deploy PUBLIC ${ONNXRUNTIME_LIB})

add_executable(check_numeric src/check_numeric.cpp)
target_link_libraries(check_numeric PRIVATE uni_deploy)

option(UNI_BUILD_ROBOT "Build the real-robot controller (needs unitree_sdk2)" ON)
if(UNI_BUILD_ROBOT)
  # unitree_sdk2 discovery, in order:
  #   1. -DUNITREE_SDK2_ROOT=... or $UNITREE_SDK2_ROOT
  #   2. system install under /usr/local
  #   3. $HOME/unitree_sdk2
  #   4. sibling checkout next to this repository (../../unitree_sdk2)
  set(_sdk2_candidates "")
  if(DEFINED UNITREE_SDK2_ROOT)
    list(APPEND _sdk2_candidates "${UNITREE_SDK2_ROOT}")
  endif()
  if(DEFINED ENV{UNITREE_SDK2_ROOT})
    list(APPEND _sdk2_candidates "$ENV{UNITREE_SDK2_ROOT}")
  endif()
  list(APPEND _sdk2_candidates "/usr/local" "$ENV{HOME}/unitree_sdk2"
       "${CMAKE_CURRENT_SOURCE_DIR}/../../../unitree_sdk2"
       "/home/unitree/ws/FlashMimic_g1_deploy/unitree_sdk2")
  set(UNITREE_SDK2_ROOT "")
  set(UNITREE_SDK2_LIB "")
  foreach(cand IN LISTS _sdk2_candidates)
    if(EXISTS "${cand}/include/unitree/robot/channel/channel_publisher.hpp")
      if(EXISTS "${cand}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a")
        set(UNITREE_SDK2_ROOT "${cand}")
        set(UNITREE_SDK2_LIB "${cand}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a")
        break()
      elseif(EXISTS "${cand}/lib/libunitree_sdk2.a")
        set(UNITREE_SDK2_ROOT "${cand}")
        set(UNITREE_SDK2_LIB "${cand}/lib/libunitree_sdk2.a")
        break()
      endif()
    endif()
  endforeach()
  if(UNITREE_SDK2_ROOT)
    message(STATUS "unitree_sdk2: ${UNITREE_SDK2_ROOT} (${UNITREE_SDK2_LIB})")
    add_executable(g1_uni_ctrl src/g1_uni_ctrl.cpp)
    target_include_directories(g1_uni_ctrl PRIVATE
      "${UNITREE_SDK2_ROOT}/include"
      /usr/local/include/ddscxx
      /usr/local/include/iceoryx/v2.0.2
    )
    target_link_libraries(g1_uni_ctrl PRIVATE uni_deploy
      "${UNITREE_SDK2_LIB}" ddsc ddscxx rt pthread)
  else()
    message(WARNING "unitree_sdk2 not found: skipping g1_uni_ctrl (check_numeric still builds). "
            "Set UNITREE_SDK2_ROOT to the sdk2 root (contains include/ and lib/<arch>/).")
  endif()
endif()
