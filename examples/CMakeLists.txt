set(DOT_SOURCE_FILE dot.cpp)
set(DOT_TARGET_NAME dot)
set(DOT_EMULATOR_TARGET ${DOT_TARGET_NAME}.fpga_emu)
set(DOT_FPGA_TARGET ${DOT_TARGET_NAME}.fpga)

set(UNBLOCKED_CHOLESKY_SOURCE_FILE unblocked_cholesky.cpp)
set(UNBLOCKED_CHOLESKY_TARGET_NAME unblocked_cholesky)
set(UNBLOCKED_CHOLESKY_EMULATOR_TARGET ${UNBLOCKED_CHOLESKY_TARGET_NAME}.fpga_emu)
set(UNBLOCKED_CHOLESKY_FPGA_TARGET ${UNBLOCKED_CHOLESKY_TARGET_NAME}.fpga)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

# Flags
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga")
set(DOT_HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET} -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsno-interleaving=default -Xsprofile ${USER_HARDWARE_FLAGS}")
set(UNBLOCKED_CHOLESKY_HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET} -Xshardware -Xstarget=${FPGA_DEVICE} -Xsffp-reassociate -Xsprofile ${USER_HARDWARE_FLAGS}")

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR demo.cpp oplib.a -o demo.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o demo.o -c demo.cpp
#    [link]    icpx -fsycl -fintelfpga demo.cpp.o -o demo.fpga_emu oplib.a
add_executable(${DOT_EMULATOR_TARGET} ${DOT_SOURCE_FILE})
target_include_directories(${DOT_EMULATOR_TARGET} PRIVATE ../include)
add_custom_target(dot_emu DEPENDS ${DOT_EMULATOR_TARGET})
target_link_libraries(${DOT_EMULATOR_TARGET} library_archive)
set_target_properties(${DOT_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${DOT_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

add_executable(${UNBLOCKED_CHOLESKY_EMULATOR_TARGET} ${UNBLOCKED_CHOLESKY_SOURCE_FILE})
target_include_directories(${UNBLOCKED_CHOLESKY_EMULATOR_TARGET} PRIVATE ../include)
add_custom_target(unblocked_cholesky_emu DEPENDS ${UNBLOCKED_CHOLESKY_EMULATOR_TARGET})
target_link_libraries(${UNBLOCKED_CHOLESKY_EMULATOR_TARGET} library_archive)
set_target_properties(${UNBLOCKED_CHOLESKY_EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${UNBLOCKED_CHOLESKY_EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> demo.cpp oplib.a -o demo.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o demo.cpp.o -c demo.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> demo.cpp.o -o demo.fpga oplib.a
add_executable(${DOT_FPGA_TARGET} ${DOT_SOURCE_FILE})
target_include_directories(${DOT_FPGA_TARGET} PRIVATE ../include)
add_custom_target(dot DEPENDS ${DOT_FPGA_TARGET})
target_link_libraries(${DOT_FPGA_TARGET} library_archive)
set_target_properties(${DOT_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${DOT_FPGA_TARGET} PROPERTIES LINK_FLAGS "${DOT_HARDWARE_LINK_FLAGS}")

add_executable(${UNBLOCKED_CHOLESKY_FPGA_TARGET} ${UNBLOCKED_CHOLESKY_SOURCE_FILE})
target_include_directories(${UNBLOCKED_CHOLESKY_FPGA_TARGET} PRIVATE ../include)
add_custom_target(unblocked_cholesky DEPENDS ${UNBLOCKED_CHOLESKY_FPGA_TARGET})
target_link_libraries(${UNBLOCKED_CHOLESKY_FPGA_TARGET} library_archive)
set_target_properties(${UNBLOCKED_CHOLESKY_FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${UNBLOCKED_CHOLESKY_FPGA_TARGET} PROPERTIES LINK_FLAGS "${UNBLOCKED_CHOLESKY_HARDWARE_LINK_FLAGS}")
