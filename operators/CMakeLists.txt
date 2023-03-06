# Operator library source files
set(OPERATOR_SPEC dot_prod.cpp)
set(SYCL_SOURCE_OBJECT dot_prod.o)
set(LIBRARY_ARCHIVE oplib.a)

###############################################################################
### Generate Library
###############################################################################

# Create source object
# This executes:
# fpga_crossgen op.cpp --target sycl -o op.o
add_custom_target(
    create_source_object
    COMMAND fpga_crossgen ${CMAKE_CURRENT_SOURCE_DIR}/${OPERATOR_SPEC} --target sycl -o ${SYCL_SOURCE_OBJECT}
)

# Create library archive
# This executes:
# fpga_libtool op.o --target sycl --create oplib.a
add_custom_target(create_library_archive DEPENDS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE})
add_custom_command(OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE}
                   COMMAND fpga_libtool ${SYCL_SOURCE_OBJECT} --target sycl --create ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE}
                   DEPENDS create_source_object)

# Tell CMake to recognize our custom library
add_library(library_archive STATIC IMPORTED GLOBAL)
add_dependencies(library_archive create_library_archive)
set_target_properties(library_archive PROPERTIES IMPORTED_LOCATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LIBRARY_ARCHIVE})
