# Use the installed SDK; no vendor headers/libraries are downloaded or vendored.
set(SYNAPSEAI_ROOT "" CACHE PATH "Optional SynapseAI SDK prefix")
find_path(SynapseAI_INCLUDE_DIR synapse_api.h
  HINTS "${SYNAPSEAI_ROOT}" ENV SYNAPSE_INCLUDE_DIR
  PATHS /usr /opt/habanalabs
  PATH_SUFFIXES include/habanalabs include)
find_library(SynapseAI_LIBRARY NAMES Synapse
  HINTS "${SYNAPSEAI_ROOT}" ENV BUILD_ROOT_LATEST
  PATHS /usr /opt/habanalabs
  PATH_SUFFIXES lib/habanalabs lib lib64)
find_library(SynapseAI_TPC_LIBRARY NAMES tpc_kernels
  HINTS "${SYNAPSEAI_ROOT}" ENV BUILD_ROOT_LATEST
  PATHS /usr /opt/habanalabs PATH_SUFFIXES lib/habanalabs lib lib64)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SynapseAI REQUIRED_VARS SynapseAI_INCLUDE_DIR SynapseAI_LIBRARY SynapseAI_TPC_LIBRARY)
set(SynapseAI_KERNEL_PATH "${SynapseAI_TPC_LIBRARY}")
if(SynapseAI_FOUND AND NOT TARGET SynapseAI::SynapseAI)
  add_library(SynapseAI::SynapseAI UNKNOWN IMPORTED)
  set_target_properties(SynapseAI::SynapseAI PROPERTIES
    IMPORTED_LOCATION "${SynapseAI_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SynapseAI_INCLUDE_DIR}"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${SynapseAI_INCLUDE_DIR}")
endif()
mark_as_advanced(SynapseAI_INCLUDE_DIR SynapseAI_LIBRARY SynapseAI_TPC_LIBRARY)
