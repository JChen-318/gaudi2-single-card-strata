# Use the same MIT ggml revision as Strata's CPU experts. Only ggml-base is
# needed for host dequantization; no CPU execution backend or GPU is loaded.
enable_language(C)
option(STRATA_HPU_CPU_EXPERTS "Build original Strata CPU ActQ and native low-bit expert pool" ON)
set(STRATA_HPU_GGML_DIR "" CACHE PATH "Optional llama.cpp checkout at Strata's pinned revision")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(flag IN ITEMS GGML_CPU GGML_CUDA GGML_HIP GGML_SYCL GGML_VULKAN GGML_METAL
    GGML_OPENCL GGML_BLAS GGML_ACCELERATE GGML_RPC GGML_OPENMP GGML_BACKEND_DL GGML_BUILD_TESTS GGML_BUILD_EXAMPLES)
  set(${flag} OFF CACHE BOOL "" FORCE)
endforeach()
if(STRATA_HPU_CPU_EXPERTS)
  set(GGML_CPU ON CACHE BOOL "" FORCE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
  foreach(flag IN ITEMS GGML_SSE42 GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${flag} ON CACHE BOOL "" FORCE)
  endforeach()
  foreach(flag IN ITEMS GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${flag} OFF CACHE BOOL "" FORCE)
  endforeach()
endif()
if(STRATA_HPU_GGML_DIR)
  foreach(pair IN ITEMS
      "ggml/include/ggml.h=12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9"
      "ggml/src/ggml.c=1a634ca1df25db2fbd526cba9bfe11fccd5aa9b31f819c3cac6b630be03f9f51"
      "ggml/src/ggml-quants.c=07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d"
      "ggml/src/ggml-common.h=0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c")
    string(REPLACE "=" ";" fields "${pair}")
    list(GET fields 0 relative)
    list(GET fields 1 expected)
    file(SHA256 "${STRATA_HPU_GGML_DIR}/${relative}" actual)
    if(NOT actual STREQUAL expected)
      message(FATAL_ERROR "STRATA_HPU_GGML_DIR differs from pinned 3cf03257: ${relative}")
    endif()
  endforeach()
  add_subdirectory("${STRATA_HPU_GGML_DIR}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
else()
  include(FetchContent)
  FetchContent_Declare(strata_hpu_ggml
    URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/3cf03257f219afbe7334045ff7c6a06ac68c627d
    URL_HASH SHA256=c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR ggml)
  FetchContent_MakeAvailable(strata_hpu_ggml)
endif()
add_library(strata_hpu_weights STATIC src/weights.cpp src/weight_slice.cpp src/rope_config.cpp src/mtp_weights.cpp src/control_vector.cpp src/q4_requantize.cpp src/expert_scales.cpp src/expert_scale_values.cpp src/native_pack.cpp src/native_pack_weights.cpp src/embedding_input.cpp)
target_include_directories(strata_hpu_weights PUBLIC include "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(strata_hpu_weights PRIVATE ggml-base)
add_executable(strata-hpu-q4-requantize-test tests/q4_requantize.cpp)
target_link_libraries(strata-hpu-q4-requantize-test PRIVATE strata_hpu_weights ggml-base)
target_compile_options(strata-hpu-q4-requantize-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_q4_requantize COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/q4_requantize.py" "$<TARGET_FILE:strata-hpu-q4-requantize-test>")
set_tests_properties(hpu_q4_requantize PROPERTIES LABELS "host" TIMEOUT 120 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
# Routed K-quant packed dot: actual TPC C semantics plus the unmodified pinned
# ggml row decoder as the byte-layout authority. Host only, no device access.
add_executable(strata-hpu-packed-kquant-test tests/packed_kquant.cpp)
target_link_libraries(strata-hpu-packed-kquant-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-kquant-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_kquant COMMAND strata-hpu-packed-kquant-test)
set_tests_properties(hpu_packed_kquant PROPERTIES LABELS "host" TIMEOUT 180)
# Q3_K superblock packed dot: 16-value sub-block scales, 3-bit hmask/qs planes and a
# signed super-block d, checked against the unmodified pinned ggml Q3_K row decoder
# and the production shared-expert flow. Host only, no device access.
add_executable(strata-hpu-packed-q3k-test tests/packed_q3k.cpp)
target_link_libraries(strata-hpu-packed-q3k-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-q3k-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_q3k COMMAND strata-hpu-packed-q3k-test)
set_tests_properties(hpu_packed_q3k PROPERTIES LABELS "host" TIMEOUT 180)
# Q6_K superblock packed dot: signed int8 sub-block scales, a 4+2-bit code split
# across ql/qh and one signed super-block d, checked against the unmodified pinned
# ggml Q6_K row decoder and the production shared-expert flow. Host only.
add_executable(strata-hpu-packed-q6k-test tests/packed_q6k.cpp)
target_link_libraries(strata-hpu-packed-q6k-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-q6k-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_q6k COMMAND strata-hpu-packed-q6k-test)
set_tests_properties(hpu_packed_q6k PROPERTIES LABELS "host" TIMEOUT 180)
# IQ4_NL (GGML type 20) nonlinear codebook packed dot: four-bit indices into the
# pinned iq4nl table with one half d per 32-value block, checked against an
# independent codebook oracle and the unmodified pinned ggml IQ4_NL row decoder.
# Table-index counterexamples are mandatory here. Host only.
add_executable(strata-hpu-packed-iq4nl-test tests/packed_iq4nl.cpp)
target_link_libraries(strata-hpu-packed-iq4nl-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq4nl-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq4nl COMMAND strata-hpu-packed-iq4nl-test)
set_tests_properties(hpu_packed_iq4nl PROPERTIES LABELS "host" TIMEOUT 180)
# IQ4_XS (GGML type 23) codebook superblock dot: eight 32-value groups with a signed
# 6-bit (raw-32) scale and a per-group d8, checked against an independent logical
# codebook oracle, a separately written scale reader and the unmodified pinned ggml
# IQ4_XS row decoder. Host only.
add_executable(strata-hpu-packed-iq4xs-test tests/packed_iq4xs.cpp)
target_link_libraries(strata-hpu-packed-iq4xs-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq4xs-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq4xs COMMAND strata-hpu-packed-iq4xs-test)
set_tests_properties(hpu_packed_iq4xs PROPERTIES LABELS "host" TIMEOUT 180)
# Packed-format coverage audit: our policy, native_mmvq's pinned block table and ggml's
# runtime traits must agree for every integer-dot type, and every upstream-admitted type
# we still decode must be named and charged as a full BF16 expert. Host only.
add_executable(strata-hpu-packed-coverage-test tests/packed_coverage.cpp)
target_link_libraries(strata-hpu-packed-coverage-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-coverage-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_coverage COMMAND strata-hpu-packed-coverage-test)
set_tests_properties(hpu_packed_coverage PROPERTIES LABELS "host" TIMEOUT 120)
# The concatenated IQ codebook blob that the packed kernel now reads as a read-only input tensor.
# A TPC kernel only gets 16 KiB of static scalar data while the six grids need 25,600 bytes, so the
# tables left the kernel; this guards that the byte form still carries them exactly on Linux: offsets
# derived from the generated tables, adjacency proven, and the codebook-owning type set asserted as a
# whole rather than per format.
add_executable(strata-hpu-packed-codebooks-test tests/packed_codebooks.cpp)
# No library links: the blob is header-only, but the target still needs the backend headers,
# which the neighbouring packed tests otherwise get transitively from their libraries.
target_include_directories(strata-hpu-packed-codebooks-test PRIVATE include)
target_compile_options(strata-hpu-packed-codebooks-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_codebooks COMMAND strata-hpu-packed-codebooks-test)
set_tests_properties(hpu_packed_codebooks PROPERTIES LABELS "host" TIMEOUT 60)
# --prefill-until: the port's clamp must equal the pinned upstream expression over the whole
# swept domain, not just the cases that look interesting. Host only.
add_executable(strata-hpu-prefill-until-test tests/prefill_until.cpp)
target_include_directories(strata-hpu-prefill-until-test PRIVATE include ../include)
# generation.hpp reaches the upstream kernels headers through session_flow/sampler, so this target
# needs the pinned include directory as well as the backend one.
target_compile_options(strata-hpu-prefill-until-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_prefill_until COMMAND strata-hpu-prefill-until-test)
set_tests_properties(hpu_prefill_until PROPERTIES LABELS "host" TIMEOUT 60)
# IQ2_XXS (GGML type 16) codebook superblock dot: eight 32-value groups of four 8-value
# grid lookups with a 7-bit parity-folded sign field and a 4-bit scale applied as the pinned
# CUDA integer sumi*(2*sc+1)/8. Checked exactly against that integer form and within the
# derived truncation bound of the unmodified pinned ggml decoder, with all 256 grid
# magnitudes recovered through the decoder. Host only.
add_executable(strata-hpu-packed-iq2xxs-test tests/packed_iq2xxs.cpp)
target_link_libraries(strata-hpu-packed-iq2xxs-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq2xxs-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq2xxs COMMAND strata-hpu-packed-iq2xxs-test)
set_tests_properties(hpu_packed_iq2xxs PROPERTIES LABELS "host" TIMEOUT 180)
# IQ3_XXS (GGML type 18) codebook superblock dot: four magnitudes per grid byte, a 7-bit
# parity-folded sign field per lane and the pinned CUDA integer scale
# (sc*sumi + sumi/2)/2, checked exactly against that form and within the derived truncation
# bound of the unmodified pinned ggml decoder, with all 256 grid entries recovered through
# the decoder. Host only.
add_executable(strata-hpu-packed-iq3xxs-test tests/packed_iq3xxs.cpp)
target_link_libraries(strata-hpu-packed-iq3xxs-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq3xxs-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq3xxs COMMAND strata-hpu-packed-iq3xxs-test)
set_tests_properties(hpu_packed_iq3xxs PROPERTIES LABELS "host" TIMEOUT 180)
# IQ2_XS (GGML type 17) codebook superblock dot: 512 uint64 grid entries addressed by a
# 9-bit field that shares each uint16 code with a 7-bit parity-folded sign field, and two
# 4-bit nibble scales per 32-value group applied as the pinned CUDA integer
# (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4. Host only.
add_executable(strata-hpu-packed-iq2xs-test tests/packed_iq2xs.cpp)
target_link_libraries(strata-hpu-packed-iq2xs-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq2xs-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq2xs COMMAND strata-hpu-packed-iq2xs-test)
set_tests_properties(hpu_packed_iq2xs PROPERTIES LABELS "host" TIMEOUT 180)
# IQ3_S (GGML type 21) codebook superblock dot: a 512-entry grid addressed with a ninth bit
# from a separate qh field, plain per-value sign bytes, and the odd integer multiplier
# (1 + 2*sc) that the pinned CUDA applies without any truncation. Host only.
add_executable(strata-hpu-packed-iq3s-test tests/packed_iq3s.cpp)
target_link_libraries(strata-hpu-packed-iq3s-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq3s-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq3s COMMAND strata-hpu-packed-iq3s-test)
set_tests_properties(hpu_packed_iq3s PROPERTIES LABELS "host" TIMEOUT 180)
# IQ2_S (GGML type 22) codebook superblock dot: a 1024-entry grid addressed with two bits
# from a separate qh byte, plain per-value sign bytes, and the truncating CUDA integer
# (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4 over the two nibble scales. Host only.
add_executable(strata-hpu-packed-iq2s-test tests/packed_iq2s.cpp)
target_link_libraries(strata-hpu-packed-iq2s-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq2s-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq2s COMMAND strata-hpu-packed-iq2s-test)
set_tests_properties(hpu_packed_iq2s PROPERTIES LABELS "host" TIMEOUT 180)
# IQ1_M (GGML type 29) ternary-grid superblock dot: a 2048-entry nibble grid addressed with
# three bits from qh, a per-lane +/-0.125 delta bit, and the fp16 super-block scale assembled
# from nibbles of the block's own scale words. Completes the admitted packed type set.
add_executable(strata-hpu-packed-iq1m-test tests/packed_iq1m.cpp)
target_link_libraries(strata-hpu-packed-iq1m-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-iq1m-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_iq1m COMMAND strata-hpu-packed-iq1m-test)
set_tests_properties(hpu_packed_iq1m PROPERTIES LABELS "host" TIMEOUT 180)
# NVFP4 (GGML type 40) 64-value block dot: four UE4M3 sub-block scales over packed E2M1
# nibbles, with the pinned helper's closing half folded into the kernel's exponent. Packs the
# global-scale expert format that upstream routes outside native_mmvq. Host only.
add_executable(strata-hpu-packed-nvfp4-test tests/packed_nvfp4.cpp)
target_link_libraries(strata-hpu-packed-nvfp4-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-packed-nvfp4-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_packed_nvfp4 COMMAND strata-hpu-packed-nvfp4-test)
set_tests_properties(hpu_packed_nvfp4 PROPERTIES LABELS "host" TIMEOUT 180)
# The device-side Q8_1 activation quantizer (hpu/tpc/activation_quant.c). Not yet wired into the
# graph: ModelHead still quantizes on the host, so this test is what compiles the kernel today. It
# runs the real TPC C through the host scalar adapter and requires its codes and metadata to equal
# both the host quantizer AND a transcription of native_mmvq's quantize_q8_1 law, pins the kernel's
# own fp16 converter over the whole fp16 domain plus the overflow/subnormal/exact-tie cases, and
# hands the kernel's buffers to the packed dot so the layout contract is executed, not asserted.
# Host only: no ISA execution, no device.
add_executable(strata-hpu-activation-quant-test tests/activation_quant_device.cpp)
target_link_libraries(strata-hpu-activation-quant-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-activation-quant-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_activation_quant COMMAND strata-hpu-activation-quant-test)
set_tests_properties(hpu_activation_quant PROPERTIES LABELS "host" TIMEOUT 180)
target_compile_options(strata_hpu_weights PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(strata-hpu-native-pack-test tests/native_pack.cpp)
target_link_libraries(strata-hpu-native-pack-test PRIVATE strata_hpu_weights)
target_compile_options(strata-hpu-native-pack-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_native_pack COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_pack.py" "$<TARGET_FILE:strata-hpu-native-pack-test>")
set_tests_properties(hpu_native_pack PROPERTIES LABELS "host" TIMEOUT 120 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
add_executable(strata-hpu-native-pack-binding-test tests/native_pack_binding.cpp)
target_link_libraries(strata-hpu-native-pack-binding-test PRIVATE strata_hpu_weights)
target_compile_options(strata-hpu-native-pack-binding-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_native_pack_binding COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_pack_binding.py" "$<TARGET_FILE:strata-hpu-native-pack-binding-test>")
set_tests_properties(hpu_native_pack_binding PROPERTIES LABELS "host" TIMEOUT 180 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
add_executable(strata-hpu-expert-scales-test tests/expert_scales.cpp src/cpu_quant_guard.cpp)
target_link_libraries(strata-hpu-expert-scales-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-expert-scales-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_expert_scales COMMAND strata-hpu-expert-scales-test)
set_tests_properties(hpu_expert_scales PROPERTIES LABELS "host" TIMEOUT 120)
add_executable(strata-hpu-control-vector-test tests/control_vector.cpp)
target_link_libraries(strata-hpu-control-vector-test PRIVATE strata_hpu_weights)
target_compile_options(strata-hpu-control-vector-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_control_vector COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/control_vector.py" "$<TARGET_FILE:strata-hpu-control-vector-test>")
set_tests_properties(hpu_control_vector PROPERTIES LABELS "host" TIMEOUT 30 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
add_executable(strata-hpu-mtp-info tools/mtp_info.cpp)
target_link_libraries(strata-hpu-mtp-info PRIVATE strata_hpu_weights)
target_compile_options(strata-hpu-mtp-info PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(strata-hpu-mtp-weights-test tests/mtp_weights.cpp)
target_link_libraries(strata-hpu-mtp-weights-test PRIVATE strata_hpu_weights)
target_compile_options(strata-hpu-mtp-weights-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_mtp_weights COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/mtp_weights.py" "$<TARGET_FILE:strata-hpu-mtp-weights-test>" "$<TARGET_FILE:strata-hpu-mtp-info>")
set_tests_properties(hpu_mtp_weights PROPERTIES LABELS "host" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1" TIMEOUT 30)
if(STRATA_HPU_CPU_EXPERTS)
  include(cmake/CpuExperts.cmake)
endif()
add_executable(strata-hpu-weights tools/weights_info.cpp)
target_link_libraries(strata-hpu-weights PRIVATE strata_hpu_weights)
add_executable(strata-hpu-model-info tools/model_info.cpp)
target_link_libraries(strata-hpu-model-info PRIVATE strata_hpu_weights)
target_compile_options(strata-hpu-model-info PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(strata-hpu-weights-test tests/weights.cpp)
target_link_libraries(strata-hpu-weights-test PRIVATE strata_hpu_weights ggml-base)
target_compile_options(strata-hpu-weights-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_weights COMMAND strata-hpu-weights-test)
set_tests_properties(hpu_weights PROPERTIES LABELS "host")
add_executable(strata-hpu-weight-table-test tests/weight_table.cpp)
target_link_libraries(strata-hpu-weight-table-test PRIVATE strata_hpu_weights)
target_compile_options(strata-hpu-weight-table-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_weight_table COMMAND strata-hpu-weight-table-test)
set_tests_properties(hpu_weight_table PROPERTIES LABELS "host")

add_library(strata_hpu_weight_transfer STATIC src/weight_transfer.cpp)
target_link_libraries(strata_hpu_weight_transfer PUBLIC strata_hpu_weights strata_hpu)
target_compile_options(strata_hpu_weight_transfer PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_library(strata_hpu_moe STATIC src/expert_cache.cpp src/expert_profile.cpp src/moe.cpp src/dense_weights.cpp src/layer.cpp src/resident_experts.cpp src/u4_resident.cpp src/u4_moe.cpp src/moe_mme.cpp src/fp8_batch.cpp)
target_link_libraries(strata_hpu_moe PUBLIC strata_hpu_weight_transfer strata_hpu_routing)
target_compile_options(strata_hpu_moe PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(strata_hpu_moe PUBLIC STRATA_HPU_CPU_EXPERTS=$<BOOL:${STRATA_HPU_CPU_EXPERTS}>)
if(STRATA_HPU_CPU_EXPERTS)
  target_link_libraries(strata_hpu_moe PUBLIC strata_hpu_cpu_experts)
endif()

# Reuse the portable original PLE I/O and dequantization unchanged. This target
# contains no CUDA/HIP source and starts no accelerator runtime.
add_library(strata_hpu_ngram STATIC
  ../src/kernels/ngram.cpp ../src/ngram/ple_reader.cpp
  ../src/platform/direct_file.cpp ../src/platform/memory.cpp)
target_include_directories(strata_hpu_ngram PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(strata_hpu_ngram PUBLIC Threads::Threads)
target_compile_options(strata_hpu_ngram PRIVATE -Wall -Wextra)
add_library(strata_hpu_ple_input STATIC src/ple_input.cpp src/ple_source.cpp)
target_link_libraries(strata_hpu_ple_input PUBLIC strata_hpu_weights strata_hpu_ngram)
target_compile_options(strata_hpu_ple_input PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(strata-hpu-ple-input-test tests/ple_input.cpp)
target_link_libraries(strata-hpu-ple-input-test PRIVATE strata_hpu_ple_input)
target_compile_options(strata-hpu-ple-input-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_ple_input COMMAND strata-hpu-ple-input-test)
set_tests_properties(hpu_ple_input PROPERTIES LABELS "host")
add_executable(strata-hpu-ple-source-test tests/ple_source.cpp)
target_link_libraries(strata-hpu-ple-source-test PRIVATE strata_hpu_ple_input)
target_compile_options(strata-hpu-ple-source-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME hpu_ple_source COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/ple_source.py" "$<TARGET_FILE:strata-hpu-ple-source-test>")
set_tests_properties(hpu_ple_source PROPERTIES LABELS "host" TIMEOUT 60 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
add_test(NAME hpu_ple_external COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/ple_external.py" "$<TARGET_FILE:strata-hpu-ple-input-test>")
set_tests_properties(hpu_ple_external PROPERTIES LABELS "host" TIMEOUT 60 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
add_executable(strata-hpu-ple-reader-test ../src/ngram/ple_reader_test.cpp)
target_link_libraries(strata-hpu-ple-reader-test PRIVATE strata_hpu_ngram)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ple-test-data")
add_test(NAME hpu_ple_reader COMMAND strata-hpu-ple-reader-test --selftest --dir "${CMAKE_CURRENT_BINARY_DIR}/ple-test-data")
set_tests_properties(hpu_ple_reader PROPERTIES LABELS "host" TIMEOUT 180 RUN_SERIAL TRUE)
# The same synthetic selftest through a gate that tolerates ONLY its two timing assertions (keep-alive pacing: this
# host observes 4..6 of the expected 8..20 reads per 300 ms under load, while a calm host passes all of it), and still
# requires every format/ticket/cache/error assertion to pass. The name deliberately does NOT contain "hpu_ple_reader":
# the no-card chain selects tests with `-E hpu_ple_reader` (a substring regex), and this gate is meant to run there.
add_test(NAME hpu_ple_formats_gate
         COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/../work/ple_reader_check.py"
                 "$<TARGET_FILE:strata-hpu-ple-reader-test>" "${CMAKE_CURRENT_BINARY_DIR}/ple-test-data"
                 "${CMAKE_CURRENT_BINARY_DIR}/ple-reader-gate.json")
set_tests_properties(hpu_ple_formats_gate PROPERTIES LABELS "host" TIMEOUT 1800 RUN_SERIAL TRUE
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                     DEPENDS strata-hpu-ple-reader-test)
add_library(strata_hpu_model_plan STATIC src/model_plan.cpp)
target_link_libraries(strata_hpu_model_plan PUBLIC strata_hpu_ple_input strata_hpu_control)
target_compile_options(strata_hpu_model_plan PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_library(strata_hpu_model STATIC src/model.cpp src/mtp.cpp src/dflash.cpp)
target_link_libraries(strata_hpu_model PUBLIC strata_hpu_model_plan strata_hpu_moe)
target_compile_options(strata_hpu_model PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(strata-hpu tools/generate.cpp)
target_link_libraries(strata-hpu PRIVATE strata_hpu_model)
target_compile_options(strata-hpu PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(strata-hpu-plan tools/model_plan.cpp)
target_link_libraries(strata-hpu-plan PRIVATE strata_hpu_model_plan)
target_compile_options(strata-hpu-plan PRIVATE -Wall -Wextra -Wpedantic -Werror)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME hpu_model_plan COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/model_plan.py" "$<TARGET_FILE:strata-hpu-plan>" "$<TARGET_FILE:strata-hpu>")
# 70.5s alone under ASan (work/remote_logs/asan-two-failures-detail-*.json) and this box also runs
# another tenant's job: the 120s limit was contention, not a defect. CMake comments are '#',
# which is why this one is not //.
set_tests_properties(hpu_model_plan PROPERTIES LABELS "host" TIMEOUT 900)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import regex, jinja2" RESULT_VARIABLE strata_hpu_python_frontend)
if(strata_hpu_python_frontend EQUAL 0)
  add_test(NAME hpu_prepare COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/prepare.py" "$<TARGET_FILE_DIR:strata-hpu>")
  set_tests_properties(hpu_prepare PROPERTIES LABELS "host" TIMEOUT 900 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
else()
  message(STATUS "HPU tokenizer/template export test unavailable: install original Strata frontend Python requirements")
endif()

# ---- the pinned-upstream MODULE audit, shipped as a guard ------------------------------------------------
# Every pinned upstream file (src/** and include/strata/**) must land in exactly one named bucket, and the audit
# REFUSES to write its record when anything is unclassified or a curated hpu carrier is missing. The second test
# proves that refusal three ways on a mutated COPY of the audit and requires the bound record to stay untouched.
# Both scripts resolve the project root from their own location, so the same guard runs on the Linux tree.
add_test(NAME hpu_upstream_module_audit COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/../work/audit_upstream_files.py")
set_tests_properties(hpu_upstream_module_audit PROPERTIES LABELS "host" TIMEOUT 300 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
add_test(NAME hpu_module_audit_negative_controls COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/../work/negative_control_module_audit.py")
set_tests_properties(hpu_module_audit_negative_controls PROPERTIES LABELS "host" TIMEOUT 600 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

# The CLI VALUE-DOMAIN audit: for every upstream option this port parses, the comparisons upstream makes on the field
# it assigns are listed and each needs a verdict - the check that caught `--kv` (a format upstream, a MiB number here).
add_test(NAME hpu_cli_value_domain_audit
         COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/../work/audit_cli_values.py" --check)
set_tests_properties(hpu_cli_value_domain_audit PROPERTIES LABELS "host" TIMEOUT 300
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_executable(strata-hpu-embedding_input-test tests/embedding_input.cpp)
target_include_directories(strata-hpu-embedding_input-test PRIVATE include)
target_link_libraries(strata-hpu-embedding_input-test PRIVATE strata_hpu_weights strata_hpu_control ggml-base)
target_compile_options(strata-hpu-embedding_input-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_embedding_input COMMAND strata-hpu-embedding_input-test)
set_tests_properties(hpu_embedding_input PROPERTIES LABELS "host" TIMEOUT 180)
# The MODEL seam's consumption law (strata/hpu/embedding_route.hpp): the owner of every gather a session
# makes, exercised without a card because the header deliberately avoids the Synapse runtime.
add_executable(strata-hpu-embedding_model-test tests/embedding_model.cpp)
target_include_directories(strata-hpu-embedding_model-test PRIVATE include)
target_link_libraries(strata-hpu-embedding_model-test PRIVATE strata_hpu_weights ggml-base)
target_compile_options(strata-hpu-embedding_model-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_embedding_model COMMAND strata-hpu-embedding_model-test)
set_tests_properties(hpu_embedding_model PROPERTIES LABELS "host" TIMEOUT 180)
add_executable(strata-hpu-expert_profile-test tests/expert_profile.cpp)
target_include_directories(strata-hpu-expert_profile-test PRIVATE include)
target_link_libraries(strata-hpu-expert_profile-test PRIVATE strata_hpu_moe strata_hpu_control ggml-base)
target_compile_options(strata-hpu-expert_profile-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_expert_profile COMMAND strata-hpu-expert_profile-test)
set_tests_properties(hpu_expert_profile PROPERTIES LABELS "host" TIMEOUT 180)
# --resident-experts / --resident-cpu-experts / --expert-cache-per-layer / --resident-budget-gib: the pure
# split+admission policy (strata/hpu/expert_residency.hpp), checked against the LRU and the adaptation engine.
add_executable(strata-hpu-expert_residency-test tests/expert_residency.cpp)
target_include_directories(strata-hpu-expert_residency-test PRIVATE include)
target_link_libraries(strata-hpu-expert_residency-test PRIVATE strata_hpu_moe strata_hpu_control ggml-base)
target_compile_options(strata-hpu-expert_residency-test PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
add_test(NAME hpu_expert_residency COMMAND strata-hpu-expert_residency-test)
set_tests_properties(hpu_expert_residency PROPERTIES LABELS "host" TIMEOUT 180)
