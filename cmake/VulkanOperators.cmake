target_include_directories(ggml-vulkan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}")
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\"" "#include \"ggml-vulkan-shaders.hpp\"\n#include \"vulkan/strict_spirv.hpp\"\n#include \"vulkan_ops.hpp\"")
laya_vk_replace(
  "    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
  "    {\n        const bool amd = device->vendor_id == VK_VENDOR_ID_AMD;\n        static const auto strict_norm = laya::vulkan_precision::preserve_spirv_fma(laya_amd_norm_spv, sizeof(laya_amd_norm_spv)/sizeof(uint32_t));\n        ggml_vk_create_pipeline(device, device->pipeline_laya_norm, \"laya_norm\", amd ? strict_norm.size()*sizeof(uint32_t) : sizeof(laya_norm_spv), amd ? strict_norm.data() : laya_norm_spv, \"main\", 4, 16, {1,1,1}, {amd ? 256u : 128u}, 1);\n    }\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")
# Fused attention keeps its explicit FMAs, and runs where its workgroup memory
# fits; global attention has a pipeline per bound of lane length.
foreach(attention local_attention global_attention_256 global_attention_512 global_attention_1024)
  string(REGEX REPLACE "_[0-9]+$" "" kind "${attention}")
  laya_vk_replace(
    "    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
    "    if (device->properties.limits.maxComputeSharedMemorySize >= laya::vulkan_precision::${kind}_shared) {\n        static const auto strict_attention = laya::vulkan_precision::preserve_spirv_fma(laya_${attention}_spv, sizeof(laya_${attention}_spv)/sizeof(uint32_t));\n        ggml_vk_create_pipeline(device, device->pipeline_laya_${attention}, \"laya_${attention}\", strict_attention.size()*sizeof(uint32_t), strict_attention.data(), \"main\", 5, 16, {256,1,1}, {}, 1);\n    }\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")
  laya_vk_replace("    vk_pipeline pipeline_norm_f32;" "    vk_pipeline pipeline_norm_f32;\n    vk_pipeline pipeline_laya_${attention};")
endforeach()
# Each shader is name:source:bindings[:glslc definitions]; zero bindings embeds
# SPIR-V without a pipeline: the norm pipeline above selects its AMD variant,
# and fused attention preserves its FMAs.
foreach(shader norm:norm:4 amd_norm:norm:0:-DLAYA_AMD_NORM=1 activation:activation:3 split:compensated:2:-DSPLIT=1
    merge:compensated:2 reduce:compensated:2:-DREDUCE=1 serial:compensated:3:-DSERIAL=1
    finish_projection:finish_projection:4 pack_qkv:pack_qkv:4 pad16:pad16:2 mask:mask:2 heads:heads:2
    local_attention:local_attention:0 global_attention_256:global_attention:0:-DSLOT_KEYS=8
    global_attention_512:global_attention:0:-DSLOT_KEYS=16 global_attention_1024:global_attention:0:-DSLOT_KEYS=32)
  string(REPLACE ":" ";" shader "${shader}")
  list(POP_FRONT shader name source bindings)
  set(header "${CMAKE_CURRENT_BINARY_DIR}/laya_${name}.spv.h")
  add_custom_command(OUTPUT "${header}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c ${shader}
      "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/${source}.comp" -o "${header}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/${source}.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rounding.glsl"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rocm_math.glsl" VERBATIM)
  add_custom_target(laya-vulkan-${name} DEPENDS "${header}")
  add_dependencies(ggml-vulkan laya-vulkan-${name})
  laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
    "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_${name}_spv[] =\n#include \"laya_${name}.spv.h\"\n;")
  if(bindings)
    laya_vk_replace("    vk_pipeline pipeline_norm_f32;" "    vk_pipeline pipeline_norm_f32;\n    vk_pipeline pipeline_laya_${name};")
  endif()
  if(bindings AND NOT name STREQUAL "norm")
    laya_vk_replace("    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
      "    ggml_vk_create_pipeline(device, device->pipeline_laya_${name}, \"laya_${name}\", sizeof(laya_${name}_spv), laya_${name}_spv, \"main\", ${bindings}, 16, {256,1,1}, {}, 1);\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")
  endif()
endforeach()
laya_vk_replace(
  "// Returns true if node has enqueued work into the queue, false otherwise"
  "#include \"vulkan_dispatch.hpp\"\n// Returns true if node has enqueued work into the queue, false otherwise")
laya_vk_replace(
  "    case GGML_OP_NORM:\n        ggml_vk_norm(ctx, compute_ctx, src0, node);"
  "    case GGML_OP_CUSTOM:\n        laya_vk_custom(ctx, compute_ctx, node);\n        break;\n    case GGML_OP_NORM:\n        ggml_vk_norm(ctx, compute_ctx, src0, node);")
laya_vk_replace(
  "    switch (op->op) {\n        case GGML_OP_UNARY:"
  "    switch (op->op) {\n        case GGML_OP_CUSTOM: return laya_vk_supports(op, *device);\n        case GGML_OP_UNARY:")
# Per-operator timings (GGML_VK_PERF_LOGGER) continue the context holding the
# input copies queued with each graph.
laya_vk_replace(
  "        GGML_ASSERT(ctx->compute_ctx.expired());\n        compute_ctx = ggml_vk_get_compute_ctx(ctx);\n        ctx->query_idx = 0;"
  "        compute_ctx = ggml_vk_get_compute_ctx(ctx);\n        ctx->query_idx = 0;")
# GGML_VK_PERF_LOGGER reports Laya operators by name, and other operators by
# their output shape, so each lane's custom passes and copies are timed apart.
laya_vk_replace(
  "        return fusion_str + ggml_op_name(node->op);\n    }\n\n    void log_timing("
  "        return fusion_str + (node->op == GGML_OP_CUSTOM ? node->name : ggml_op_name(node->op)) + \" (\" + std::to_string(node->ne[0]) + \",\" +\n            std::to_string(node->ne[1]) + \",\" + std::to_string(node->ne[2]) + \",\" + std::to_string(node->ne[3]) + \")\";\n    }\n\n    void log_timing(")
