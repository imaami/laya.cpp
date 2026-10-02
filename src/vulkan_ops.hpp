#pragma once
#include "ggml.h"
#include <array>
#include <cstring>
#include <initializer_list>
#include <limits>
namespace laya::vulkan_precision {
// Custom nodes run only by the patched Vulkan backend (vulkan_dispatch.hpp):
// op_params[0] carries the operator flags and op_params[1] the operator, which
// the name confirms once, when the backend admits the node.
enum class op { pad16, finish, split, merge, pack_qkv, serial, reduce, mlp_bf16, mlp_f16, gelu_bf16, gelu_f16, norm, none };
inline constexpr const char* op_names[]={"laya.pad16-vulkan","laya.finish-projection-vulkan","laya.split-vulkan",
    "laya.merge-vulkan","laya.pack-qkv-vulkan","laya.serial-vulkan","laya.reduce-vulkan","laya.mlp-bf16-vulkan",
    "laya.mlp-f16-vulkan","laya.gelu-bf16-vulkan","laya.gelu-f16-vulkan","laya.norm-vulkan"};
inline op kind(const ggml_tensor* t) {
    const auto k=uint32_t(t->op_params[1]);
    return k<uint32_t(op::none) && !std::strcmp(t->name,op_names[k]) ? op(k) : op::none;
}
[[noreturn]] inline void no_cpu(ggml_tensor*, int, int, void*) { GGML_ABORT("Laya Vulkan operators require a Vulkan GPU"); }
inline ggml_tensor* custom(ggml_context* ctx, op kind, ggml_type type, std::array<int64_t,4> ne,
                           std::initializer_list<ggml_tensor*> inputs, int32_t flags=0) {
    ggml_tensor* args[3];
    int count=0;
    for (auto input:inputs) if (input) args[count++]=input;
    auto output=ggml_custom_4d(ctx,type,ne[0],ne[1],ne[2],ne[3],args,count,no_cpu,1,nullptr);
    output->op_params[0]=flags;
    output->op_params[1]=int32_t(kind);
    ggml_set_name(output,op_names[int(kind)]);
    return output;
}
inline int32_t storage(ggml_type type) {
    if (type!=GGML_TYPE_F32 && type!=GGML_TYPE_F16 && type!=GGML_TYPE_BF16) GGML_ABORT("Invalid Vulkan storage precision");
    return type==GGML_TYPE_F16 ? 1 : type==GGML_TYPE_BF16 ? 2 : 0;
}
// Copy storage bits directly; padding adds positive zero without float casts.
inline ggml_tensor* pad16(ggml_context* ctx, ggml_tensor* x, int64_t padding) {
    if ((x->type!=GGML_TYPE_F16 && x->type!=GGML_TYPE_BF16) || !ggml_is_contiguous(x) ||
        x->ne[0]<=0 || ggml_nrows(x)<=0 || padding<0 || padding>std::numeric_limits<int64_t>::max()-x->ne[0])
        GGML_ABORT("Invalid Vulkan 16-bit padding input");
    if (x->ne[0]+padding>std::numeric_limits<uint32_t>::max()/ggml_nrows(x))
        GGML_ABORT("Vulkan 16-bit padding exceeds the shader index range");
    return padding ? custom(ctx,op::pad16,x->type,{x->ne[0]+padding,x->ne[1],x->ne[2],x->ne[3]},{x}) : x;
}
// Bias precedes storage rounding; the FP32 residual is added afterwards.
inline ggml_tensor* finish_projection(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias,
                                      ggml_tensor* residual, ggml_type stored_type) {
    return custom(ctx,op::finish,GGML_TYPE_F32,{x->ne[0],x->ne[1],x->ne[2],x->ne[3]},{x,bias ? bias : x,residual ? residual : x},
                  storage(stored_type)|(bias ? 4 : 0)|(residual ? 8 : 0));
}
// Pack Q/K/V and apply rotary positions in one dispatch, retaining explicit
// product/addition rounding and the requested projection storage precision.
inline ggml_tensor* pack_qkv(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cosine,
                             ggml_tensor* sine, int64_t length, int64_t batches, ggml_type stored_type) {
    if (length<=0 || batches<=0 || x->ne[0]%192 || x->ne[1]!=length*batches || bool(cosine)!=bool(sine))
        GGML_ABORT("Invalid Vulkan QKV packing geometry");
    return custom(ctx,op::pack_qkv,GGML_TYPE_F32,{64,length,x->ne[0]/192,3*batches},{x,cosine ? cosine : x,sine ? sine : x},
                  storage(stored_type)|(cosine ? 4 : 0));
}
// A finite FP16 high part has a rounding residual of at most 16.
// Scaling by 1024 preserves small corrections without overflowing at large inputs.
inline ggml_tensor* split_half(ggml_context* ctx, ggml_tensor* x) {
    return custom(ctx,op::split,GGML_TYPE_F16,{x->ne[0],x->ne[1]*2,1,1},{x});
}
inline ggml_tensor* merge_half(ggml_context* ctx, ggml_tensor* x) {
    return custom(ctx,op::merge,GGML_TYPE_F32,{x->ne[0],x->ne[1]/2,1,1},{x});
}
// Keep split-K partials in their original order. Tree reductions can cross
// a half-precision rounding midpoint even when every partial is exact. Biased
// low-precision projections store the reduction before their bias epilogue.
inline ggml_tensor* reduce_partials(ggml_context* ctx, ggml_tensor* x, ggml_type stored_type=GGML_TYPE_F32) {
    auto output=custom(ctx,op::reduce,GGML_TYPE_F32,{x->ne[0],x->ne[1],1,1},{x});
    return stored_type==GGML_TYPE_F32 ? output : ggml_cast(ctx,ggml_cast(ctx,output,stored_type),GGML_TYPE_F32);
}
// Serial matrix partitions store a low-precision running result between steps.
inline ggml_tensor* serial_partials(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias,
                                   bool bf16, bool bias_after_storage=false, bool bias_first=false) {
    return custom(ctx,op::serial,GGML_TYPE_F32,{x->ne[0],x->ne[1],1,1},{x,bias ? bias : x},
                  (bf16 ? 1 : 0)|(bias ? 2 : 0)|(bias_after_storage ? 4 : 0)|(bias_first ? 8 : 0));
}
inline ggml_tensor* activation(ggml_context* ctx, ggml_tensor* x, ggml_tensor* table, bool gated, bool bf16) {
    return custom(ctx,op(int(op::mlp_bf16)+(gated ? 0 : 2)+(bf16 ? 0 : 1)),GGML_TYPE_F32,
                  {x->ne[0]/(gated ? 2 : 1),x->ne[1],x->ne[2],x->ne[3]},{x,table});
}
inline ggml_tensor* norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias,
                         ggml_type stored_type=GGML_TYPE_F32) {
    return custom(ctx,op::norm,GGML_TYPE_F32,{x->ne[0],x->ne[1],x->ne[2],x->ne[3]},{x,weight,bias},storage(stored_type));
}
}
