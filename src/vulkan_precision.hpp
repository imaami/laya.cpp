#pragma once
#include "vulkan_ops.hpp"
#include "vulkan/projection_plans.hpp"

// Portable Vulkan graph operations preserve the BF16 boundaries of autocast:
// projections and activations round to BF16, while residuals and normalization
// stay in FP32. These expressions can subsequently be fused without changing
// their rounding points.
namespace laya::vulkan_precision {
inline ggml_tensor* round(ggml_context* ctx, ggml_tensor* x, ggml_type type) {
    return ggml_cast(ctx, ggml_cast(ctx,x,type),GGML_TYPE_F32);
}
inline ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, ggml_tensor* residual,
                          ggml_type type, projection_plan plan={}, bool amd_matching=false) {
    if (x->type!=GGML_TYPE_F32) x=ggml_cast(ctx,x,GGML_TYPE_F32);
    ggml_tensor* product;
    if (const int split_k=plan.chunk) {
        const int64_t parts=(weight->ne[0]+split_k-1)/split_k, padding=parts*split_k-weight->ne[0];
        auto padded_w=padding ? pad16(ctx,weight,padding) : weight;
        auto padded_x=padding ? ggml_pad(ctx,x,padding,0,0,0) : x;
        auto wa=ggml_permute(ctx,ggml_reshape_3d(ctx,padded_w,split_k,parts,weight->ne[1]),0,2,1,3);
        auto xa=ggml_permute(ctx,ggml_reshape_3d(ctx,padded_x,split_k,parts,x->ne[1]),0,2,1,3);
        auto partial=ggml_mul_mat(ctx,wa,xa);
        ggml_prec_set_acc(partial,GGML_PREC_F32);
        if (plan.serial) {
            // The separate-bias algorithm seeds the first serial partition;
            // parallel reductions instead apply that bias after storing the sum.
            auto result=serial_partials(ctx,partial,bias,type==GGML_TYPE_BF16,false,plan.bias_after_storage);
            return residual ? ggml_add(ctx,residual,result) : result;
        }
        if (plan.round_partial) partial=round(ctx,partial,type);
        product=reduce_partials(ctx,partial,plan.bias_after_storage ? type : GGML_TYPE_F32);
    } else {
        product=ggml_mul_mat(ctx,weight,x);
        ggml_prec_set_acc(product,GGML_PREC_F32);
        if (amd_matching) ggml_set_name(product,"laya.amd-low-projection");
    }
    return finish_projection(ctx,product,bias,residual,type);
}
}
