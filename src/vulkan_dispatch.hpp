// Included inside the generated ggml Vulkan translation unit (C++17).
#include "vulkan_ops.hpp"
static bool laya_vk_supports(const ggml_tensor* op, const vk_device_struct& device) {
    using K=laya::vulkan_precision::op;
    const auto kind=laya::vulkan_precision::kind(op);
    const ggml_tensor *x=op->src[0],*b=op->src[1],*r=op->src[2];
    const uint32_t flags=uint32_t(op->op_params[0]);
    const bool indexable=ggml_nelements(op)<=UINT32_MAX;
    const auto f32=[](const ggml_tensor* t) { return t && t->type==GGML_TYPE_F32 && ggml_is_contiguous(t); };
    switch (kind) {
    case K::pad16:
        return x && (x->type==GGML_TYPE_F16 || x->type==GGML_TYPE_BF16) && op->type==x->type &&
            ggml_is_contiguous(x) && ggml_is_contiguous(op) && x->ne[0]>0 && op->ne[0]>=x->ne[0] &&
            op->ne[1]==x->ne[1] && op->ne[2]==x->ne[2] && op->ne[3]==x->ne[3] && ggml_nelements(op)>0 && indexable;
    case K::finish:
        return f32(x) && f32(b) && f32(r) && f32(op) && ggml_are_same_shape(x,op) && indexable && flags<16 &&
            (flags&3)!=3 && (!(flags&4) || ggml_nelements(b)==x->ne[0]) && (!(flags&8) || ggml_are_same_shape(r,x));
    case K::split: case K::merge:
        return f32(x) && op->type==(kind==K::split ? GGML_TYPE_F16 : GGML_TYPE_F32) && ggml_is_contiguous(op) &&
            x->ne[0]%2==0 && x->ne[2]==1 && x->ne[3]==1 && op->ne[2]==1 && op->ne[3]==1 && x->ne[0]==op->ne[0] &&
            indexable && (kind==K::split ? op->ne[1]==2*x->ne[1] : x->ne[1]==2*op->ne[1]);
    case K::pack_qkv:
        return f32(x) && f32(b) && f32(r) && f32(op) && op->ne[0]==64 && op->ne[3]%3==0 && x->ne[2]==1 &&
            x->ne[3]==1 && x->ne[0]==192*op->ne[2] && x->ne[1]==op->ne[1]*(op->ne[3]/3) && flags<32 && (flags&3)!=3 &&
            (!(flags&4) || (ggml_nelements(b)==64*op->ne[1] && ggml_nelements(r)==64*op->ne[1])) && indexable;
    case K::serial: case K::reduce:
        return f32(x) && f32(op) && x->ne[3]==1 && op->ne[2]==1 && op->ne[3]==1 && x->ne[0]==op->ne[0] &&
            x->ne[1]==op->ne[1] && ggml_nelements(x)<=UINT32_MAX && (kind==K::reduce || (f32(b) && flags<16 &&
            (flags&12)!=12 && (!(flags&2) || ggml_nelements(b)==x->ne[0])));
    case K::mlp_bf16: case K::mlp_f16: case K::gelu_bf16: case K::gelu_f16:
        return f32(op) && f32(x) && f32(b) && ggml_nelements(b)==65536 &&
            x->ne[0]==op->ne[0]*(kind<=K::mlp_f16 ? 2 : 1) && ggml_nrows(x)==ggml_nrows(op);
    case K::norm:
        return f32(op) && flags<=2 && f32(x) && f32(b) && (!r || f32(r)) && op->ne[0]%4==0 &&
            ggml_are_same_shape(op,x) && ggml_nelements(b)==op->ne[0] && (!r || ggml_nelements(r)==op->ne[0]);
    case K::mask:
        return x && x->type==GGML_TYPE_I32 && ggml_is_contiguous(x) && (op->type==GGML_TYPE_F16 || op->type==GGML_TYPE_F32) &&
            ggml_is_contiguous(op) && flags<2 && op->ne[0]>0 && op->ne[1]>=op->ne[0] && op->ne[1]<=65535 && op->ne[2]==1 &&
            op->ne[3]==ggml_nelements(x) && op->ne[3]<=65535 && indexable;
    case K::heads:
        return f32(x) && f32(op) && x->ne[0]==64 && op->ne[0]==64*x->ne[2] && op->ne[1]==x->ne[1]*x->ne[3] &&
            op->ne[2]==1 && op->ne[3]==1 && flags<3 && indexable;
    case K::local_attention: {
        const ggml_tensor* lengths=op->src[3];
        return device.pipeline_laya_local_attention && f32(x) && f32(b) && f32(r) && f32(op) && lengths &&
            lengths->type==GGML_TYPE_I32 && ggml_is_contiguous(lengths) && x->ne[0]==64 && ggml_are_same_shape(x,b) &&
            ggml_are_same_shape(x,op) && r->ne[0]==x->ne[1] && r->ne[1]==64 && r->ne[2]==x->ne[2] && r->ne[3]==x->ne[3] &&
            ggml_nelements(lengths)==x->ne[3] && x->ne[1]<=laya::vulkan_precision::local_attention_keys &&
            x->ne[2]<=65535 && x->ne[3]<=65535;
    }
    case K::none: break;
    }
    return false;
}
// The scheduler admits only nodes accepted above, so dispatch needs no checks.
static void laya_vk_custom(ggml_backend_vk_context* ctx, vk_context& subctx, ggml_tensor* op) {
    using K=laya::vulkan_precision::op;
    const auto kind=K(op->op_params[1]);
    ggml_tensor *x=op->src[0],*b=op->src[1],*r=op->src[2];
    const uint32_t n=uint32_t(ggml_nelements(op)),rows=uint32_t(ggml_nrows(op)),flags=uint32_t(op->op_params[0]);
    const auto run=[&](vk_pipeline& pipeline, std::array<uint32_t,4> params, std::array<uint32_t,3> elements, auto*... buffers) {
        ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
        ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{vk::DescriptorBufferInfo(ggml_vk_tensor_subbuffer(ctx,buffers))...},
            params,elements);
    };
    auto& device=*ctx->device;
    // Row-wise shaders step through rows beyond the smallest workgroup count
    // limit Vulkan guarantees.
    const auto capped=[](uint64_t count) { return uint32_t(std::min<uint64_t>(count,65535)); };
    const uint32_t width=uint32_t(op->ne[0]);
    switch (kind) {
    case K::pad16: return run(device.pipeline_laya_pad16,{uint32_t(x->ne[0]),uint32_t(op->ne[0]),n,0},{n,1,1},x,op);
    case K::finish: return run(device.pipeline_laya_finish_projection,{width,rows,flags,0},{width,capped(rows),1},x,b,r,op);
    case K::split: return run(device.pipeline_laya_split,{n/2,0,0,0},{n/4,1,1},x,op);
    case K::merge: return run(device.pipeline_laya_merge,{n,0,0,0},{n,1,1},x,op);
    case K::pack_qkv: {
        // One workgroup per block of 32 tokens of a head of a sequence.
        const uint32_t tokens=uint32_t(op->ne[1]),heads=uint32_t(op->ne[2]),batches=uint32_t(op->ne[3]/3);
        return run(device.pipeline_laya_pack_qkv,{tokens,heads,batches,flags},{(tokens+31)/32*256,heads,capped(batches)},x,b,r,op);
    }
    case K::heads: {
        const uint32_t tokens=uint32_t(x->ne[1]),batches=uint32_t(x->ne[3]);
        return run(device.pipeline_laya_heads,{width,tokens,batches,flags},{width,capped(tokens),capped(batches)},x,op);
    }
    case K::serial: return run(device.pipeline_laya_serial,{n,uint32_t(x->ne[2]),uint32_t(op->ne[0]),flags},{n,1,1},x,op,b);
    case K::reduce: return run(device.pipeline_laya_reduce,{n,uint32_t(x->ne[2]),0,0},{n,1,1},x,op);
    case K::norm: return run(device.pipeline_laya_norm,{width,rows,r ? 1u : 0u,flags},{capped(rows),1,1},x,b,r ? r : x,op);
    case K::mask:
        return run(device.pipeline_laya_mask,{uint32_t(op->ne[0]),uint32_t(op->ne[1]),flags|(op->type==GGML_TYPE_F16 ? 2u : 0u),0},
            {uint32_t(op->ne[0]),uint32_t(op->ne[1]),uint32_t(op->ne[3])},x,op);
    case K::local_attention: {
        // One workgroup per block of 32 queries of a head of a sequence.
        const uint32_t tokens=uint32_t(op->ne[1]),heads=uint32_t(op->ne[2]),batches=uint32_t(op->ne[3]);
        return run(device.pipeline_laya_local_attention,{tokens,0,0,0},{(tokens+31)/32*256,heads,batches},x,b,r,op->src[3],op);
    }
    case K::none: GGML_ABORT("Unsupported Laya Vulkan operator");
    default:
        return run(device.pipeline_laya_activation,{width,rows,kind<=K::mlp_f16 ? 1u : 0u,
            kind==K::mlp_bf16 || kind==K::gelu_bf16 ? 1u : 0u},{width,capped(rows),1},x,b,op);
    }
}
