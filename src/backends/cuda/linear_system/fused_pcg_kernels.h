#pragma once

#include <type_define.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
void launch_fused_pcg_abd_update_apply_dot(
    cuda_tool::CBufferView<Matrix12x12> diag_inv,
    cuda_tool::DenseVectorView<Float>   x,
    cuda_tool::CDenseVectorView<Float>  p,
    cuda_tool::DenseVectorView<Float>   r,
    cuda_tool::CDenseVectorView<Float>  Ap,
    cuda_tool::DenseVectorView<Float>   z,
    cuda_tool::CVarView<Float>          rz,
    cuda_tool::CVarView<Float>          pAp,
    cuda_tool::VarView<Float>           rz_new,
    cuda_tool::CVarView<IndexT>         converged,
    cudaStream_t                        stream);
}  // namespace uipc::backend::cuda
