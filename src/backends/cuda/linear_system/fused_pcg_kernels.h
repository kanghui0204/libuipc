#pragma once

#include <linear_system/fused_pcg_common.h>
#include <muda/buffer/buffer_view.h>
#include <muda/buffer/var_view.h>
#include <muda/ext/linear_system/dense_vector_view.h>

namespace uipc::backend::cuda
{
void launch_fused_pcg_prepare_alpha(muda::CVarView<Float> rz_old,
                                    muda::CVarView<Float> pAp,
                                    muda::VarView<Float>  alpha,
                                    muda::VarView<IndexT> status,
                                    muda::VarView<FusedPcgCheckState> check_state,
                                    muda::CVarView<FusedPcgDeviceParams> params,
                                    IndexT       iteration_in_chunk,
                                    cudaStream_t stream);

void launch_fused_pcg_update_convergence(muda::CVarView<Float> rz_old,
                                         muda::CVarView<Float> rz_new,
                                         muda::VarView<Float>  beta,
                                         muda::VarView<IndexT> status,
                                         muda::VarView<FusedPcgCheckState> check_state,
                                         muda::CVarView<FusedPcgDeviceParams> params,
                                         IndexT       iteration_in_chunk,
                                         cudaStream_t stream);

void launch_fused_pcg_update_p_prepare_next(muda::DenseVectorView<Float>  p,
                                            muda::CDenseVectorView<Float> z,
                                            muda::CVarView<Float>         beta,
                                            muda::CVarView<Float>  rz_new,
                                            muda::VarView<Float>   rz_old_next,
                                            muda::VarView<Float>   rz_new_next,
                                            muda::CVarView<IndexT> status,
                                            muda::CVarView<FusedPcgDeviceParams> params,
                                            IndexT       iteration_in_chunk,
                                            cudaStream_t stream);

void launch_fused_pcg_identity_update_apply_dot(muda::DenseVectorView<Float>  x,
                                                muda::CDenseVectorView<Float> p,
                                                muda::DenseVectorView<Float>  r,
                                                muda::CDenseVectorView<Float> Ap,
                                                muda::DenseVectorView<Float> z,
                                                muda::CVarView<Float>  alpha,
                                                muda::VarView<Float>   rz_new,
                                                muda::CVarView<IndexT> status,
                                                muda::CVarView<FusedPcgDeviceParams> params,
                                                IndexT       iteration_in_chunk,
                                                cudaStream_t stream);

void launch_abd_diag_preconditioner_apply(muda::CBufferView<Matrix12x12> diag_inv,
                                          muda::CDenseVectorView<Float> r,
                                          muda::DenseVectorView<Float>  z,
                                          muda::CVarView<IndexT>        status,
                                          cudaStream_t stream = nullptr);

void launch_fused_pcg_abd_update_apply_dot(muda::CBufferView<Matrix12x12> diag_inv,
                                           muda::DenseVectorView<Float>  x,
                                           muda::CDenseVectorView<Float> p,
                                           muda::DenseVectorView<Float>  r,
                                           muda::CDenseVectorView<Float> Ap,
                                           muda::DenseVectorView<Float>  z,
                                           muda::CVarView<Float>         alpha,
                                           muda::VarView<Float>          rz_new,
                                           muda::CVarView<IndexT>        status,
                                           muda::CVarView<FusedPcgDeviceParams> params,
                                           IndexT       iteration_in_chunk,
                                           cudaStream_t stream);

void launch_fem_diag_preconditioner_apply(muda::CBufferView<Matrix3x3> diag_inv,
                                          muda::CDenseVectorView<Float> r,
                                          muda::DenseVectorView<Float>  z,
                                          muda::CVarView<IndexT>        status,
                                          cudaStream_t stream = nullptr);

void launch_fused_pcg_fem_update_apply_dot(muda::CBufferView<Matrix3x3> diag_inv,
                                           muda::DenseVectorView<Float>  x,
                                           muda::CDenseVectorView<Float> p,
                                           muda::DenseVectorView<Float>  r,
                                           muda::CDenseVectorView<Float> Ap,
                                           muda::DenseVectorView<Float>  z,
                                           muda::CVarView<Float>         alpha,
                                           muda::VarView<Float>          rz_new,
                                           muda::CVarView<IndexT>        status,
                                           muda::CVarView<FusedPcgDeviceParams> params,
                                           IndexT       iteration_in_chunk,
                                           cudaStream_t stream);
}  // namespace uipc::backend::cuda
