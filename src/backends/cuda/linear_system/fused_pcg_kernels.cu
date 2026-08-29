#include <linear_system/fused_pcg_kernels.h>

#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <uipc/common/log.h>

namespace uipc::backend::cuda
{
namespace
{
    constexpr int PcgVectorBlockSize = 64;

    __device__ bool iteration_is_active(const FusedPcgDeviceParams* params, IndexT iteration_in_chunk)
    {
        return iteration_in_chunk <= params->active_iterations;
    }

    __device__ __forceinline__ bool load_alpha(const Float* __restrict__ rz_old,
                                               const Float* __restrict__ pAp,
                                               const IndexT* __restrict__ status,
                                               const FusedPcgDeviceParams* __restrict__ params,
                                               IndexT iteration_in_chunk,
                                               Float& alpha)
    {
        if(!iteration_is_active(params, iteration_in_chunk)
           || *status != static_cast<IndexT>(FusedPcgStatus::Running))
            return false;

        alpha = *rz_old / *pAp;
        return true;
    }

    __global__ void update_convergence_kernel(const Float* __restrict__ rz_old,
                                              const Float* __restrict__ rz_new,
                                              Float* __restrict__ beta,
                                              IndexT* __restrict__ status,
                                              FusedPcgCheckState* __restrict__ check_state,
                                              const FusedPcgDeviceParams* __restrict__ params,
                                              IndexT iteration_in_chunk)
    {
        if(threadIdx.x != 0 || blockIdx.x != 0 || !iteration_is_active(params, iteration_in_chunk)
           || *status != static_cast<IndexT>(FusedPcgStatus::Running))
            return;

        const Float old_value = *rz_old;
        const Float new_value = *rz_new;
        if(fabs(new_value) <= params->tolerance)
            *status = static_cast<IndexT>(FusedPcgStatus::Converged);
        else
            *beta = new_value / old_value;

        check_state->rz                 = new_value;
        check_state->status             = *status;
        check_state->iteration_in_chunk = iteration_in_chunk;
    }

    __global__ void update_p_prepare_next_kernel(Float* __restrict__ p,
                                                 const Float* __restrict__ z,
                                                 SizeT vector_size,
                                                 const Float* __restrict__ beta,
                                                 const Float* __restrict__ rz_new,
                                                 Float* __restrict__ rz_old_next,
                                                 Float* __restrict__ rz_new_next,
                                                 const IndexT* __restrict__ status,
                                                 const FusedPcgDeviceParams* __restrict__ params,
                                                 IndexT iteration_in_chunk)
    {
        if(!iteration_is_active(params, iteration_in_chunk)
           || *status != static_cast<IndexT>(FusedPcgStatus::Running))
            return;

        const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(i < vector_size)
            p[i] = z[i] + *beta * p[i];

        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            *rz_old_next = *rz_new;
            *rz_new_next = Float{0.0};
        }
    }

    __global__ void update_convergence_p_prepare_next_kernel(
        Float* __restrict__                    p,
        const Float* __restrict__              z,
        SizeT                                  vector_size,
        const Float* __restrict__              rz_old,
        const Float* __restrict__              rz_new,
        Float* __restrict__                    beta,
        Float* __restrict__                    rz_old_next,
        Float* __restrict__                    rz_new_next,
        IndexT* __restrict__                   status,
        FusedPcgCheckState* __restrict__ check_state,
        const FusedPcgDeviceParams* __restrict__ params,
        IndexT                                  iteration_in_chunk)
    {
        __shared__ Float  block_beta;
        __shared__ IndexT block_updates_p;

        if(threadIdx.x == 0)
        {
            const bool active = iteration_is_active(params, iteration_in_chunk);
            const bool running =
                active && *status == static_cast<IndexT>(FusedPcgStatus::Running);
            block_updates_p = 0;
            if(running)
            {
                const Float old_value = *rz_old;
                const Float new_value = *rz_new;
                const bool converged  = fabs(new_value) <= params->tolerance;
                if(!converged)
                {
                    block_beta     = new_value / old_value;
                    block_updates_p = 1;
                }

                if(blockIdx.x == 0)
                {
                    if(converged)
                        *status = static_cast<IndexT>(FusedPcgStatus::Converged);
                    else
                        *beta = block_beta;

                    check_state->rz = new_value;
                    check_state->status =
                        converged ? static_cast<IndexT>(FusedPcgStatus::Converged) :
                                    static_cast<IndexT>(FusedPcgStatus::Running);
                    check_state->iteration_in_chunk = iteration_in_chunk;

                    // Preserve the legacy zero-length behavior: the old
                    // prepare-next launch was skipped when p.size() == 0.
                    if(!converged && vector_size != 0)
                    {
                        *rz_old_next = new_value;
                        *rz_new_next = Float{0.0};
                    }
                }
            }
        }
        __syncthreads();

        const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(block_updates_p && i < vector_size)
            p[i] = z[i] + block_beta * p[i];
    }

    __global__ void identity_update_apply_dot_kernel(Float* __restrict__ x,
                                                     const Float* __restrict__ p,
                                                     Float* __restrict__ r,
                                                     const Float* __restrict__ Ap,
                                                     Float* __restrict__ z,
                                                     SizeT vector_size,
                                                     const Float* __restrict__ rz_old,
                                                     const Float* __restrict__ pAp,
                                                     Float* __restrict__ rz_new,
                                                     const IndexT* __restrict__ status,
                                                     const FusedPcgDeviceParams* __restrict__ params,
                                                     IndexT iteration_in_chunk)
    {
        using BlockReduce = cub::BlockReduce<Float, PcgVectorBlockSize>;
        __shared__ typename BlockReduce::TempStorage reduce_storage;

        const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
        Float      alpha = Float{0.0};
        const bool running =
            load_alpha(rz_old, pAp, status, params, iteration_in_chunk, alpha);
        Float local_dot = 0.0;
        if(running && i < vector_size)
        {
            x[i] += alpha * p[i];
            const Float r_new = r[i] - alpha * Ap[i];
            r[i]              = r_new;
            z[i]              = r_new;
            local_dot         = r_new * r_new;
        }

        const Float block_sum = BlockReduce(reduce_storage).Sum(local_dot);
        if(threadIdx.x == 0 && block_sum != Float{0.0})
            atomicAdd(rz_new, block_sum);
    }

    __global__ void abd_fused_update_apply_dot_kernel(const Matrix12x12* __restrict__ diag_inv,
                                                      SizeT body_count,
                                                      Float* __restrict__ x,
                                                      const Float* __restrict__ p,
                                                      Float* __restrict__ r,
                                                      const Float* __restrict__ Ap,
                                                      Float* __restrict__ z,
                                                      const Float* __restrict__ rz_old,
                                                      const Float* __restrict__ pAp,
                                                      Float* __restrict__ rz_new,
                                                      const IndexT* __restrict__ status,
                                                      const FusedPcgDeviceParams* __restrict__ params,
                                                      IndexT iteration_in_chunk)
    {
        constexpr int WarpSize      = 32;
        const int     lane          = threadIdx.x & (WarpSize - 1);
        const int     warp_in_block = threadIdx.x / WarpSize;
        const int  body  = blockIdx.x * (blockDim.x / WarpSize) + warp_in_block;
        Float      alpha = Float{0.0};
        const bool running =
            load_alpha(rz_old, pAp, status, params, iteration_in_chunk, alpha);
        const bool valid = running && body < body_count && lane < 12;

        Float r_value = 0.0;
        if(valid)
        {
            const int i = body * 12 + lane;
            x[i] += alpha * p[i];
            r_value = r[i] - alpha * Ap[i];
            r[i]    = r_value;
        }

        Float z_value = 0.0;
#pragma unroll
        for(int col = 0; col < 12; ++col)
        {
            const Float r_col = __shfl_sync(0xffffffffu, r_value, col);
            if(valid)
                z_value = fma(diag_inv[body](lane, col), r_col, z_value);
        }
        if(valid)
            z[body * 12 + lane] = z_value;

        Float dot = valid ? r_value * z_value : 0.0;
#pragma unroll
        for(int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        if(lane == 0 && running && body < body_count)
            atomicAdd(rz_new, dot);
    }

    __global__ void full_abd_update_residual_kernel(Float* __restrict__ x,
                                                    const Float* __restrict__ p,
                                                    Float* __restrict__ r,
                                                    const Float* __restrict__ Ap,
                                                    SizeT vector_size,
                                                    const Float* __restrict__ rz_old,
                                                    const Float* __restrict__ pAp,
                                                    const IndexT* __restrict__ status,
                                                    const FusedPcgDeviceParams* __restrict__ params,
                                                    IndexT iteration_in_chunk)
    {
        const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
        Float      alpha = Float{0.0};
        const bool running =
            load_alpha(rz_old, pAp, status, params, iteration_in_chunk, alpha);
        if(!running || i >= vector_size)
            return;

        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }

    __global__ void full_abd_apply_dot_kernel(const Float* __restrict__ full_inv,
                                              SizeT vector_size,
                                              const Float* __restrict__ r,
                                              Float* __restrict__ z,
                                              Float* __restrict__ rz_new,
                                              const IndexT* __restrict__ status,
                                              const FusedPcgDeviceParams* __restrict__ params,
                                              IndexT iteration_in_chunk)
    {
        using BlockReduce = cub::BlockReduce<Float, PcgVectorBlockSize>;
        __shared__ typename BlockReduce::TempStorage reduce_storage;

        const SizeT row =
            static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
        const bool running = iteration_is_active(params, iteration_in_chunk)
                             && *status == static_cast<IndexT>(FusedPcgStatus::Running);
        Float local_dot = 0.0;
        if(running && row < vector_size)
        {
            Float z_value = 0.0;
            for(SizeT col = 0; col < vector_size; ++col)
                z_value = fma(full_inv[row + col * vector_size], r[col], z_value);
            z[row]    = z_value;
            local_dot = r[row] * z_value;
        }

        const Float block_sum = BlockReduce(reduce_storage).Sum(local_dot);
        if(threadIdx.x == 0 && block_sum != Float{0.0})
            atomicAdd(rz_new, block_sum);
    }

    __global__ void fem_fused_update_apply_dot_kernel(const Matrix3x3* __restrict__ diag_inv,
                                                      SizeT vertex_count,
                                                      Float* __restrict__ x,
                                                      const Float* __restrict__ p,
                                                      Float* __restrict__ r,
                                                      const Float* __restrict__ Ap,
                                                      Float* __restrict__ z,
                                                      const Float* __restrict__ rz_old,
                                                      const Float* __restrict__ pAp,
                                                      Float* __restrict__ rz_new,
                                                      const IndexT* __restrict__ status,
                                                      const FusedPcgDeviceParams* __restrict__ params,
                                                      IndexT iteration_in_chunk)
    {
        using BlockReduce = cub::BlockReduce<Float, PcgVectorBlockSize>;
        __shared__ typename BlockReduce::TempStorage reduce_storage;

        const SizeT vertex =
            static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
        Float      alpha = Float{0.0};
        const bool running =
            load_alpha(rz_old, pAp, status, params, iteration_in_chunk, alpha);
        Float local_dot = 0.0;
        if(running && vertex < vertex_count)
        {
            Eigen::Matrix<Float, 3, 1> r_vertex;
#pragma unroll
            for(int component = 0; component < 3; ++component)
            {
                const SizeT i = vertex * 3 + component;
                x[i] += alpha * p[i];
                r_vertex(component) = r[i] - alpha * Ap[i];
                r[i]                = r_vertex(component);
            }
            const Eigen::Matrix<Float, 3, 1> z_vertex = diag_inv[vertex] * r_vertex;
#pragma unroll
            for(int component = 0; component < 3; ++component)
                z[vertex * 3 + component] = z_vertex(component);
            local_dot = r_vertex.dot(z_vertex);
        }

        const Float block_sum = BlockReduce(reduce_storage).Sum(local_dot);
        if(threadIdx.x == 0 && running)
            atomicAdd(rz_new, block_sum);
    }
}  // namespace

void launch_fused_pcg_update_convergence(muda::CVarView<Float> rz_old,
                                         muda::CVarView<Float> rz_new,
                                         muda::VarView<Float>  beta,
                                         muda::VarView<IndexT> status,
                                         muda::VarView<FusedPcgCheckState> check_state,
                                         muda::CVarView<FusedPcgDeviceParams> params,
                                         IndexT       iteration_in_chunk,
                                         cudaStream_t stream)
{
    update_convergence_kernel<<<1, 32, 0, stream>>>(rz_old.data(),
                                                    rz_new.data(),
                                                    beta.data(),
                                                    status.data(),
                                                    check_state.data(),
                                                    params.data(),
                                                    iteration_in_chunk);
}

void launch_fused_pcg_update_p_prepare_next(muda::DenseVectorView<Float>  p,
                                            muda::CDenseVectorView<Float> z,
                                            muda::CVarView<Float>         beta,
                                            muda::CVarView<Float>  rz_new,
                                            muda::VarView<Float>   rz_old_next,
                                            muda::VarView<Float>   rz_new_next,
                                            muda::CVarView<IndexT> status,
                                            muda::CVarView<FusedPcgDeviceParams> params,
                                            IndexT       iteration_in_chunk,
                                            cudaStream_t stream)
{
    const int grid_size =
        static_cast<int>((p.size() + PcgVectorBlockSize - 1) / PcgVectorBlockSize);
    if(grid_size == 0)
        return;
    update_p_prepare_next_kernel<<<grid_size, PcgVectorBlockSize, 0, stream>>>(
        p.data(),
        z.data(),
        p.size(),
        beta.data(),
        rz_new.data(),
        rz_old_next.data(),
        rz_new_next.data(),
        status.data(),
        params.data(),
        iteration_in_chunk);
}

void launch_fused_pcg_update_convergence_p_prepare_next(
    muda::DenseVectorView<Float>        p,
    muda::CDenseVectorView<Float>       z,
    muda::CVarView<Float>               rz_old,
    muda::CVarView<Float>               rz_new,
    muda::VarView<Float>                beta,
    muda::VarView<Float>                rz_old_next,
    muda::VarView<Float>                rz_new_next,
    muda::VarView<IndexT>               status,
    muda::VarView<FusedPcgCheckState> check_state,
    muda::CVarView<FusedPcgDeviceParams> params,
    IndexT                               iteration_in_chunk,
    cudaStream_t                         stream)
{
    UIPC_ASSERT(p.size() == z.size(),
                "fused PCG p/z vectors must have the same size");
    const int grid_size =
        std::max(1,
                 static_cast<int>((p.size() + PcgVectorBlockSize - 1)
                                  / PcgVectorBlockSize));
    update_convergence_p_prepare_next_kernel<<<grid_size, PcgVectorBlockSize, 0, stream>>>(
        p.data(),
        z.data(),
        p.size(),
        rz_old.data(),
        rz_new.data(),
        beta.data(),
        rz_old_next.data(),
        rz_new_next.data(),
        status.data(),
        check_state.data(),
        params.data(),
        iteration_in_chunk);
}

void launch_fused_pcg_identity_update_apply_dot(muda::DenseVectorView<Float>  x,
                                                muda::CDenseVectorView<Float> p,
                                                muda::DenseVectorView<Float>  r,
                                                muda::CDenseVectorView<Float> Ap,
                                                muda::DenseVectorView<Float> z,
                                                muda::CVarView<Float>  rz_old,
                                                muda::CVarView<Float>  pAp,
                                                muda::VarView<Float>   rz_new,
                                                muda::CVarView<IndexT> status,
                                                muda::CVarView<FusedPcgDeviceParams> params,
                                                IndexT       iteration_in_chunk,
                                                cudaStream_t stream)
{
    const int grid_size =
        static_cast<int>((x.size() + PcgVectorBlockSize - 1) / PcgVectorBlockSize);
    if(grid_size == 0)
        return;
    identity_update_apply_dot_kernel<<<grid_size, PcgVectorBlockSize, 0, stream>>>(
        x.data(),
        p.data(),
        r.data(),
        Ap.data(),
        z.data(),
        x.size(),
        rz_old.data(),
        pAp.data(),
        rz_new.data(),
        status.data(),
        params.data(),
        iteration_in_chunk);
}

void launch_abd_diag_preconditioner_apply(muda::CBufferView<Matrix12x12> diag_inv,
                                          muda::CDenseVectorView<Float> r,
                                          muda::DenseVectorView<Float>  z,
                                          muda::CVarView<IndexT>        status,
                                          cudaStream_t                  stream)
{
    // The legacy path intentionally keeps MUDA's original default launch
    // policy. The stream argument exists so the test and production launcher
    // signatures match, but this legacy operation only supports the default
    // stream, exactly as the pre-extraction implementation did.
    UIPC_ASSERT(stream == nullptr, "legacy ABD preconditioner apply uses the default stream");
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diag_inv.size(),
               [r      = r.cviewer().name("r"),
                z      = z.viewer().name("z"),
                status = status.cviewer().name("converged"),
                diag_inv = diag_inv.cviewer().name("diag_inv")] __device__(int i) mutable
               {
                   if(*status != 0)
                       return;
                   z.segment<12>(i * 12).as_eigen() =
                       diag_inv(i) * r.segment<12>(i * 12).as_eigen();
               });
}

void launch_fused_pcg_abd_update_apply_dot(muda::CBufferView<Matrix12x12> diag_inv,
                                           muda::DenseVectorView<Float>  x,
                                           muda::CDenseVectorView<Float> p,
                                           muda::DenseVectorView<Float>  r,
                                           muda::CDenseVectorView<Float> Ap,
                                           muda::DenseVectorView<Float>  z,
                                           muda::CVarView<Float>         rz_old,
                                           muda::CVarView<Float>         pAp,
                                           muda::VarView<Float>          rz_new,
                                           muda::CVarView<IndexT>        status,
                                           muda::CVarView<FusedPcgDeviceParams> params,
                                           IndexT       iteration_in_chunk,
                                           cudaStream_t stream)
{
    constexpr int BlockSize     = 64;
    constexpr int WarpsPerBlock = BlockSize / 32;
    const int     grid_size =
        static_cast<int>((diag_inv.size() + WarpsPerBlock - 1) / WarpsPerBlock);
    if(grid_size == 0)
        return;
    abd_fused_update_apply_dot_kernel<<<grid_size, BlockSize, 0, stream>>>(
        diag_inv.data(),
        diag_inv.size(),
        x.data(),
        p.data(),
        r.data(),
        Ap.data(),
        z.data(),
        rz_old.data(),
        pAp.data(),
        rz_new.data(),
        status.data(),
        params.data(),
        iteration_in_chunk);
}

void launch_fused_pcg_full_abd_update_apply_dot(muda::CBufferView<Float> full_inv,
                                                muda::DenseVectorView<Float>  x,
                                                muda::CDenseVectorView<Float> p,
                                                muda::DenseVectorView<Float>  r,
                                                muda::CDenseVectorView<Float> Ap,
                                                muda::DenseVectorView<Float> z,
                                                muda::CVarView<Float>  rz_old,
                                                muda::CVarView<Float>  pAp,
                                                muda::VarView<Float>   rz_new,
                                                muda::CVarView<IndexT> status,
                                                muda::CVarView<FusedPcgDeviceParams> params,
                                                IndexT       iteration_in_chunk,
                                                cudaStream_t stream)
{
    const SizeT vector_size = x.size();
    UIPC_ASSERT(p.size() == vector_size && r.size() == vector_size
                    && Ap.size() == vector_size && z.size() == vector_size,
                "full ABD fused PCG vectors must have the same size");
    UIPC_ASSERT(vector_size == 0 || full_inv.size() == vector_size * vector_size,
                "full ABD inverse has {} scalars for {} degrees of freedom",
                full_inv.size(),
                vector_size);

    const int grid_size =
        static_cast<int>((vector_size + PcgVectorBlockSize - 1) / PcgVectorBlockSize);
    if(grid_size == 0)
        return;

    full_abd_update_residual_kernel<<<grid_size, PcgVectorBlockSize, 0, stream>>>(
        x.data(),
        p.data(),
        r.data(),
        Ap.data(),
        vector_size,
        rz_old.data(),
        pAp.data(),
        status.data(),
        params.data(),
        iteration_in_chunk);
    full_abd_apply_dot_kernel<<<grid_size, PcgVectorBlockSize, 0, stream>>>(
        full_inv.data(),
        vector_size,
        r.data(),
        z.data(),
        rz_new.data(),
        status.data(),
        params.data(),
        iteration_in_chunk);
}

void launch_fem_diag_preconditioner_apply(muda::CBufferView<Matrix3x3> diag_inv,
                                          muda::CDenseVectorView<Float> r,
                                          muda::DenseVectorView<Float>  z,
                                          muda::CVarView<IndexT>        status,
                                          cudaStream_t                  stream)
{
    UIPC_ASSERT(stream == nullptr, "legacy FEM preconditioner apply uses the default stream");
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diag_inv.size(),
               [r      = r.cviewer().name("r"),
                z      = z.viewer().name("z"),
                status = status.cviewer().name("converged"),
                diag_inv = diag_inv.cviewer().name("diag_inv")] __device__(int i) mutable
               {
                   if(*status != 0)
                       return;
                   z.segment<3>(i * 3).as_eigen() =
                       diag_inv(i) * r.segment<3>(i * 3).as_eigen();
               });
}

void launch_fused_pcg_fem_update_apply_dot(muda::CBufferView<Matrix3x3> diag_inv,
                                           muda::DenseVectorView<Float>  x,
                                           muda::CDenseVectorView<Float> p,
                                           muda::DenseVectorView<Float>  r,
                                           muda::CDenseVectorView<Float> Ap,
                                           muda::DenseVectorView<Float>  z,
                                           muda::CVarView<Float>         rz_old,
                                           muda::CVarView<Float>         pAp,
                                           muda::VarView<Float>          rz_new,
                                           muda::CVarView<IndexT>        status,
                                           muda::CVarView<FusedPcgDeviceParams> params,
                                           IndexT       iteration_in_chunk,
                                           cudaStream_t stream)
{
    const int grid_size =
        static_cast<int>((diag_inv.size() + PcgVectorBlockSize - 1) / PcgVectorBlockSize);
    if(grid_size == 0)
        return;
    fem_fused_update_apply_dot_kernel<<<grid_size, PcgVectorBlockSize, 0, stream>>>(
        diag_inv.data(),
        diag_inv.size(),
        x.data(),
        p.data(),
        r.data(),
        Ap.data(),
        z.data(),
        rz_old.data(),
        pAp.data(),
        rz_new.data(),
        status.data(),
        params.data(),
        iteration_in_chunk);
}
}  // namespace uipc::backend::cuda
