#include <linear_system/linear_fused_pcg.h>
#include <linear_system/linear_fused_pcg_graph_resources.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <uipc/common/timer.h>
#include <cub/warp/warp_reduce.cuh>
#include "fused_pcg_kernels.h"
#include <muda/check/check_cuda_errors.h>
#include <algorithm>
namespace uipc::backend::cuda
{
namespace
{
    void fused_dot(muda::CDenseVectorView<Float> x,
                   muda::CDenseVectorView<Float> y,
                   muda::VarView<Float>          d_result);
}  // namespace

REGISTER_SIM_SYSTEM(LinearFusedPCG);

LinearFusedPCG::~LinearFusedPCG() = default;

void LinearFusedPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto        solver_attr = config.find<std::string>("linear_system/solver");
    std::string solver_name =
        solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "fused_pcg")
    {
        throw SimSystemException("LinearFusedPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    global_tol_rate    = tol_rate_attr->view()[0];

    auto check_attr = config.find<IndexT>("linear_system/check_interval");
    if(check_attr)
        check_interval = check_attr->view()[0];

    auto graph_attr = config.find<IndexT>("linear_system/fused_pcg/graph_enable");
    if(graph_attr)
        graph_enabled = graph_attr->view()[0] != 0;

    auto fused_preconditioner_attr =
        config.find<IndexT>("linear_system/fused_pcg/fused_preconditioner_enable");
    if(fused_preconditioner_attr)
        fused_preconditioner_enabled = fused_preconditioner_attr->view()[0] != 0;

    auto dump_attr = config.find<IndexT>("extras/debug/dump_linear_pcg");
    if(dump_attr && dump_attr->view()[0] != 0)
        logger::warn(
            "LinearFusedPCG: extras/debug/dump_linear_pcg is enabled but "
            "fused_pcg does not support PCG vector dumps. "
            "Set linear_system/solver to \"linear_pcg\" to use this feature.");

    logger::info("LinearFusedPCG: max_iter_ratio = {}, tol_rate = {}, check_interval = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 check_interval);
    if(graph_enabled)
        logger::debug("LinearFusedPCG CUDA Graph enabled, fused_preconditioner = {}",
                      fused_preconditioner_enabled);
}

void LinearFusedPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    x.buffer_view().fill(0);

    auto N = x.size();
    if(r.capacity() < N)
    {
        auto M = reserve_ratio * N;
        r.reserve(M);
        z.reserve(M);
        p.reserve(M);
        Ap.reserve(M);
    }

    r.resize(N);
    z.resize(N);
    p.resize(N);
    Ap.resize(N);

    const bool use_graph =
        graph_enabled && fused_preconditioner_enabled && supports_fused_pcg();
    if(use_graph)
    {
        UIPC_ASSERT(check_interval > 0 && check_interval <= 1024,
                    "LinearFusedPCG CUDA Graph check_interval must be in [1, 1024], got {}.",
                    check_interval);
        if(!graph_resources)
            graph_resources = std::make_unique<GraphResources>();
        for(auto& graph_vector : graph_resources->graph_Ap)
        {
            if(graph_vector.capacity() < N)
                graph_vector.reserve(reserve_ratio * N);
            if(graph_vector.size() != N)
                graph_vector.resize(N);
        }
    }
    auto iter = use_graph ? graph_fused_pcg(x, b, max_iter_ratio * b.size()) :
                            fused_pcg(x, b, max_iter_ratio * b.size());

    info.iter_count(iter);
    info.effective_iter_count(use_graph ? last_effective_iteration : iter);
}

SizeT LinearFusedPCG::current_triplet_bucket() const
{
    UIPC_ASSERT(graph_resources, "LinearFusedPCG Graph resources are not initialized.");
    return graph_resources->triplet_bucket;
}

FusedPcgGraphSignature LinearFusedPCG::current_graph_signature(muda::DenseVectorView<Float> x) const
{
    const auto             A = fused_pcg_matrix_capacity_view();
    FusedPcgGraphSignature signature;
    signature.matrix_rows              = A.row_indices().data();
    signature.matrix_cols              = A.col_indices().data();
    signature.matrix_values            = A.values().data();
    signature.x                        = x.data();
    signature.r                        = r.view().data();
    signature.z                        = z.view().data();
    signature.p                        = p.view().data();
    const auto& resources              = *graph_resources;
    signature.Ap_0                     = resources.graph_Ap[0].view().data();
    signature.Ap_1                     = resources.graph_Ap[1].view().data();
    signature.pAp_0                    = resources.graph_pAp[0].data();
    signature.pAp_1                    = resources.graph_pAp[1].data();
    signature.rz_old_0                 = resources.graph_rz_old[0].data();
    signature.rz_old_1                 = resources.graph_rz_old[1].data();
    signature.rz_new_0                 = resources.graph_rz_new[0].data();
    signature.rz_new_1                 = resources.graph_rz_new[1].data();
    signature.device_params            = resources.device_params.data();
    signature.status                   = d_converged.data();
    signature.check_state              = resources.check_state.data();
    signature.beta                     = resources.beta.data();
    signature.scalar_dof_count         = x.size();
    signature.triplet_bucket           = current_triplet_bucket();
    signature.preconditioner_signature = fused_pcg_preconditioner_signature();
    signature.check_interval           = check_interval;
    return signature;
}

void LinearFusedPCG::destroy_graphs() noexcept
{
    if(!graph_resources)
        return;
    if(graph_resources->graph_start_slot_0)
        cudaGraphExecDestroy(graph_resources->graph_start_slot_0);
    if(graph_resources->graph_start_slot_1)
        cudaGraphExecDestroy(graph_resources->graph_start_slot_1);
    graph_resources->graph_start_slot_0 = nullptr;
    graph_resources->graph_start_slot_1 = nullptr;
    graph_resources->signature_valid    = false;
}

void LinearFusedPCG::capture_graph(cudaGraphExec_t&             graph_exec,
                                   muda::DenseVectorView<Float> x,
                                   IndexT                       iteration_count,
                                   IndexT                       starting_slot)
{
    UIPC_ASSERT(graph_resources, "LinearFusedPCG Graph resources are not initialized.");
    cudaGraph_t graph  = nullptr;
    const auto  stream = graph_resources->capture_stream;
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for(IndexT local_iteration = 1; local_iteration <= iteration_count; ++local_iteration)
    {
        const IndexT current_slot = (starting_slot + local_iteration - 1) & 1;
        launch_graph_iteration(x, current_slot, local_iteration, stream);
    }
    checkCudaErrors(cudaStreamEndCapture(stream, &graph));
    checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    checkCudaErrors(cudaGraphDestroy(graph));
}

void LinearFusedPCG::rebuild_graphs(muda::DenseVectorView<Float>  x,
                                    const FusedPcgGraphSignature& signature)
{
    UIPC_ASSERT(graph_resources, "LinearFusedPCG Graph resources are not initialized.");
    destroy_graphs();
    if(!graph_resources->capture_stream)
        checkCudaErrors(cudaStreamCreateWithFlags(&graph_resources->capture_stream,
                                                  cudaStreamNonBlocking));
    const IndexT interval = static_cast<IndexT>(check_interval);
    capture_graph(graph_resources->graph_start_slot_0, x, interval, 0);
    if((check_interval & 1) != 0)
        capture_graph(graph_resources->graph_start_slot_1, x, interval, 1);
    graph_resources->signature       = signature;
    graph_resources->signature_valid = true;
    ++graph_resources->generation;
}

void LinearFusedPCG::launch_graph_iteration(muda::DenseVectorView<Float> x,
                                            IndexT       current_slot,
                                            IndexT       iteration_in_chunk,
                                            cudaStream_t stream)
{
    UIPC_ASSERT(graph_resources, "LinearFusedPCG Graph resources are not initialized.");
    auto&        resources = *graph_resources;
    const IndexT next_slot = current_slot ^ 1;

    fused_pcg_spmv_pipelined(p.cview(),
                             resources.graph_Ap[current_slot].view(),
                             resources.graph_pAp[current_slot].view(),
                             resources.graph_Ap[next_slot].view(),
                             resources.graph_pAp[next_slot].view(),
                             d_converged.view(),
                             resources.device_params.view(),
                             iteration_in_chunk,
                             current_triplet_bucket(),
                             stream);

    fused_pcg_update_apply_dot(x,
                               p.cview(),
                               r.view(),
                               resources.graph_Ap[current_slot].cview(),
                               z.view(),
                               resources.graph_rz_old[current_slot].view(),
                               resources.graph_pAp[current_slot].view(),
                               resources.graph_rz_new[current_slot].view(),
                               d_converged.view(),
                               resources.device_params.view(),
                               iteration_in_chunk,
                               stream);

    launch_fused_pcg_convergence_update_p_prepare_next(
        p.view(),
        z.cview(),
        resources.graph_rz_old[current_slot].view(),
        resources.graph_rz_new[current_slot].view(),
        resources.beta.view(),
        resources.graph_rz_old[next_slot].view(),
        resources.graph_rz_new[next_slot].view(),
        d_converged.view(),
        resources.check_state.view(),
        resources.device_params.view(),
        iteration_in_chunk,
        stream);
}

SizeT LinearFusedPCG::graph_fused_pcg(muda::DenseVectorView<Float>  x,
                                      muda::CDenseVectorView<Float> b,
                                      SizeT                         max_iter)
{
    UIPC_ASSERT(graph_resources, "LinearFusedPCG Graph resources are not initialized.");
    auto&        resources = *graph_resources;
    Timer        pcg_timer{"FusedPCGGraph"};
    cudaStream_t stream      = nullptr;
    last_terminal_iteration  = 0;
    last_effective_iteration = 0;
    last_reported_iteration  = 0;

    const IndexT running = static_cast<IndexT>(FusedPcgStatus::Running);
    checkCudaErrors(cudaMemcpyAsync(
        d_converged.data(), &running, sizeof(running), cudaMemcpyHostToDevice, stream));

    r.buffer_view().copy_from(b.buffer_view());
    apply_preconditioner(z, r, d_converged.view());
    p = z;

    fused_dot(r.cview(), z.cview(), d_rz.view());
    const Float rz_host = d_rz;
    check_init_rz_nan_inf(rz_host);
    const Float abs_rz0 = std::abs(rz_host);
    if(abs_rz0 == Float{0.0})
        return 0;

    for(auto& value : resources.graph_Ap)
        value.buffer_view().fill(0);
    for(auto& value : resources.graph_pAp)
        checkCudaErrors(cudaMemsetAsync(value.data(), 0, sizeof(Float), stream));
    for(auto& value : resources.graph_rz_new)
        checkCudaErrors(cudaMemsetAsync(value.data(), 0, sizeof(Float), stream));

    checkCudaErrors(cudaMemcpyAsync(
        resources.graph_rz_old[0].data(), d_rz.data(), sizeof(Float), cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemsetAsync(resources.graph_rz_old[1].data(), 0, sizeof(Float), stream));

    FusedPcgDeviceParams params;
    params.tolerance         = global_tol_rate * abs_rz0;
    params.triplet_count     = static_cast<IndexT>(fused_pcg_triplet_count());
    params.active_iterations = static_cast<IndexT>(check_interval);

    FusedPcgCheckState check_state;
    check_state.rz                 = rz_host;
    check_state.status             = running;
    check_state.iteration_in_chunk = 0;
    checkCudaErrors(cudaMemcpyAsync(
        resources.device_params.data(), &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(
        resources.check_state.data(), &check_state, sizeof(check_state), cudaMemcpyHostToDevice, stream));

    constexpr SizeT BucketGranularity = 1024;
    const SizeT     triplet_count     = fused_pcg_triplet_count();
    const SizeT     triplet_capacity  = fused_pcg_triplet_capacity();
    const SizeT     required_bucket =
        std::min(triplet_capacity,
                 std::max(triplet_count,
                          ((triplet_count + BucketGranularity - 1) / BucketGranularity)
                              * BucketGranularity));
    if(required_bucket > resources.triplet_bucket || resources.triplet_bucket > triplet_capacity)
        resources.triplet_bucket = required_bucket;

    const auto signature = current_graph_signature(x);
    if(!resources.signature_valid || signature != resources.signature)
        rebuild_graphs(x, signature);

    const SizeT max_executed = max_iter > 0 ? max_iter - 1 : 0;
    SizeT       completed    = 0;

    while(completed < max_executed)
    {
        const SizeT active = std::min<SizeT>(check_interval, max_executed - completed);
        if(active != check_interval)
        {
            params.active_iterations = static_cast<IndexT>(active);
            checkCudaErrors(cudaMemcpyAsync(
                resources.device_params.data(), &params, sizeof(params), cudaMemcpyHostToDevice, stream));
        }

        cudaGraphExec_t graph = (completed & 1) == 0 ? resources.graph_start_slot_0 :
                                                       resources.graph_start_slot_1;
        UIPC_ASSERT(graph, "LinearFusedPCG has no CUDA Graph for starting slot {}.", completed & 1);
        checkCudaErrors(cudaGraphLaunch(graph, stream));
        checkCudaErrors(cudaMemcpyAsync(&check_state,
                                        resources.check_state.data(),
                                        sizeof(check_state),
                                        cudaMemcpyDeviceToHost,
                                        stream));
        checkCudaErrors(cudaStreamSynchronize(stream));

        const auto status = static_cast<FusedPcgStatus>(check_state.status);
        check_iter_rz_nan_inf(check_state.rz, completed + active);
        if(status != FusedPcgStatus::Running)
        {
            last_terminal_iteration =
                completed + static_cast<SizeT>(check_state.iteration_in_chunk);
            last_effective_iteration = last_terminal_iteration;
            last_reported_iteration  = completed + active;
            return last_reported_iteration;
        }

        completed += active;
    }

    // No terminal state was observed. Preserve the legacy reported count,
    // record the iterations that actually executed, and leave the terminal
    // iteration unset.
    last_effective_iteration = max_executed;
    last_reported_iteration  = max_iter;
    return max_iter;
}

void LinearFusedPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearFusedPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_ok   = std::isfinite(norm_r);
        bool z_bad  = !std::isfinite(norm_z);
        auto hint = (r_ok && z_bad) ?
                        "preconditioner failed, likely due to inverse matrix calculation failure" :
                        "PCG iteration diverged";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    k,
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

// d_result = x^T * y  (cublas-free, device-only, CUB warp reduction)
namespace
{
    void fused_dot(muda::CDenseVectorView<Float> x,
                   muda::CDenseVectorView<Float> y,
                   muda::VarView<Float>          d_result)
    {
        using namespace muda;

        cudaMemsetAsync(d_result.data(), 0, sizeof(Float));

        constexpr int block_dim   = 256;
        constexpr int warp_size   = 32;
        constexpr int num_warps   = block_dim / warp_size;
        int           n           = x.size();
        int           block_count = (n + block_dim - 1) / block_dim;

        Launch(block_count, block_dim)
            .file_line(__FILE__, __LINE__)
            .apply(
                [x        = x.cviewer().name("x"),
                 y        = y.cviewer().name("y"),
                 d_result = d_result.viewer().name("d_result"),
                 n] __device__() mutable
                {
                    using WarpReduce = cub::WarpReduce<Float, warp_size>;
                    __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];

                    int   i   = blockIdx.x * blockDim.x + threadIdx.x;
                    Float val = (i < n) ? x(i) * y(i) : Float(0);

                    int   warp_id  = threadIdx.x / warp_size;
                    int   lane_id  = threadIdx.x & (warp_size - 1);
                    Float warp_sum = WarpReduce(temp_storage[warp_id]).Sum(val);

                    if(lane_id == 0)
                        muda::atomic_add(d_result.data(), warp_sum);
                });
    }
}  // namespace

// Same as linear_pcg update_xr: alpha = rz/pAp, x += alpha*p, r -= alpha*Ap. Alpha computed on device from d_rz, d_pAp.
void fused_update_xr(muda::CVarView<Float>         d_rz,
                     muda::CVarView<Float>         d_pAp,
                     muda::CVarView<IndexT>        d_converged,
                     muda::DenseVectorView<Float>  x,
                     muda::CDenseVectorView<Float> p,
                     muda::DenseVectorView<Float>  r,
                     muda::CDenseVectorView<Float> Ap)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [d_rz        = d_rz.cviewer().name("d_rz"),
                d_pAp       = d_pAp.cviewer().name("d_pAp"),
                d_converged = d_converged.cviewer().name("d_converged"),
                x           = x.viewer().name("x"),
                p           = p.cviewer().name("p"),
                r           = r.viewer().name("r"),
                Ap          = Ap.cviewer().name("Ap")] __device__(int i) mutable
               {
                   if(*d_converged != 0)
                       return;
                   Float alpha = *d_rz / *d_pAp;
                   x(i) += alpha * p(i);
                   r(i) -= alpha * Ap(i);
               });
}

// Same as linear_pcg update_p: p = z + beta*p.
// Beta is computed by fused_update_state; convergence is guarded by d_converged.
void fused_update_p(muda::CVarView<Float>         d_beta,
                    muda::CVarView<IndexT>        d_converged,
                    muda::DenseVectorView<Float>  p,
                    muda::CDenseVectorView<Float> z)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(p.size(),
               [d_beta      = d_beta.cviewer().name("d_beta"),
                d_converged = d_converged.cviewer().name("d_converged"),
                p           = p.viewer().name("p"),
                z           = z.cviewer().name("z")] __device__(int i) mutable
               {
                   if(*d_converged != 0)
                       return;
                   p(i) = z(i) + *d_beta * p(i);
               });
}

void fused_update_state(muda::CVarView<Float> d_rz_new,
                        muda::VarView<Float>  d_rz,
                        muda::VarView<Float>  d_beta,
                        muda::VarView<IndexT> d_converged,
                        Float                 rz_tol)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(1,
               [d_rz_new    = d_rz_new.cviewer().name("d_rz_new"),
                d_rz        = d_rz.viewer().name("d_rz"),
                d_beta      = d_beta.viewer().name("d_beta"),
                d_converged = d_converged.viewer().name("d_converged"),
                rz_tol] __device__(int) mutable
               {
                   Float rz_new = *d_rz_new;
                   *d_converged = abs(rz_new) <= rz_tol ? 1 : 0;
                   *d_beta      = rz_new / *d_rz;
                   *d_rz        = rz_new;
               });
}

SizeT LinearFusedPCG::fused_pcg(muda::DenseVectorView<Float>  x,
                                muda::CDenseVectorView<Float> b,
                                SizeT                         max_iter)
{
    Timer pcg_timer{"FusedPCG"};

    SizeT k     = 0;
    d_converged = 0;

    // r = b - A*x, but x0 = 0 so r = b
    r.buffer_view().copy_from(b.buffer_view());

    // z = P^{-1} * r
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z, r, d_converged.view());
    }

    // p = z
    p = z;

    // rz = r^T * z
    fused_dot(r.cview(), z.cview(), d_rz.view());
    Float rz_host = d_rz;
    check_init_rz_nan_inf(rz_host);
    Float abs_rz0 = std::abs(rz_host);

    if(abs_rz0 == Float{0.0})
        return 0;

    Float rz_tol = global_tol_rate * abs_rz0;
    SizeT effective_check_interval = check_interval > 0 ? check_interval : SizeT{1};

    for(k = 1; k < max_iter; ++k)
    {
        // Ap = A * p,  pAp = p^T * Ap
        {
            Timer timer{"SpMV"};
            spmv_dot(p.cview(), Ap.view(), d_pAp.view());
        }

        // alpha = rz / pAp,  x += alpha * p,  r -= alpha * Ap
        fused_update_xr(
            d_rz.view(), d_pAp.view(), d_converged.view(), x, p.cview(), r.view(), Ap.cview());

        // z = P^{-1} * r
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z, r, d_converged.view());
        }

        // rz_new = r^T * z, keep convergence flag on device for preconditioner skip.
        fused_dot(r.cview(), z.cview(), d_rz_new.view());
        fused_update_state(
            d_rz_new.view(), d_rz.view(), d_beta.view(), d_converged.view(), rz_tol);

        // Check error ratio periodically to avoid per-iteration D2H synchronization.
        bool do_check = (k % effective_check_interval == 0) || (k + 1 == max_iter);
        if(do_check)
        {
            Float rz_new_host = d_rz_new;
            check_iter_rz_nan_inf(rz_new_host, k);
            if((std::abs(rz_new_host) / abs_rz0) <= global_tol_rate)
                break;
        }

        // p = z + beta * p (skip when abs(rz_new) <= rz_tol).
        fused_update_p(d_beta.view(), d_converged.view(), p.view(), z.cview());
    }

    return k;
}
}  // namespace uipc::backend::cuda
