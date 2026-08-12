#pragma once
#include <linear_system/iterative_solver.h>
#include <muda/buffer/device_var.h>
#include <linear_system/fused_pcg_common.h>
#include <array>

namespace uipc::backend::cuda
{
// Fused PCG: keeps dot-product scalars (rz, pAp, rz_new) on device
// to eliminate per-iteration host synchronizations.  The update kernels read
// alpha = rz/pAp and beta = rz_new/rz directly from device memory.
// SpMV and dot(p,Ap) are fused into a single kernel pass.
// Convergence is checked every `check_interval` iterations via a single D2H copy.
class LinearFusedPCG : public IterativeSolver
{
  public:
    using IterativeSolver::IterativeSolver;
    ~LinearFusedPCG() override;

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;

  private:
    using DeviceDenseVector = muda::DeviceDenseVector<Float>;

    SizeT fused_pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
    SizeT graph_fused_pcg(muda::DenseVectorView<Float>  x,
                          muda::CDenseVectorView<Float> b,
                          SizeT                         max_iter);
    void  launch_graph_iteration(muda::DenseVectorView<Float> x,
                                 IndexT                       current_slot,
                                 IndexT                       iteration_in_chunk,
                                 cudaStream_t                 stream);
    void  rebuild_graphs(muda::DenseVectorView<Float>  x,
                         const FusedPcgGraphSignature& signature);
    void  destroy_graphs() noexcept;
    void  capture_graph(cudaGraphExec_t&             graph_exec,
                        muda::DenseVectorView<Float> x,
                        IndexT                       iteration_count,
                        IndexT                       starting_slot);
    FusedPcgGraphSignature current_graph_signature(muda::DenseVectorView<Float> x) const;
    SizeT current_triplet_bucket() const;
    void  check_init_rz_nan_inf(Float rz);
    void  check_iter_rz_nan_inf(Float rz, SizeT k);

    DeviceDenseVector r;
    DeviceDenseVector z;
    DeviceDenseVector p;
    DeviceDenseVector Ap;

    std::array<DeviceDenseVector, 2> graph_Ap;

    std::array<muda::DeviceVar<Float>, 2> graph_pAp;
    std::array<muda::DeviceVar<Float>, 2> graph_rz_old;
    std::array<muda::DeviceVar<Float>, 2> graph_rz_new;

    muda::DeviceVar<Float>                d_alpha;
    muda::DeviceVar<Float>                d_beta;
    muda::DeviceVar<FusedPcgDeviceParams> d_graph_params;
    muda::DeviceVar<FusedPcgCheckState>   d_check_state;

    muda::DeviceVar<Float>  d_rz;
    muda::DeviceVar<Float>  d_pAp;
    muda::DeviceVar<Float>  d_rz_new;
    muda::DeviceVar<IndexT> d_converged;

    Float max_iter_ratio  = 2.0;
    Float global_tol_rate = 1e-4;
    Float reserve_ratio   = 1.5;
    SizeT check_interval  = 10;

    bool  graph_enabled                = true;
    bool  fused_preconditioner_enabled = true;
    SizeT last_terminal_iteration      = 0;
    SizeT last_reported_iteration      = 0;

    cudaStream_t           graph_capture_stream = nullptr;
    cudaGraphExec_t        graph_start_slot_0   = nullptr;
    cudaGraphExec_t        graph_start_slot_1   = nullptr;
    FusedPcgGraphSignature graph_signature{};
    bool                   graph_signature_valid = false;
    SizeT                  graph_triplet_bucket  = 0;
};
}  // namespace uipc::backend::cuda
