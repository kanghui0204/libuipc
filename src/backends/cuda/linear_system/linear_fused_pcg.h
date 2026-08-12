#pragma once
#include <linear_system/iterative_solver.h>
#include <muda/buffer/device_var.h>
#include <linear_system/fused_pcg_common.h>
#include <memory>

namespace uipc::backend::cuda
{
class LinearFusedPcgTestAccess;

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
    friend class LinearFusedPcgTestAccess;

    using DeviceDenseVector = muda::DeviceDenseVector<Float>;
    struct GraphResources;

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

    std::unique_ptr<GraphResources> graph_resources;

    muda::DeviceVar<Float>  d_rz;
    muda::DeviceVar<Float>  d_pAp;
    muda::DeviceVar<Float>  d_rz_new;
    muda::DeviceVar<IndexT> d_converged;

    Float max_iter_ratio  = 2.0;
    Float global_tol_rate = 1e-4;
    Float reserve_ratio   = 1.5;
    SizeT check_interval  = 5;

    bool  graph_enabled                = false;
    bool  fused_preconditioner_enabled = true;
    SizeT last_terminal_iteration      = 0;
    SizeT last_reported_iteration      = 0;
};
}  // namespace uipc::backend::cuda
