#pragma once
#include <sim_system.h>
#include <muda/ext/linear_system.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::cuda
{
class IterativeSolver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;

    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) = 0;


    /**********************************************************************************************
    * Util functions for derived classes
    ***********************************************************************************************/

    void spmv(Float a, muda::CDenseVectorView<Float> x, Float b, muda::DenseVectorView<Float> y);
    void spmv(muda::CDenseVectorView<Float> x, muda::DenseVectorView<Float> y);
    void spmv_dot(muda::CDenseVectorView<Float> x,
                  muda::DenseVectorView<Float>  y,
                  muda::VarView<Float>          d_dot);
    GlobalLinearSystem::CBCOOMatrixView fused_pcg_matrix_capacity_view() const;
    SizeT                               fused_pcg_triplet_count() const;
    SizeT                               fused_pcg_triplet_capacity() const;
    bool                                supports_fused_pcg() const;
    SizeT fused_pcg_preconditioner_signature() const;
    void  fused_pcg_update_apply_dot(muda::DenseVectorView<Float>         x,
                                     muda::CDenseVectorView<Float>        p,
                                     muda::DenseVectorView<Float>         r,
                                     muda::CDenseVectorView<Float>        Ap,
                                     muda::DenseVectorView<Float>         z,
                                     muda::CVarView<Float>                rz_old,
                                     muda::CVarView<Float>                pAp,
                                     muda::VarView<Float>                 rz_new,
                                     muda::CVarView<IndexT>               status,
                                     muda::CVarView<FusedPcgDeviceParams> params,
                                     IndexT       iteration_in_chunk,
                                     cudaStream_t stream);
    void  fused_pcg_spmv_pipelined(muda::CDenseVectorView<Float>        p,
                                   muda::DenseVectorView<Float>         Ap,
                                   muda::VarView<Float>                 pAp,
                                   muda::DenseVectorView<Float>         next_Ap,
                                   muda::VarView<Float>                 next_pAp,
                                   muda::CVarView<IndexT>               status,
                                   muda::CVarView<FusedPcgDeviceParams> params,
                                   IndexT       iteration_in_chunk,
                                   SizeT        triplet_bucket,
                                   cudaStream_t stream);
    void  apply_preconditioner(muda::DenseVectorView<Float>  z,
                               muda::CDenseVectorView<Float> r,
                               muda::CVarView<IndexT>        converged);
    bool  accuracy_statisfied(muda::DenseVectorView<Float> r);
    muda::LinearSystemContext& ctx() const;

  private:
    friend class GlobalLinearSystem;
    GlobalLinearSystem* m_system;

    virtual void do_build() final override;

    void solve(GlobalLinearSystem::SolvingInfo& info);
};
}  // namespace uipc::backend::cuda
