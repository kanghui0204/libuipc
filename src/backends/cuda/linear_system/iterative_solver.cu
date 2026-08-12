#include <linear_system/iterative_solver.h>
#include <linear_system/global_linear_system.h>
namespace uipc::backend::cuda
{
void IterativeSolver::do_build()
{
    m_system = &require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    m_system->add_solver(this);
}

void IterativeSolver::spmv(Float                         a,
                           muda::CDenseVectorView<Float> x,
                           Float                         b,
                           muda::DenseVectorView<Float>  y)
{
    m_system->m_impl.spmv(a, x, b, y);
}

void IterativeSolver::spmv(muda::CDenseVectorView<Float> x, muda::DenseVectorView<Float> y)
{
    spmv(1.0, x, 0.0, y);
}

void IterativeSolver::spmv_dot(muda::CDenseVectorView<Float> x,
                               muda::DenseVectorView<Float>  y,
                               muda::VarView<Float>          d_dot)
{
    m_system->m_impl.spmv_dot(x, y, d_dot);
}

GlobalLinearSystem::CBCOOMatrixView IterativeSolver::fused_pcg_matrix_capacity_view() const
{
    return m_system->m_impl.fused_pcg_matrix_capacity_view();
}

SizeT IterativeSolver::fused_pcg_triplet_count() const
{
    return m_system->m_impl.bcoo_A.triplet_count();
}

SizeT IterativeSolver::fused_pcg_triplet_capacity() const
{
    return m_system->m_impl.bcoo_A.triplet_capacity();
}

bool IterativeSolver::supports_fused_pcg() const
{
    return m_system->m_impl.supports_fused_pcg();
}

SizeT IterativeSolver::fused_pcg_preconditioner_signature() const
{
    return m_system->m_impl.fused_pcg_preconditioner_signature();
}

void IterativeSolver::fused_pcg_update_apply_dot(muda::DenseVectorView<Float> x,
                                                 muda::CDenseVectorView<Float> p,
                                                 muda::DenseVectorView<Float> r,
                                                 muda::CDenseVectorView<Float> Ap,
                                                 muda::DenseVectorView<Float> z,
                                                 muda::CVarView<Float>  alpha,
                                                 muda::VarView<Float>   rz_new,
                                                 muda::CVarView<IndexT> status,
                                                 muda::CVarView<FusedPcgDeviceParams> params,
                                                 IndexT iteration_in_chunk,
                                                 cudaStream_t stream)
{
    m_system->m_impl.fused_pcg_update_apply_dot(
        x, p, r, Ap, z, alpha, rz_new, status, params, iteration_in_chunk, stream);
}

void IterativeSolver::fused_pcg_spmv_pipelined(muda::CDenseVectorView<Float> p,
                                               muda::DenseVectorView<Float>  Ap,
                                               muda::VarView<Float> pAp,
                                               muda::DenseVectorView<Float> next_Ap,
                                               muda::VarView<Float>   next_pAp,
                                               muda::CVarView<IndexT> status,
                                               muda::CVarView<FusedPcgDeviceParams> params,
                                               IndexT       iteration_in_chunk,
                                               SizeT        triplet_bucket,
                                               cudaStream_t stream)
{
    m_system->m_impl.spmver.rbk_sym_spmv_dot_pipelined(m_system->m_impl.fused_pcg_matrix_capacity_view(),
                                                       p,
                                                       Ap,
                                                       pAp,
                                                       next_Ap,
                                                       next_pAp,
                                                       status,
                                                       params,
                                                       iteration_in_chunk,
                                                       triplet_bucket,
                                                       stream);
}

void IterativeSolver::apply_preconditioner(muda::DenseVectorView<Float>  z,
                                           muda::CDenseVectorView<Float> r,
                                           muda::CVarView<IndexT>        converged)
{
    m_system->m_impl.apply_preconditioner(z, r, converged);
}

bool IterativeSolver::accuracy_statisfied(muda::DenseVectorView<Float> r)
{
    return m_system->m_impl.accuracy_statisfied(r);
}

muda::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_impl.ctx;
}

void IterativeSolver::solve(GlobalLinearSystem::SolvingInfo& info)
{
    do_solve(info);
}
}  // namespace uipc::backend::cuda
