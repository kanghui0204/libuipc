#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>
#include <linear_system/fused_pcg_kernels.h>
#include <Eigen/Cholesky>
#include <vector>

namespace uipc::backend::cuda
{
class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;
    muda::DeviceBuffer<Float>       full_inv;
    bool                            use_full_block = false;
    FullAbdApplyPolicy full_abd_apply_policy = FullAbdApplyPolicy::SingleLane;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        auto full_block_attr = world().scene().config().find<IndexT>(
            "linear_system/abd_full_block_preconditioner");
        use_full_block = full_block_attr && full_block_attr->view()[0] != 0;

        info.connect(abd_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace muda;

        auto diag_hessian = abd_linear_subsystem->diag_hessian();
        diag_inv.resize(diag_hessian.size());

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [diag_hessian = diag_hessian.viewer().name("diag_hessian"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   { diag_inv(i) = muda::eigen::inverse(diag_hessian(i)); });

        if(!use_full_block)
        {
            full_abd_apply_policy = FullAbdApplyPolicy::SingleLane;
            return;
        }

        auto A           = info.A();
        auto dof_offset  = info.dof_offset();
        auto dof_count   = info.dof_count();
        auto block_begin = dof_offset / 3;
        auto block_end   = block_begin + dof_count / 3;

        // Freeze the first positive-definite ABD block as a reference
        // preconditioner. PCG still multiplies by the current matrix and only
        // requires its fixed preconditioner to remain positive definite.
        if(full_inv.size() != 0)
        {
            const SizeT expected_size = dof_count * dof_count;
            if(full_inv.size() == expected_size)
                return;

            logger::info(
                "ABDDiagPreconditioner: Full-ABD size changed from {} to {}; "
                "rebuilding the frozen preconditioner",
                full_inv.size(),
                expected_size);
            full_inv.resize(0);
            full_abd_apply_policy = FullAbdApplyPolicy::SingleLane;
        }

        std::vector<IndexT>    rows(A.triplet_count());
        std::vector<IndexT>    cols(A.triplet_count());
        std::vector<Matrix3x3> values(A.triplet_count());
        A.row_indices().copy_to(rows.data());
        A.col_indices().copy_to(cols.data());
        A.values().copy_to(values.data());

        using MatrixX = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
        MatrixX dense = MatrixX::Zero(dof_count, dof_count);
        for(SizeT i = 0; i < A.triplet_count(); ++i)
        {
            auto row = rows[i];
            auto col = cols[i];
            if(row < block_begin || row >= block_end || col < block_begin || col >= block_end)
                continue;

            auto local_row = (row - block_begin) * 3;
            auto local_col = (col - block_begin) * 3;
            dense.template block<3, 3>(local_row, local_col) += values[i];
            if(row != col)
                dense.template block<3, 3>(local_col, local_row) += values[i].transpose();
        }

        Eigen::LDLT<MatrixX> ldlt(dense);
        if(ldlt.info() != Eigen::Success || !ldlt.isPositive())
        {
            logger::warn(
                "ABDDiagPreconditioner: full ABD block is not positive definite; "
                "falling back to 12x12 block Jacobi");
            full_inv.resize(0);
            full_abd_apply_policy = FullAbdApplyPolicy::SingleLane;
            return;
        }

        MatrixX inverse = ldlt.solve(MatrixX::Identity(dof_count, dof_count));
        if(ldlt.info() != Eigen::Success || !inverse.allFinite())
        {
            logger::warn(
                "ABDDiagPreconditioner: full ABD inverse is invalid; "
                "falling back to 12x12 block Jacobi");
            full_inv.resize(0);
            full_abd_apply_policy = FullAbdApplyPolicy::SingleLane;
            return;
        }

        full_inv.resize(dof_count * dof_count);
        full_inv.view().copy_from(inverse.data());
        full_abd_apply_policy = select_full_abd_apply_policy(dof_count);
        logger::info("ABDDiagPreconditioner: enabled full {}x{} ABD block with apply policy {}",
                     dof_count,
                     dof_count,
                     static_cast<int>(full_abd_apply_policy));
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        if(use_full_block && full_inv.size() != 0)
        {
            using namespace muda;
            auto converged = info.converged();
            auto dof_count = info.r().size();
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(dof_count,
                       [r         = info.r().viewer().name("r"),
                        z         = info.z().viewer().name("z"),
                        converged = converged.cviewer().name("converged"),
                        full_inv  = full_inv.viewer().name("full_inv"),
                        dof_count] __device__(int i) mutable
                       {
                           if(*converged != 0)
                               return;
                           Float value = 0.0;
                           for(int j = 0; j < dof_count; ++j)
                               value += full_inv(i + j * dof_count) * r(j);
                           z(i) = value;
                       });
            return;
        }

        launch_abd_diag_preconditioner_apply(
            diag_inv.view(), info.r(), info.z(), info.converged());
    }

    virtual bool do_supports_fused_pcg() const override { return true; }

    virtual SizeT do_fused_pcg_signature() const override
    {
        constexpr SizeT Kind    = 0x4142445f504347ull;
        SizeT           seed    = Kind;
        const auto      combine = [&seed](SizeT value)
        { seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2); };

        const bool full_mode = use_full_block && full_inv.size() != 0;
        combine(static_cast<SizeT>(full_mode));
        combine(reinterpret_cast<SizeT>(full_inv.data()));
        combine(static_cast<SizeT>(full_inv.size()));
        combine(static_cast<SizeT>(full_abd_apply_policy));
        combine(reinterpret_cast<SizeT>(diag_inv.data()));
        combine(static_cast<SizeT>(diag_inv.size()));
        return seed;
    }

    virtual void do_fused_pcg_apply(GlobalLinearSystem::FusedPcgIterationInfo& info) override
    {
        if(use_full_block && full_inv.size() != 0)
        {
            launch_fused_pcg_full_abd_update_apply_dot(full_inv.view(),
                                                       info.x(),
                                                       info.p(),
                                                       info.r(),
                                                       info.Ap(),
                                                       info.z(),
                                                       info.rz_old(),
                                                       info.pAp(),
                                                       info.rz_new(),
                                                       info.status(),
                                                       info.params(),
                                                       full_abd_apply_policy,
                                                       info.iteration_in_chunk(),
                                                       info.stream());
            return;
        }

        launch_fused_pcg_abd_update_apply_dot(diag_inv.view(),
                                              info.x(),
                                              info.p(),
                                              info.r(),
                                              info.Ap(),
                                              info.z(),
                                              info.rz_old(),
                                              info.pAp(),
                                              info.rz_new(),
                                              info.status(),
                                              info.params(),
                                              info.iteration_in_chunk(),
                                              info.stream());
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
