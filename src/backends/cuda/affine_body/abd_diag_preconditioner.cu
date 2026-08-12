#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>
#include <linear_system/fused_pcg_kernels.h>

namespace uipc::backend::cuda
{
class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<Matrix12x12> diag_inv;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

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
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        launch_abd_diag_preconditioner_apply(
            diag_inv.view(), info.r(), info.z(), info.converged());
    }

    virtual bool do_supports_fused_pcg() const override { return true; }

    virtual SizeT do_fused_pcg_signature() const override
    {
        constexpr SizeT Kind = 0x4142445f504347ull;
        SizeT           seed = reinterpret_cast<SizeT>(diag_inv.data());
        seed ^= static_cast<SizeT>(diag_inv.size()) + Kind + (seed << 6) + (seed >> 2);
        return seed;
    }

    virtual void do_fused_pcg_apply(GlobalLinearSystem::FusedPcgIterationInfo& info) override
    {
        launch_fused_pcg_abd_update_apply_dot(diag_inv.view(),
                                              info.x(),
                                              info.p(),
                                              info.r(),
                                              info.Ap(),
                                              info.z(),
                                              info.alpha(),
                                              info.rz_new(),
                                              info.status(),
                                              info.params(),
                                              info.iteration_in_chunk(),
                                              info.stream());
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
