#include <type_define.h>
#include <Eigen/Dense>
#include <linear_system/local_preconditioner.h>
#include <finite_element/finite_element_method.h>
#include <linear_system/global_linear_system.h>
#include <finite_element/fem_linear_subsystem.h>
#include <global_geometry/global_vertex_manager.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <uipc/geometry/simplicial_complex.h>
#include <linear_system/fused_pcg_kernels.h>

namespace uipc::backend::cuda
{
class FEMDiagPreconditioner : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    FiniteElementMethod* finite_element_method = nullptr;
    GlobalLinearSystem*  global_linear_system  = nullptr;
    FEMLinearSubsystem*  fem_linear_subsystem  = nullptr;

    muda::DeviceBuffer<Matrix3x3> diag_inv;

    virtual void do_build(BuildInfo& info) override
    {
        finite_element_method       = &require<FiniteElementMethod>();
        global_linear_system        = &require<GlobalLinearSystem>();
        fem_linear_subsystem        = &require<FEMLinearSubsystem>();
        auto& global_vertex_manager = require<GlobalVertexManager>();

        // If ANY FEM geometry has mesh_part, defer to FEMMASPreconditioner,
        // which handles both partitioned (MAS) and unpartitioned (diag) vertices.
        auto geo_slots = world().scene().geometries();
        for(SizeT i = 0; i < geo_slots.size(); i++)
        {
            auto& geo = geo_slots[i]->geometry();
            auto* sc  = geo.as<geometry::SimplicialComplex>();
            if(sc && sc->dim() >= 1)
            {
                auto mesh_part = sc->vertices().find<IndexT>("mesh_part");
                if(mesh_part)
                {
                    throw SimSystemException(
                        "FEMDiagPreconditioner: mesh_part found, "
                        "deferring to FEMMASPreconditioner.");
                }
            }
        }

        // This FEMDiagPreconditioner depends on FEMLinearSubsystem
        info.connect(fem_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace muda;

        diag_inv.resize(finite_element_method->xs().size());

        // 1) collect diagonal blocks
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.A().triplet_count(),
                   [triplet            = info.A().cviewer().name("triplet"),
                    diag_inv           = diag_inv.viewer().name("diag_inv"),
                    fem_segment_offset = info.dof_offset() / 3,
                    fem_segment_count = info.dof_count() / 3] __device__(int I) mutable
                   {
                       auto&& [g_i, g_j, H3x3] = triplet(I);

                       IndexT i = g_i - fem_segment_offset;
                       IndexT j = g_j - fem_segment_offset;

                       if(i >= fem_segment_count || j >= fem_segment_count)
                       {
                           return;
                       }

                       if(i == j)
                       {
                           diag_inv(i) = eigen::inverse(H3x3);
                       }
                   });
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        launch_fem_diag_preconditioner_apply(
            diag_inv.view(), info.r(), info.z(), info.converged());
    }

    virtual bool do_supports_fused_pcg() const override { return true; }

    virtual SizeT do_fused_pcg_signature() const override
    {
        constexpr SizeT Kind = 0x46454d5f504347ull;
        SizeT           seed = reinterpret_cast<SizeT>(diag_inv.data());
        seed ^= static_cast<SizeT>(diag_inv.size()) + Kind + (seed << 6) + (seed >> 2);
        return seed;
    }

    virtual void do_fused_pcg_apply(GlobalLinearSystem::FusedPcgIterationInfo& info) override
    {
        launch_fused_pcg_fem_update_apply_dot(diag_inv.view(),
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

REGISTER_SIM_SYSTEM(FEMDiagPreconditioner);
}  // namespace uipc::backend::cuda
