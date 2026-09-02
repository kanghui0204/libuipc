#include <contact_system/simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <contact_system/contact_models/ipc_simplex_normal_contact_energy.h>
#include <utils/distance/distance_flagged.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <utils/matrix_assembler.h>
#include <utils/four_vertex_translation_free_spd.h>
#include <utils/make_spd.h>
#include <utils/fixed_bank_soa_evd.h>
#include <utils/contact_type_block_layout.h>
#include <utils/primitive_d_hat.h>
#include <pipeline/ipc_pipeline_flag.h>

namespace uipc::backend::cuda
{
class IPCSimplexNormalContact final : public SimplexNormalContact
{
  public:
    using SimplexNormalContact::SimplexNormalContact;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        launch_ipc_simplex_normal_contact_energy(
            IPCSimplexNormalContactEnergyLaunchInfo{
                .contact_tabular     = info.contact_tabular(),
                .contact_element_ids = info.contact_element_ids(),
                .positions           = info.positions(),
                .rest_positions      = info.rest_positions(),
                .thicknesses         = info.thicknesses(),
                .d_hats              = info.d_hats(),
                .PTs                 = info.PTs(),
                .EEs                 = info.EEs(),
                .PEs                 = info.PEs(),
                .PPs                 = info.PPs(),
                .PT_energies         = info.PT_energies(),
                .EE_energies         = info.EE_energies(),
                .PE_energies         = info.PE_energies(),
                .PP_energies         = info.PP_energies(),
                .dt                  = info.dt()});
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_simplex_contact;

        // Fused kernel: PT + EE + PE + PP in one launch using offset-based dispatch.
        // Reduces 4 kernel launches to 1, improving GPU occupancy by providing
        // more threads in a single launch.
        auto pt_count = (IndexT)info.PTs().size();
        auto ee_count = (IndexT)info.EEs().size();
        auto pe_count = (IndexT)info.PEs().size();
        auto pp_count = (IndexT)info.PPs().size();
        auto total    = pt_count + ee_count + pe_count + pp_count;

        if(total == 0)
            return;

        // Keep each contact type in its own CTA. The padding lanes return
        // immediately instead of sharing a warp with the next contact formula.
        constexpr int BlockSize = 16;
        const auto layout = make_contact_type_block_layout<BlockSize>(
            pt_count, ee_count, pe_count, pp_count);
        const IndexT pt_end       = layout.pt_end;
        const IndexT ee_offset    = layout.ee_offset;
        const IndexT ee_end       = layout.ee_end;
        const IndexT pe_offset    = layout.pe_offset;
        const IndexT pe_end       = layout.pe_end;
        const IndexT pp_offset    = layout.pp_offset;
        const IndexT padded_total = layout.padded_total;

        auto assemble_kernel =
            [gradient_only = info.gradient_only(),
                 table = info.contact_tabular().viewer().name("contact_tabular"),
                 contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 Ps          = info.positions().viewer().name("Ps"),
                 rest_Ps     = info.rest_positions().viewer().name("rest_Ps"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 d_hats      = info.d_hats().viewer().name("d_hats"),
                 dt          = info.dt(),
                 // PT
                 PTs   = info.PTs().viewer().name("PTs"),
                 PT_Gs = info.PT_gradients().viewer().name("PT_Gs"),
                 PT_Hs = info.PT_hessians().viewer().name("PT_Hs"),
                 // EE
                 EEs   = info.EEs().viewer().name("EEs"),
                 EE_Gs = info.EE_gradients().viewer().name("EE_Gs"),
                 EE_Hs = info.EE_hessians().viewer().name("EE_Hs"),
                 // PE
                 PEs   = info.PEs().viewer().name("PEs"),
                 PE_Gs = info.PE_gradients().viewer().name("PE_Gs"),
                 PE_Hs = info.PE_hessians().viewer().name("PE_Hs"),
                 // PP
                 PPs   = info.PPs().viewer().name("PPs"),
                 PP_Gs = info.PP_gradients().viewer().name("PP_Gs"),
                 PP_Hs = info.PP_hessians().viewer().name("PP_Hs"),
                 // offsets
                 pt_end,
                 ee_offset,
                 ee_end,
                 pe_offset,
                 pe_end,
                pp_offset] __device__(IndexT idx) mutable
                {
                    constexpr int SharedLanePitch = 16;
                    __shared__ Float shared_h[12 * 12 * SharedLanePitch];

                    if(idx < pt_end)  // PT
                    {
                        int      i    = idx;
                        Vector4i PT   = PTs(i);
                        Vector4i cids = {contact_ids(PT[0]),
                                         contact_ids(PT[1]),
                                         contact_ids(PT[2]),
                                         contact_ids(PT[3])};
                        Float    kt2  = PT_kappa(table, cids) * dt * dt;

                        const auto& P  = Ps(PT[0]);
                        const auto& T0 = Ps(PT[1]);
                        const auto& T1 = Ps(PT[2]);
                        const auto& T2 = Ps(PT[3]);

                        Float thickness = PT_thickness(thicknesses(PT(0)),
                                                       thicknesses(PT(1)),
                                                       thicknesses(PT(2)),
                                                       thicknesses(PT(3)));
                        Float d_hat     = PT_d_hat(
                            d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));
                        Vector4i flag =
                            distance::point_triangle_distance_flag(P, T0, T1, T2);

                        Vector12 G;
                        if(gradient_only)
                        {
                            PT_barrier_gradient(G, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                            DoubletVectorAssembler DVA{PT_Gs};
                            DVA.segment<4>(i * 4).write(PT, G);
                        }
                        else
                        {
                            FixedBankSoAMap<12, SharedLanePitch> H(
                                shared_h + threadIdx.x);
                            Vector12 eigen_values;
                            PT_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                            selfadjoint_evd_four_vertex_translation_free_fixed_bank<
                                SharedLanePitch>(H, eigen_values);
                            DoubletVectorAssembler DVA{PT_Gs};
                            DVA.segment<4>(i * 4).write(PT, G);
                            TripletMatrixAssembler TMA{PT_Hs};
                            TMA.half_block<4>(i * PTHalfHessianSize)
                                .write_psd_from_eigendecomposition(PT, H, eigen_values);
                        }
                    }
                    else if(idx < ee_offset)
                    {
                        return;  // PT padding
                    }
                    else if(idx < ee_end)  // EE
                    {
                        int      i    = idx - ee_offset;
                        Vector4i EE   = EEs(i);
                        Vector4i cids = {contact_ids(EE[0]),
                                         contact_ids(EE[1]),
                                         contact_ids(EE[2]),
                                         contact_ids(EE[3])};
                        Float    kt2  = EE_kappa(table, cids) * dt * dt;

                        const auto& E0     = Ps(EE[0]);
                        const auto& E1     = Ps(EE[1]);
                        const auto& E2     = Ps(EE[2]);
                        const auto& E3     = Ps(EE[3]);
                        const auto& t0_Ea0 = rest_Ps(EE[0]);
                        const auto& t0_Ea1 = rest_Ps(EE[1]);
                        const auto& t0_Eb0 = rest_Ps(EE[2]);
                        const auto& t0_Eb1 = rest_Ps(EE[3]);

                        Float thickness = EE_thickness(thicknesses(EE(0)),
                                                       thicknesses(EE(1)),
                                                       thicknesses(EE(2)),
                                                       thicknesses(EE(3)));
                        Float d_hat     = EE_d_hat(
                            d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3)));
                        Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                        Vector12 G;
                        if(gradient_only)
                        {
                            mollified_EE_barrier_gradient(
                                G, flag, kt2, d_hat, thickness, t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, E0, E1, E2, E3);
                            DoubletVectorAssembler DVA{EE_Gs};
                            DVA.segment<4>(i * 4).write(EE, G);
                        }
                        else
                        {
                            FixedBankSoAMap<12, SharedLanePitch> H(
                                shared_h + threadIdx.x);
                            Vector12 eigen_values;
                            mollified_EE_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, E0, E1, E2, E3);
                            selfadjoint_evd_four_vertex_translation_free_fixed_bank<
                                SharedLanePitch>(H, eigen_values);
                            DoubletVectorAssembler DVA{EE_Gs};
                            DVA.segment<4>(i * 4).write(EE, G);
                            TripletMatrixAssembler TMA{EE_Hs};
                            TMA.half_block<4>(i * EEHalfHessianSize)
                                .write_psd_from_eigendecomposition(EE, H, eigen_values);
                        }
                    }
                    else if(idx < pe_offset)
                    {
                        return;  // EE padding
                    }
                    else if(idx < pe_end)  // PE
                    {
                        int      i    = idx - pe_offset;
                        Vector3i PE   = PEs(i);
                        Vector3i cids = {contact_ids(PE[0]),
                                         contact_ids(PE[1]),
                                         contact_ids(PE[2])};
                        Float    kt2  = PE_kappa(table, cids) * dt * dt;

                        const auto& P  = Ps(PE[0]);
                        const auto& E0 = Ps(PE[1]);
                        const auto& E1 = Ps(PE[2]);

                        Float thickness = PE_thickness(thicknesses(PE(0)),
                                                       thicknesses(PE(1)),
                                                       thicknesses(PE(2)));
                        Float d_hat =
                            PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));
                        Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                        Vector9 G;
                        if(gradient_only)
                        {
                            PE_barrier_gradient(G, flag, kt2, d_hat, thickness, P, E0, E1);
                            DoubletVectorAssembler DVA{PE_Gs};
                            DVA.segment<3>(i * 3).write(PE, G);
                        }
                        else
                        {
                            FixedBankSoAMap<9, SharedLanePitch> H(
                                shared_h + threadIdx.x);
                            Vector9 eigen_values;
                            PE_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, P, E0, E1);
                            selfadjoint_evd_fixed_bank_shared<9>(H, eigen_values);
                            DoubletVectorAssembler DVA{PE_Gs};
                            DVA.segment<3>(i * 3).write(PE, G);
                            TripletMatrixAssembler TMA{PE_Hs};
                            TMA.half_block<3>(i * PEHalfHessianSize)
                                .write_psd_from_eigendecomposition(PE, H, eigen_values);
                        }
                    }
                    else if(idx < pp_offset)
                    {
                        return;  // PE padding
                    }
                    else  // PP
                    {
                        int         i  = idx - pp_offset;
                        const auto& PP = PPs(i);
                        Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                        Float kt2 = PP_kappa(table, cids) * dt * dt;

                        const auto& P0 = Ps(PP[0]);
                        const auto& P1 = Ps(PP[1]);

                        Float thickness =
                            PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));
                        Float d_hat = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));
                        Vector2i flag = distance::point_point_distance_flag(P0, P1);

                        Vector6 G;
                        if(gradient_only)
                        {
                            PP_barrier_gradient(G, flag, kt2, d_hat, thickness, P0, P1);
                            DoubletVectorAssembler DVA{PP_Gs};
                            DVA.segment<2>(i * 2).write(PP, G);
                        }
                        else
                        {
                            FixedBankSoAMap<6, SharedLanePitch> H(
                                shared_h + threadIdx.x);
                            Vector6 eigen_values;
                            PP_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, P0, P1);
                            selfadjoint_evd_fixed_bank_shared<6>(H, eigen_values);
                            DoubletVectorAssembler DVA{PP_Gs};
                            DVA.segment<2>(i * 2).write(PP, G);
                            TripletMatrixAssembler TMA{PP_Hs};
                            TMA.half_block<2>(i * PPHalfHessianSize)
                                .write_psd_from_eigendecomposition(PP, H, eigen_values);
                        }
                    }
                };

        using AssembleCallable = std::decay_t<decltype(assemble_kernel)>;
        static const cudaError_t preferred_carveout_status =
            cudaFuncSetAttribute(
                reinterpret_cast<const void*>(
                    muda::details::parallel_for_kernel<
                        AssembleCallable,
                        muda::Default>),
                cudaFuncAttributePreferredSharedMemoryCarveout,
                32);
        checkCudaErrors(preferred_carveout_status);

        ParallelFor(BlockSize)
            .file_line(__FILE__, __LINE__)
            .apply(padded_total, std::move(assemble_kernel));
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContact);
}  // namespace uipc::backend::cuda
