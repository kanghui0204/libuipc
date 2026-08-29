#include <contact_system/contact_models/ipc_simplex_normal_contact_energy.h>

#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <muda/launch/parallel_for.h>
#include <utils/codim_thickness.h>
#include <utils/distance/distance_flagged.h>
#include <utils/primitive_d_hat.h>

namespace uipc::backend::cuda
{
void launch_ipc_simplex_normal_contact_energy(
    const IPCSimplexNormalContactEnergyLaunchInfo& info)
{
    using namespace muda;
    using namespace sym::codim_ipc_simplex_contact;

    const IndexT pt_end = static_cast<IndexT>(info.PTs.size());
    const IndexT ee_end = pt_end + static_cast<IndexT>(info.EEs.size());
    const IndexT pe_end = ee_end + static_cast<IndexT>(info.PEs.size());
    const IndexT pp_end = pe_end + static_cast<IndexT>(info.PPs.size());

    if(pp_end == 0)
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(pp_end,
               [table = info.contact_tabular.viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids.viewer().name("contact_element_ids"),
                Ps          = info.positions.viewer().name("Ps"),
                rest_Ps     = info.rest_positions.viewer().name("rest_Ps"),
                thicknesses = info.thicknesses.viewer().name("thicknesses"),
                d_hats      = info.d_hats.viewer().name("d_hats"),
                PTs         = info.PTs.viewer().name("PTs"),
                EEs         = info.EEs.viewer().name("EEs"),
                PEs         = info.PEs.viewer().name("PEs"),
                PPs         = info.PPs.viewer().name("PPs"),
                PT_Es       = info.PT_energies.viewer().name("PT_Es"),
                EE_Es       = info.EE_energies.viewer().name("EE_Es"),
                PE_Es       = info.PE_energies.viewer().name("PE_Es"),
                PP_Es       = info.PP_energies.viewer().name("PP_Es"),
                dt          = info.dt,
                pt_end,
                ee_end,
                pe_end,
                pp_end] __device__(IndexT idx) mutable
               {
                   if(idx < pt_end)
                   {
                       const IndexT i  = idx;
                       Vector4i    PT = PTs(i);

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

                       Float d_hat = PT_d_hat(
                           d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));

                       Vector4i flag =
                           distance::point_triangle_distance_flag(P, T0, T1, T2);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_triangle_distance2(flag, P, T0, T1, T2, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PT[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PT(0),
                                       PT(1),
                                       PT(2),
                                       PT(3),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       PT_Es(i) =
                           PT_barrier_energy(flag, kt2, d_hat, thickness, P, T0, T1, T2);
                   }
                   else if(idx < ee_end)
                   {
                       const IndexT i  = idx - pt_end;
                       Vector4i    EE = EEs(i);

                       Vector4i cids = {contact_ids(EE[0]),
                                        contact_ids(EE[1]),
                                        contact_ids(EE[2]),
                                        contact_ids(EE[3])};
                       Float    kt2  = EE_kappa(table, cids) * dt * dt;

                       const auto& E0 = Ps(EE[0]);
                       const auto& E1 = Ps(EE[1]);
                       const auto& E2 = Ps(EE[2]);
                       const auto& E3 = Ps(EE[3]);

                       const auto& t0_Ea0 = rest_Ps(EE[0]);
                       const auto& t0_Ea1 = rest_Ps(EE[1]);
                       const auto& t0_Eb0 = rest_Ps(EE[2]);
                       const auto& t0_Eb1 = rest_Ps(EE[3]);

                       Float thickness = EE_thickness(thicknesses(EE(0)),
                                                      thicknesses(EE(1)),
                                                      thicknesses(EE(2)),
                                                      thicknesses(EE(3)));

                       Float d_hat = EE_d_hat(
                           d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3)));

                       Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);
                           Vector2 range = D_range(thickness, d_hat);
                           MUDA_ASSERT(is_active_D(range, D),
                                       "EE[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       EE(0),
                                       EE(1),
                                       EE(2),
                                       EE(3),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       EE_Es(i) = mollified_EE_barrier_energy(flag,
                                                              kt2,
                                                              d_hat,
                                                              thickness,
                                                              t0_Ea0,
                                                              t0_Ea1,
                                                              t0_Eb0,
                                                              t0_Eb1,
                                                              E0,
                                                              E1,
                                                              E2,
                                                              E3);
                   }
                   else if(idx < pe_end)
                   {
                       const IndexT i  = idx - ee_end;
                       Vector3i    PE = PEs(i);

                       Vector3i cids = {
                           contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
                       Float kt2 = PE_kappa(table, cids) * dt * dt;

                       const auto& P  = Ps(PE[0]);
                       const auto& E0 = Ps(PE[1]);
                       const auto& E1 = Ps(PE[2]);

                       Float thickness = PE_thickness(
                           thicknesses(PE(0)), thicknesses(PE(1)), thicknesses(PE(2)));

                       Float d_hat =
                           PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));

                       Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_edge_distance2(flag, P, E0, E1, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PE[%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PE(0),
                                       PE(1),
                                       PE(2),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       PE_Es(i) = PE_barrier_energy(flag, kt2, d_hat, thickness, P, E0, E1);
                   }
                   else if(idx < pp_end)
                   {
                       const IndexT i  = idx - pe_end;
                       Vector2i    PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       Float    kt2  = PP_kappa(table, cids) * dt * dt;

                       const auto& Pa = Ps(PP[0]);
                       const auto& Pb = Ps(PP[1]);

                       Float thickness =
                           PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));

                       Float d_hat = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));

                       Vector2i flag = distance::point_point_distance_flag(Pa, Pb);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_point_distance2(flag, Pa, Pb, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PP[%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PP(0),
                                       PP(1),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       PP_Es(i) = PP_barrier_energy(flag, kt2, d_hat, thickness, Pa, Pb);
                   }
               });
}
}  // namespace uipc::backend::cuda
