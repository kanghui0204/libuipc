#include <app/app.h>
#include <contact_system/contact_models/ipc_simplex_normal_contact_energy.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_buffer_2d.h>
#include <muda/launch/parallel_for.h>
#include <type_define.h>
#include <utils/codim_thickness.h>
#include <utils/distance/distance_flagged.h>
#include <utils/primitive_d_hat.h>

#include <cmath>
#include <vector>

using namespace muda;
using namespace uipc;
using namespace uipc::backend::cuda;

namespace
{
struct Counts
{
    int pt = 0;
    int ee = 0;
    int pe = 0;
    int pp = 0;
};

void launch_four_kernel_oracle(const IPCSimplexNormalContactEnergyLaunchInfo& info)
{
    using namespace sym::codim_ipc_simplex_contact;

    ParallelFor()
        .kernel_name("ls09_oracle_pt")
        .apply(info.PTs.size(),
               [table = info.contact_tabular.viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids.viewer().name("contact_element_ids"),
                PTs         = info.PTs.viewer().name("PTs"),
                Es          = info.PT_energies.viewer().name("Es"),
                Ps          = info.positions.viewer().name("Ps"),
                thicknesses = info.thicknesses.viewer().name("thicknesses"),
                d_hats      = info.d_hats.viewer().name("d_hats"),
                dt          = info.dt] __device__(int i) mutable
               {
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
                   Float d_hat = PT_d_hat(
                       d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));
                   Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);

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

                   Es(i) = PT_barrier_energy(flag, kt2, d_hat, thickness, P, T0, T1, T2);
               });

    ParallelFor()
        .kernel_name("ls09_oracle_ee")
        .apply(info.EEs.size(),
               [table = info.contact_tabular.viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids.viewer().name("contact_element_ids"),
                EEs         = info.EEs.viewer().name("EEs"),
                Es          = info.EE_energies.viewer().name("Es"),
                Ps          = info.positions.viewer().name("Ps"),
                rest_Ps     = info.rest_positions.viewer().name("rest_Ps"),
                thicknesses = info.thicknesses.viewer().name("thicknesses"),
                d_hats      = info.d_hats.viewer().name("d_hats"),
                dt          = info.dt] __device__(int i) mutable
               {
                   Vector4i EE   = EEs(i);
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

                   Es(i) = mollified_EE_barrier_energy(flag,
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
               });

    ParallelFor()
        .kernel_name("ls09_oracle_pe")
        .apply(info.PEs.size(),
               [table = info.contact_tabular.viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids.viewer().name("contact_element_ids"),
                PEs         = info.PEs.viewer().name("PEs"),
                Es          = info.PE_energies.viewer().name("Es"),
                Ps          = info.positions.viewer().name("Ps"),
                thicknesses = info.thicknesses.viewer().name("thicknesses"),
                d_hats      = info.d_hats.viewer().name("d_hats"),
                dt          = info.dt] __device__(int i) mutable
               {
                   Vector3i PE   = PEs(i);
                   Vector3i cids = {
                       contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
                   Float kt2 = PE_kappa(table, cids) * dt * dt;

                   const auto& P  = Ps(PE[0]);
                   const auto& E0 = Ps(PE[1]);
                   const auto& E1 = Ps(PE[2]);
                   Float thickness = PE_thickness(
                       thicknesses(PE(0)), thicknesses(PE(1)), thicknesses(PE(2)));
                   Float d_hat = PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));
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

                   Es(i) = PE_barrier_energy(flag, kt2, d_hat, thickness, P, E0, E1);
               });

    ParallelFor()
        .kernel_name("ls09_oracle_pp")
        .apply(info.PPs.size(),
               [table = info.contact_tabular.viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids.viewer().name("contact_element_ids"),
                PPs         = info.PPs.viewer().name("PPs"),
                Es          = info.PP_energies.viewer().name("Es"),
                Ps          = info.positions.viewer().name("Ps"),
                thicknesses = info.thicknesses.viewer().name("thicknesses"),
                d_hats      = info.d_hats.viewer().name("d_hats"),
                dt          = info.dt] __device__(int i) mutable
               {
                   Vector2i PP   = PPs(i);
                   Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                   Float    kt2  = PP_kappa(table, cids) * dt * dt;

                   const auto& Pa = Ps(PP[0]);
                   const auto& Pb = Ps(PP[1]);
                   Float thickness = PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));
                   Float d_hat     = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));
                   Vector2i flag   = distance::point_point_distance_flag(Pa, Pb);

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

                   Es(i) = PP_barrier_energy(flag, kt2, d_hat, thickness, Pa, Pb);
               });
}

template <typename T>
void resize_and_copy_prefix(DeviceBuffer<T>& buffer, const std::vector<T>& values, int count)
{
    buffer.resize(count);
    if(count > 0)
        buffer.view().copy_from(values.data());
}

class EnergyFixture
{
  public:
    static constexpr int MaxPerType = 5;

    EnergyFixture()
        : contact_tabular(Extent2D{1, 1})
    {
        ContactCoeff coeff;
        coeff.kappa = 3.25;
        coeff.mu    = 0.0;
        contact_tabular.view().copy_from(&coeff);

        std::vector<Vector3> positions;
        positions.reserve(65);

        for(int i = 0; i < MaxPerType; ++i)
        {
            const Float x = Float{3.0} * i;
            const Float d = Float{0.20} + Float{0.04} * i;
            const int   b = static_cast<int>(positions.size());
            positions.push_back(Vector3{x + Float{0.2}, Float{0.2}, d});
            positions.push_back(Vector3{x, Float{0.0}, Float{0.0}});
            positions.push_back(Vector3{x + Float{1.0}, Float{0.0}, Float{0.0}});
            positions.push_back(Vector3{x, Float{1.0}, Float{0.0}});
            PTs.push_back(Vector4i{b, b + 1, b + 2, b + 3});
        }

        for(int i = 0; i < MaxPerType; ++i)
        {
            const Float x = Float{20.0} + Float{3.0} * i;
            const Float d = Float{0.22} + Float{0.04} * i;
            const int   b = static_cast<int>(positions.size());
            positions.push_back(Vector3{x - Float{1.0}, Float{0.0}, Float{0.0}});
            positions.push_back(Vector3{x + Float{1.0}, Float{0.0}, Float{0.0}});
            positions.push_back(Vector3{x, Float{-1.0}, d});
            positions.push_back(Vector3{x, Float{1.0}, d});
            EEs.push_back(Vector4i{b, b + 1, b + 2, b + 3});
        }

        for(int i = 0; i < MaxPerType; ++i)
        {
            const Float x = Float{40.0} + Float{3.0} * i;
            const Float d = Float{0.24} + Float{0.04} * i;
            const int   b = static_cast<int>(positions.size());
            positions.push_back(Vector3{x, d, Float{0.0}});
            positions.push_back(Vector3{x - Float{1.0}, Float{0.0}, Float{0.0}});
            positions.push_back(Vector3{x + Float{1.0}, Float{0.0}, Float{0.0}});
            PEs.push_back(Vector3i{b, b + 1, b + 2});
        }

        for(int i = 0; i < MaxPerType; ++i)
        {
            const Float x = Float{60.0} + Float{3.0} * i;
            const Float d = Float{0.26} + Float{0.04} * i;
            const int   b = static_cast<int>(positions.size());
            positions.push_back(Vector3{x, Float{0.0}, Float{0.0}});
            positions.push_back(Vector3{x + d, Float{0.0}, Float{0.0}});
            PPs.push_back(Vector2i{b, b + 1});
        }

        std::vector<IndexT> contact_ids(positions.size(), 0);
        std::vector<Float>  thicknesses(positions.size(), 0.0);
        std::vector<Float>  d_hats(positions.size(), 1.0);

        d_positions.resize(positions.size());
        d_rest_positions.resize(positions.size());
        d_contact_ids.resize(contact_ids.size());
        d_thicknesses.resize(thicknesses.size());
        d_d_hats.resize(d_hats.size());
        d_positions.view().copy_from(positions.data());
        d_rest_positions.view().copy_from(positions.data());
        d_contact_ids.view().copy_from(contact_ids.data());
        d_thicknesses.view().copy_from(thicknesses.data());
        d_d_hats.view().copy_from(d_hats.data());
    }

    void compare(Counts counts)
    {
        REQUIRE(counts.pt >= 0);
        REQUIRE(counts.ee >= 0);
        REQUIRE(counts.pe >= 0);
        REQUIRE(counts.pp >= 0);
        REQUIRE(counts.pt <= MaxPerType);
        REQUIRE(counts.ee <= MaxPerType);
        REQUIRE(counts.pe <= MaxPerType);
        REQUIRE(counts.pp <= MaxPerType);

        resize_and_copy_prefix(d_PTs, PTs, counts.pt);
        resize_and_copy_prefix(d_EEs, EEs, counts.ee);
        resize_and_copy_prefix(d_PEs, PEs, counts.pe);
        resize_and_copy_prefix(d_PPs, PPs, counts.pp);

        resize_outputs(reference_PT, fused_PT, counts.pt);
        resize_outputs(reference_EE, fused_EE, counts.ee);
        resize_outputs(reference_PE, fused_PE, counts.pe);
        resize_outputs(reference_PP, fused_PP, counts.pp);

        auto reference_info = make_info(reference_PT, reference_EE, reference_PE, reference_PP);
        auto fused_info     = make_info(fused_PT, fused_EE, fused_PE, fused_PP);

        launch_four_kernel_oracle(reference_info);
        launch_ipc_simplex_normal_contact_energy(fused_info);

        compare_buffer(reference_PT, fused_PT);
        compare_buffer(reference_EE, fused_EE);
        compare_buffer(reference_PE, fused_PE);
        compare_buffer(reference_PP, fused_PP);
    }

  private:
    static void resize_outputs(DeviceBuffer<Float>& reference,
                               DeviceBuffer<Float>& fused,
                               int                  count)
    {
        reference.resize(count);
        fused.resize(count);
        if(count > 0)
        {
            reference.view().fill(Float{-111.0});
            fused.view().fill(Float{-222.0});
        }
    }

    static void compare_buffer(DeviceBuffer<Float>& reference, DeviceBuffer<Float>& fused)
    {
        std::vector<Float> reference_host(reference.size());
        std::vector<Float> fused_host(fused.size());
        if(!reference_host.empty())
        {
            reference.view().copy_to(reference_host.data());
            fused.view().copy_to(fused_host.data());
        }
        REQUIRE(reference_host == fused_host);
        for(Float energy : fused_host)
            REQUIRE(std::isfinite(energy));
    }

    IPCSimplexNormalContactEnergyLaunchInfo make_info(DeviceBuffer<Float>& pt_energy,
                                                       DeviceBuffer<Float>& ee_energy,
                                                       DeviceBuffer<Float>& pe_energy,
                                                       DeviceBuffer<Float>& pp_energy)
    {
        return IPCSimplexNormalContactEnergyLaunchInfo{
            .contact_tabular     = contact_tabular.view(),
            .contact_element_ids = d_contact_ids.view(),
            .positions           = d_positions.view(),
            .rest_positions      = d_rest_positions.view(),
            .thicknesses         = d_thicknesses.view(),
            .d_hats              = d_d_hats.view(),
            .PTs                 = d_PTs.view(),
            .EEs                 = d_EEs.view(),
            .PEs                 = d_PEs.view(),
            .PPs                 = d_PPs.view(),
            .PT_energies         = pt_energy.view(),
            .EE_energies         = ee_energy.view(),
            .PE_energies         = pe_energy.view(),
            .PP_energies         = pp_energy.view(),
            .dt                  = Float{0.1}};
    }

    DeviceBuffer2D<ContactCoeff> contact_tabular;
    DeviceBuffer<IndexT>         d_contact_ids;
    DeviceBuffer<Vector3>        d_positions;
    DeviceBuffer<Vector3>        d_rest_positions;
    DeviceBuffer<Float>          d_thicknesses;
    DeviceBuffer<Float>          d_d_hats;

    std::vector<Vector4i> PTs;
    std::vector<Vector4i> EEs;
    std::vector<Vector3i> PEs;
    std::vector<Vector2i> PPs;
    DeviceBuffer<Vector4i> d_PTs;
    DeviceBuffer<Vector4i> d_EEs;
    DeviceBuffer<Vector3i> d_PEs;
    DeviceBuffer<Vector2i> d_PPs;

    DeviceBuffer<Float> reference_PT;
    DeviceBuffer<Float> reference_EE;
    DeviceBuffer<Float> reference_PE;
    DeviceBuffer<Float> reference_PP;
    DeviceBuffer<Float> fused_PT;
    DeviceBuffer<Float> fused_EE;
    DeviceBuffer<Float> fused_PE;
    DeviceBuffer<Float> fused_PP;
};
}  // namespace

TEST_CASE("IPC simplex normal-contact energy four-launch oracle matches fused launch",
          "[cuda][line_search][energy][LS09]")
{
    EnergyFixture fixture;

    SECTION("all contact classes empty")
    {
        fixture.compare(Counts{});
    }

    SECTION("each contact class works alone")
    {
        fixture.compare(Counts{.pt = 3});
        fixture.compare(Counts{.ee = 3});
        fixture.compare(Counts{.pe = 3});
        fixture.compare(Counts{.pp = 3});
    }

    SECTION("mixed ranges cover both sides of every type boundary")
    {
        fixture.compare(Counts{.pt = 2, .ee = 3, .pe = 4, .pp = 5});
    }

    SECTION("empty interior ranges preserve cumulative offsets")
    {
        fixture.compare(Counts{.pt = 2, .pe = 3});
        fixture.compare(Counts{.ee = 2, .pp = 3});
    }

    SECTION("nonzero to zero to nonzero reuses output capacity safely")
    {
        fixture.compare(Counts{.pt = 5, .ee = 4, .pe = 3, .pp = 2});
        fixture.compare(Counts{});
        fixture.compare(Counts{.pt = 1, .ee = 2, .pe = 3, .pp = 4});
    }
}
