#include <app/app.h>

#include <muda/viewer/dense/dense_2d.h>
#include <backends/cuda/contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <backends/cuda/utils/make_spd.h>
#include <muda/buffer/device_buffer.h>
#include <muda/launch/parallel_for.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

using namespace muda;
using namespace uipc;
using namespace uipc::backend::cuda;

namespace
{
namespace contact  = uipc::backend::cuda::sym::codim_ipc_contact;
namespace friction = uipc::backend::cuda::friction;

constexpr Float Kappa     = 2.0;
constexpr Float DHat      = 0.5;
constexpr Float Thickness = 0.0;
constexpr Float EpsVh     = 0.125;
constexpr int   RCount    = 4;
constexpr int   MuCount   = 3;
constexpr int   CaseCount = RCount * MuCount;
constexpr int   MaxDof     = 12;

// Production assumes kappa > 0, d_hat > 0, a contact inside the barrier,
// and mu >= 0.  The negative-mu cases below are deliberately non-physical:
// they lock down the old full-H PSD projection's finite-input behavior.
UIPC_DEVICE Float case_r(int index)
{
    switch(index % RCount)
    {
        case 0: return 0.0;
        case 1: return EpsVh * 0.5;
        case 2: return EpsVh;
        default: return EpsVh * 2.0;
    }
}

UIPC_DEVICE Float case_mu(int index)
{
    switch(index / RCount)
    {
        case 0: return 0.5;
        case 1: return 0.0;
        default: return -0.5;
    }
}

struct FrictionCaseResult
{
    Float gradient[MaxDof];
    Float gradient_only[MaxDof];
    Float hessian[MaxDof * MaxDof];
    Float legacy_hessian[MaxDof * MaxDof];
    Float requested_r;
    Float measured_r;
    Float normal_force;
    Float signed_scale;
    int   dof;
};

template <int N>
UIPC_DEVICE void store_result(FrictionCaseResult&      result,
                              const Vector<Float, N>&  gradient,
                              const Vector<Float, N>&  gradient_only,
                              const Matrix<Float, N, N>& hessian,
                              Matrix<Float, N, N>        legacy_hessian,
                              Float                      requested_r,
                              const Vector2&             tan_rel_dx,
                              Float                      normal_force,
                              Float                      mu)
{
    uipc::backend::cuda::make_spd<N>(legacy_hessian);

    result.requested_r = requested_r;
    result.measured_r  = tan_rel_dx.norm();
    result.normal_force = normal_force;
    result.signed_scale = mu * normal_force;
    result.dof          = N;

#pragma unroll
    for(int i = 0; i < N; ++i)
    {
        result.gradient[i]      = gradient(i);
        result.gradient_only[i] = gradient_only(i);
#pragma unroll
        for(int j = 0; j < N; ++j)
        {
            result.hessian[i * MaxDof + j] = hessian(i, j);
            result.legacy_hessian[i * MaxDof + j] = legacy_hessian(i, j);
        }
    }
}

struct PTCase
{
    static constexpr int Dof = 12;
    static constexpr std::string_view Name = "PT";

    UIPC_DEVICE static void evaluate(FrictionCaseResult& result, Float r, Float mu)
    {
        const Vector3 prev_P{0.0, 0.0, 0.25};
        const Vector3 prev_T0{-1.0, -1.0, 0.0};
        const Vector3 prev_T1{1.0, -1.0, 0.0};
        const Vector3 prev_T2{0.0, 1.0, 0.0};

        Matrix<Float, 3, 2> move_basis;
        friction::point_triangle_tangent_basis(
            prev_P, prev_T0, prev_T1, prev_T2, move_basis);
        const Vector3 P = prev_P + r * move_basis.col(0);

        Vector12    G;
        Vector12    G_only;
        Matrix12x12 H;
        contact::PT_friction_gradient_hessian(G,
                                              H,
                                              Kappa,
                                              DHat,
                                              Thickness,
                                              mu,
                                              EpsVh,
                                              prev_P,
                                              prev_T0,
                                              prev_T1,
                                              prev_T2,
                                              P,
                                              prev_T0,
                                              prev_T1,
                                              prev_T2);
        contact::PT_friction_gradient(G_only,
                                      Kappa,
                                      DHat,
                                      Thickness,
                                      mu,
                                      EpsVh,
                                      prev_P,
                                      prev_T0,
                                      prev_T1,
                                      prev_T2,
                                      P,
                                      prev_T0,
                                      prev_T1,
                                      prev_T2);

        Float               force;
        Vector2             beta;
        Matrix<Float, 3, 2> basis;
        Vector2             tan_rel_dx;
        contact::PT_friction_basis(force,
                                   beta,
                                   basis,
                                   tan_rel_dx,
                                   Kappa,
                                   DHat,
                                   Thickness,
                                   prev_P,
                                   prev_T0,
                                   prev_T1,
                                   prev_T2,
                                   P,
                                   prev_T0,
                                   prev_T1,
                                   prev_T2);
        Matrix<Float, 2, Dof> J;
        friction::point_triangle_jacobi(basis, beta, J);
        Matrix2x2 H2;
        contact::friction_hessian(H2, mu, force, EpsVh, tan_rel_dx);
        Matrix12x12 legacy = J.transpose() * H2 * J;
        store_result(result, G, G_only, H, legacy, r, tan_rel_dx, force, mu);
    }
};

struct EECase
{
    static constexpr int Dof = 12;
    static constexpr std::string_view Name = "EE";

    UIPC_DEVICE static void evaluate(FrictionCaseResult& result, Float r, Float mu)
    {
        const Vector3 prev_Ea0{-1.0, 0.0, 0.25};
        const Vector3 prev_Ea1{1.0, 0.0, 0.25};
        const Vector3 prev_Eb0{0.0, -1.0, 0.0};
        const Vector3 prev_Eb1{0.0, 1.0, 0.0};

        Matrix<Float, 3, 2> move_basis;
        friction::edge_edge_tangent_basis(
            prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, move_basis);
        const Vector3 displacement = r * move_basis.col(0);
        const Vector3 Ea0          = prev_Ea0 + displacement;
        const Vector3 Ea1          = prev_Ea1 + displacement;

        Vector12    G;
        Vector12    G_only;
        Matrix12x12 H;
        contact::EE_friction_gradient_hessian(G,
                                              H,
                                              Kappa,
                                              DHat,
                                              Thickness,
                                              mu,
                                              EpsVh,
                                              prev_Ea0,
                                              prev_Ea1,
                                              prev_Eb0,
                                              prev_Eb1,
                                              Ea0,
                                              Ea1,
                                              prev_Eb0,
                                              prev_Eb1);
        contact::EE_friction_gradient(G_only,
                                      Kappa,
                                      DHat,
                                      Thickness,
                                      mu,
                                      EpsVh,
                                      prev_Ea0,
                                      prev_Ea1,
                                      prev_Eb0,
                                      prev_Eb1,
                                      Ea0,
                                      Ea1,
                                      prev_Eb0,
                                      prev_Eb1);

        Float               force;
        Vector2             gamma;
        Matrix<Float, 3, 2> basis;
        Vector2             tan_rel_dx;
        contact::EE_friction_basis(force,
                                   gamma,
                                   basis,
                                   tan_rel_dx,
                                   Kappa,
                                   DHat,
                                   Thickness,
                                   prev_Ea0,
                                   prev_Ea1,
                                   prev_Eb0,
                                   prev_Eb1,
                                   Ea0,
                                   Ea1,
                                   prev_Eb0,
                                   prev_Eb1);
        Matrix<Float, 2, Dof> J;
        friction::edge_edge_jacobi(basis, gamma, J);
        Matrix2x2 H2;
        contact::friction_hessian(H2, mu, force, EpsVh, tan_rel_dx);
        Matrix12x12 legacy = J.transpose() * H2 * J;
        store_result(result, G, G_only, H, legacy, r, tan_rel_dx, force, mu);
    }
};

struct PECase
{
    static constexpr int Dof = 9;
    static constexpr std::string_view Name = "PE";

    UIPC_DEVICE static void evaluate(FrictionCaseResult& result, Float r, Float mu)
    {
        const Vector3 prev_P{0.0, 0.0, 0.25};
        const Vector3 prev_E0{-1.0, 0.0, 0.0};
        const Vector3 prev_E1{1.0, 0.0, 0.0};

        Matrix<Float, 3, 2> move_basis;
        friction::point_edge_tangent_basis(prev_P, prev_E0, prev_E1, move_basis);
        const Vector3 P = prev_P + r * move_basis.col(0);

        Vector9   G;
        Vector9   G_only;
        Matrix9x9 H;
        contact::PE_friction_gradient_hessian(G,
                                              H,
                                              Kappa,
                                              DHat,
                                              Thickness,
                                              mu,
                                              EpsVh,
                                              prev_P,
                                              prev_E0,
                                              prev_E1,
                                              P,
                                              prev_E0,
                                              prev_E1);
        contact::PE_friction_gradient(G_only,
                                      Kappa,
                                      DHat,
                                      Thickness,
                                      mu,
                                      EpsVh,
                                      prev_P,
                                      prev_E0,
                                      prev_E1,
                                      P,
                                      prev_E0,
                                      prev_E1);

        Float               force;
        Float               eta;
        Matrix<Float, 3, 2> basis;
        Vector2             tan_rel_dx;
        contact::PE_friction_basis(force,
                                   eta,
                                   basis,
                                   tan_rel_dx,
                                   Kappa,
                                   DHat,
                                   Thickness,
                                   prev_P,
                                   prev_E0,
                                   prev_E1,
                                   P,
                                   prev_E0,
                                   prev_E1);
        Matrix<Float, 2, Dof> J;
        friction::point_edge_jacobi(basis, eta, J);
        Matrix2x2 H2;
        contact::friction_hessian(H2, mu, force, EpsVh, tan_rel_dx);
        Matrix9x9 legacy = J.transpose() * H2 * J;
        store_result(result, G, G_only, H, legacy, r, tan_rel_dx, force, mu);
    }
};

struct PPCase
{
    static constexpr int Dof = 6;
    static constexpr std::string_view Name = "PP";

    UIPC_DEVICE static void evaluate(FrictionCaseResult& result, Float r, Float mu)
    {
        const Vector3 prev_P0{0.0, 0.0, 0.25};
        const Vector3 prev_P1{0.0, 0.0, 0.0};

        Matrix<Float, 3, 2> move_basis;
        friction::point_point_tangent_basis(prev_P0, prev_P1, move_basis);
        const Vector3 P0 = prev_P0 + r * move_basis.col(0);

        Vector6   G;
        Vector6   G_only;
        Matrix6x6 H;
        contact::PP_friction_gradient_hessian(G,
                                              H,
                                              Kappa,
                                              DHat,
                                              Thickness,
                                              mu,
                                              EpsVh,
                                              prev_P0,
                                              prev_P1,
                                              P0,
                                              prev_P1);
        contact::PP_friction_gradient(G_only,
                                      Kappa,
                                      DHat,
                                      Thickness,
                                      mu,
                                      EpsVh,
                                      prev_P0,
                                      prev_P1,
                                      P0,
                                      prev_P1);

        Float               force;
        Matrix<Float, 3, 2> basis;
        Vector2             tan_rel_dx;
        contact::PP_friction_basis(force,
                                   basis,
                                   tan_rel_dx,
                                   Kappa,
                                   DHat,
                                   Thickness,
                                   prev_P0,
                                   prev_P1,
                                   P0,
                                   prev_P1);
        Matrix<Float, 2, Dof> J;
        friction::point_point_jacobi(basis, J);
        Matrix2x2 H2;
        contact::friction_hessian(H2, mu, force, EpsVh, tan_rel_dx);
        Matrix6x6 legacy = J.transpose() * H2 * J;
        store_result(result, G, G_only, H, legacy, r, tan_rel_dx, force, mu);
    }
};

template <typename ContactCase>
std::vector<FrictionCaseResult> evaluate_cases()
{
    DeviceBuffer<FrictionCaseResult> device_results(CaseCount);
    ParallelFor(32)
        .kernel_name("simplex_friction_hessian_lift_test")
        .apply(CaseCount,
               [results = device_results.viewer()] __device__(int index) mutable
               {
                   ContactCase::evaluate(results(index), case_r(index), case_mu(index));
               });

    std::vector<FrictionCaseResult> results;
    device_results.copy_to(results);
    return results;
}

bool near(Float actual, Float expected, Float abs_tol, Float rel_tol)
{
    const Float scale = std::max(std::abs(actual), std::abs(expected));
    return std::abs(actual - expected) <= abs_tol + rel_tol * scale;
}

template <typename ContactCase>
void validate_cases()
{
    constexpr Float ValueAbsTol = 2.0e-9;
    constexpr Float ValueRelTol = 2.0e-9;
    constexpr Float RadiusTol   = 32.0 * std::numeric_limits<Float>::epsilon();

    const auto results = evaluate_cases<ContactCase>();
    REQUIRE(results.size() == CaseCount);

    for(int case_index = 0; case_index < CaseCount; ++case_index)
    {
        const auto& result = results[case_index];
        INFO("contact=" << ContactCase::Name << ", case=" << case_index
                         << ", r=" << result.requested_r
                         << ", signed_scale=" << result.signed_scale);

        REQUIRE(result.dof == ContactCase::Dof);
        REQUIRE(std::isfinite(result.normal_force));
        REQUIRE(result.normal_force > 0.0);
        REQUIRE(std::isfinite(result.signed_scale));
        REQUIRE(near(result.measured_r, result.requested_r, RadiusTol, RadiusTol));

        Float max_abs_hessian = 0.0;
        for(int i = 0; i < ContactCase::Dof; ++i)
        {
            REQUIRE(std::isfinite(result.gradient[i]));
            REQUIRE(std::isfinite(result.gradient_only[i]));
            REQUIRE(near(result.gradient[i], result.gradient_only[i], 1.0e-12, 1.0e-12));

            for(int j = 0; j < ContactCase::Dof; ++j)
            {
                const Float actual = result.hessian[i * MaxDof + j];
                const Float legacy = result.legacy_hessian[i * MaxDof + j];
                REQUIRE(std::isfinite(actual));
                REQUIRE(std::isfinite(legacy));
                REQUIRE(near(actual, legacy, ValueAbsTol, ValueRelTol));
                REQUIRE(near(actual,
                             result.hessian[j * MaxDof + i],
                             ValueAbsTol,
                             ValueRelTol));
                max_abs_hessian = std::max(max_abs_hessian, std::abs(actual));
            }
        }

        Matrix<Float, ContactCase::Dof, ContactCase::Dof> hessian;
        for(int i = 0; i < ContactCase::Dof; ++i)
            for(int j = 0; j < ContactCase::Dof; ++j)
                hessian(i, j) = result.hessian[i * MaxDof + j];

        Eigen::SelfAdjointEigenSolver<decltype(hessian)> solver(
            0.5 * (hessian + hessian.transpose()));
        REQUIRE(solver.info() == Eigen::Success);
        const Float psd_tol = 2.0e-9 * std::max<Float>(1.0, max_abs_hessian);
        REQUIRE(solver.eigenvalues().minCoeff() >= -psd_tol);

        const int mu_index = case_index / RCount;
        if(mu_index == 0)
            REQUIRE(result.signed_scale > 0.0);
        else if(mu_index == 1)
            REQUIRE(result.signed_scale == 0.0);
        else
        {
            REQUIRE(result.signed_scale < 0.0);
            REQUIRE(max_abs_hessian <= ValueAbsTol);
        }
    }
}
}  // namespace

TEST_CASE("simplex friction Hessian lift matches the legacy full PSD projection",
          "[cuda][contact][friction][hessian]")
{
    SECTION("point-triangle")
    {
        validate_cases<PTCase>();
    }
    SECTION("edge-edge")
    {
        validate_cases<EECase>();
    }
    SECTION("point-edge")
    {
        validate_cases<PECase>();
    }
    SECTION("point-point")
    {
        validate_cases<PPCase>();
    }
}
