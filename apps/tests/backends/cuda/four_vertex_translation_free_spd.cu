#include <app/app.h>

#include <muda/viewer/dense/dense_2d.h>
#include <backends/cuda/contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <backends/cuda/utils/four_vertex_translation_free_spd.h>
#include <backends/cuda/utils/make_spd.h>
#include <muda/buffer/device_buffer.h>
#include <muda/launch/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace muda;
using namespace uipc;
using namespace uipc::backend::cuda;

namespace
{
constexpr int PTCaseCount = 7;
constexpr int EECaseCount = 10;
constexpr int ParameterCount = 3;
constexpr int ContactCaseCount = PTCaseCount + EECaseCount;
constexpr int CaseCount = 1 + ParameterCount * ContactCaseCount;
constexpr int Dof         = 12;

struct ProjectionResult
{
    Float legacy[Dof * Dof];
    Float reduced[Dof * Dof];
    Float raw_asymmetry;
    Float raw_translation_residual;
    Float legacy_translation_residual;
    Float reduced_translation_residual;
};

UIPC_DEVICE Float absolute(Float value)
{
    return value < 0 ? -value : value;
}

UIPC_DEVICE Float max_value(Float lhs, Float rhs)
{
    return lhs > rhs ? lhs : rhs;
}

UIPC_DEVICE Float translation_residual(const Matrix12x12& H)
{
    Float residual = 0;
#pragma unroll
    for(int row = 0; row < Dof; ++row)
    {
#pragma unroll
        for(int axis = 0; axis < 3; ++axis)
        {
            Float sum = 0;
#pragma unroll
            for(int vertex = 0; vertex < 4; ++vertex)
                sum += H(row, 3 * vertex + axis);
            residual = max_value(residual, absolute(sum));
        }
    }
    return residual;
}

UIPC_DEVICE Float asymmetry(const Matrix12x12& H)
{
    Float result = 0;
#pragma unroll
    for(int row = 0; row < Dof; ++row)
#pragma unroll
        for(int col = 0; col < Dof; ++col)
            result = max_value(result, absolute(H(row, col) - H(col, row)));
    return result;
}

UIPC_DEVICE void make_synthetic(Matrix12x12& H)
{
    constexpr Float signs[4][3] = {
        {0.5, 0.5, 0.5},
        {0.5, -0.5, -0.5},
        {-0.5, 0.5, -0.5},
        {-0.5, -0.5, 0.5},
    };

    Matrix9x9 relative;
#pragma unroll
    for(int row = 0; row < 9; ++row)
    {
#pragma unroll
        for(int col = 0; col < 9; ++col)
        {
            if(row == col)
                relative(row, col) = static_cast<Float>(row - 4);
            else
                relative(row, col) = 0.03125 * static_cast<Float>(row + col + 1);
        }
    }

#pragma unroll
    for(int vertex_a = 0; vertex_a < 4; ++vertex_a)
#pragma unroll
        for(int axis_a = 0; axis_a < 3; ++axis_a)
#pragma unroll
            for(int vertex_b = 0; vertex_b < 4; ++vertex_b)
#pragma unroll
                for(int axis_b = 0; axis_b < 3; ++axis_b)
                {
                    Float value = 0;
#pragma unroll
                    for(int mode_a = 0; mode_a < 3; ++mode_a)
#pragma unroll
                        for(int mode_b = 0; mode_b < 3; ++mode_b)
                            value += signs[vertex_a][mode_a]
                                     * relative(3 * mode_a + axis_a,
                                                3 * mode_b + axis_b)
                                     * signs[vertex_b][mode_b];
                    H(3 * vertex_a + axis_a, 3 * vertex_b + axis_b) = value;
                }
}

UIPC_DEVICE Float case_kappa(int parameter_index)
{
    return parameter_index == 0 ? 0.25 : parameter_index == 1 ? 2.0 : 64.0;
}

UIPC_DEVICE Float case_thickness(int parameter_index)
{
    return parameter_index == 0 ? 0.0 : parameter_index == 1 ? 0.05 : 0.075;
}

UIPC_DEVICE void make_pt(Matrix12x12& H, int flag_index, int parameter_index)
{
    using namespace sym::codim_ipc_simplex_contact;
    const Vector4i flags[PTCaseCount] = {
        {1, 1, 0, 0},
        {1, 0, 1, 0},
        {1, 0, 0, 1},
        {1, 1, 1, 0},
        {1, 1, 0, 1},
        {1, 0, 1, 1},
        {1, 1, 1, 1},
    };
    const Vector3 P{0.0, 0.0, 0.1};
    const Vector3 T0{-1.0, -1.0, 0.0};
    const Vector3 T1{1.0, -1.0, 0.0};
    const Vector3 T2{0.0, 1.0, 0.0};
    Vector12      G;
    PT_barrier_gradient_hessian(G,
                                H,
                                flags[flag_index],
                                case_kappa(parameter_index),
                                5.0,
                                case_thickness(parameter_index),
                                P,
                                T0,
                                T1,
                                T2);
}

UIPC_DEVICE void make_ee(Matrix12x12& H, int flag_index, int parameter_index)
{
    using namespace sym::codim_ipc_simplex_contact;
    const Vector4i flags[EECaseCount - 1] = {
        {1, 0, 1, 0},
        {1, 0, 0, 1},
        {0, 1, 1, 0},
        {0, 1, 0, 1},
        {1, 0, 1, 1},
        {0, 1, 1, 1},
        {1, 1, 1, 0},
        {1, 1, 0, 1},
        {1, 1, 1, 1},
    };
    const Vector3 rest_Ea0{-1.0, 0.0, 0.1};
    const Vector3 rest_Ea1{1.0, 0.0, 0.1};
    const Vector3 rest_Eb0{0.0, -1.0, 0.0};
    const Vector3 rest_Eb1{0.0, 1.0, 0.0};

    Vector3 Ea0 = rest_Ea0;
    Vector3 Ea1 = rest_Ea1;
    Vector3 Eb0 = rest_Eb0;
    Vector3 Eb1 = rest_Eb1;
    Vector4i flag;
    if(flag_index < EECaseCount - 1)
        flag = flags[flag_index];
    else
    {
        // Force the mollifier's near-parallel branch while keeping the rest
        // geometry non-parallel so eps_x remains positive.
        Eb0  = Vector3{-0.75, 0.2, 0.0};
        Eb1  = Vector3{0.75, 0.2, 0.0};
        flag = {1, 0, 1, 1};
    }

    Vector12 G;
    mollified_EE_barrier_gradient_hessian(G,
                                           H,
                                           flag,
                                           case_kappa(parameter_index),
                                           5.0,
                                           case_thickness(parameter_index),
                                           rest_Ea0,
                                           rest_Ea1,
                                           rest_Eb0,
                                           rest_Eb1,
                                           Ea0,
                                           Ea1,
                                           Eb0,
                                           Eb1);
}

UIPC_DEVICE void evaluate(ProjectionResult& result, int case_index)
{
    Matrix12x12 raw;
    if(case_index == 0)
        make_synthetic(raw);
    else
    {
        const int contact_case   = case_index - 1;
        const int parameter_index = contact_case / ContactCaseCount;
        const int topology_case   = contact_case % ContactCaseCount;
        if(topology_case < PTCaseCount)
            make_pt(raw, topology_case, parameter_index);
        else
            make_ee(raw, topology_case - PTCaseCount, parameter_index);
    }

    Matrix12x12 legacy = raw;
    Matrix12x12 reduced = raw;
    make_spd<12>(legacy);
    make_spd_four_vertex_translation_free(reduced);

    result.raw_asymmetry             = asymmetry(raw);
    result.raw_translation_residual  = translation_residual(raw);
    result.legacy_translation_residual = translation_residual(legacy);
    result.reduced_translation_residual = translation_residual(reduced);

#pragma unroll
    for(int row = 0; row < Dof; ++row)
#pragma unroll
        for(int col = 0; col < Dof; ++col)
        {
            result.legacy[row * Dof + col] = legacy(row, col);
            result.reduced[row * Dof + col] = reduced(row, col);
        }
}
}  // namespace

TEST_CASE("four-vertex translation-free PSD projection matches full EVD",
          "[cuda][contact][normal][evd]")
{
    DeviceBuffer<ProjectionResult> device_results(CaseCount);

    ParallelFor(1).apply(CaseCount,
                         [results = device_results.viewer()] __device__(int index) mutable
                         {
                             evaluate(results(index), index);
                         });

    std::vector<ProjectionResult> results(CaseCount);
    device_results.copy_to(results);

    for(int case_index = 0; case_index < CaseCount; ++case_index)
    {
        const auto& result = results[case_index];
        Float       max_abs_difference = 0;
        Float       max_magnitude      = 0;
        bool        all_finite         = true;
        for(int i = 0; i < Dof * Dof; ++i)
        {
            all_finite = all_finite && std::isfinite(result.legacy[i])
                         && std::isfinite(result.reduced[i]);
            max_abs_difference =
                std::max(max_abs_difference,
                         std::abs(result.legacy[i] - result.reduced[i]));
            max_magnitude = std::max(
                max_magnitude,
                std::max(std::abs(result.legacy[i]), std::abs(result.reduced[i])));
        }

        const Float tolerance = 2e-9 * (1.0 + max_magnitude);
        INFO("case=" << case_index
                     << " (0=synthetic; then 3 parameter sets x 7 PT + 10 EE)");
        INFO("raw_asymmetry=" << result.raw_asymmetry);
        INFO("raw_translation_residual=" << result.raw_translation_residual);
        INFO("legacy_translation_residual=" << result.legacy_translation_residual);
        INFO("reduced_translation_residual=" << result.reduced_translation_residual);
        INFO("max_abs_difference=" << max_abs_difference);
        INFO("scale=" << max_magnitude);

        REQUIRE(all_finite);
        REQUIRE(result.raw_asymmetry <= 1e-12 * (1.0 + max_magnitude));
        REQUIRE(result.raw_translation_residual <= 1e-10 * (1.0 + max_magnitude));
        REQUIRE(result.reduced_translation_residual <= tolerance);
        REQUIRE(max_abs_difference <= tolerance);
    }
}
