#pragma once

#include <uipc/backend/macro.h>
#include <cuda_tool/buffer.h>
#include <cuda_tool/linear_system/views.h>
#include <type_define.h>

namespace uipc::backend::cuda
{
// These are the production dispatch interfaces, shared by the constitutions
// and regression tests. Empty input is a no-op; gradient_only never uses H.
struct BendingGradientHessianLaunchInfo
{
    cuda_tool::BufferView<Vector4i> stencils;
    cuda_tool::BufferView<Float> stiffnesses, theta_bars, h_bars, volumes, rest_lengths;
    cuda_tool::CBufferView<Vector3> positions;
    cuda_tool::DoubletVectorView<Float, 3> gradients;
    cuda_tool::TripletMatrixView<Float, 3> hessians;
    Float dt = 0;
    bool gradient_only = false;
    cuda_tool::BufferView<Float> yield_stresses;
};
UIPC_BACKEND_API void launch_discrete_shell_bending_gradient_hessian(const BendingGradientHessianLaunchInfo&);
UIPC_BACKEND_API void launch_stress_plastic_bending_gradient_hessian(const BendingGradientHessianLaunchInfo&);

struct NeoHookeanShellGradientHessianLaunchInfo
{
    cuda_tool::CBufferView<Float> lambdas, mus;
    cuda_tool::CBufferView<Vector3i> indices;
    cuda_tool::CBufferView<Vector3> positions;
    cuda_tool::CBufferView<Matrix2x2> inverse_rest_shape_matrices;
    cuda_tool::CBufferView<Float> thicknesses;
    cuda_tool::DoubletVectorView<Float, 3> gradients;
    cuda_tool::TripletMatrixView<Float, 3> hessians;
    cuda_tool::CBufferView<Float> rest_areas;
    Float dt = 0;
    bool gradient_only = false;
};
UIPC_BACKEND_API void launch_neo_hookean_shell_gradient_hessian(const NeoHookeanShellGradientHessianLaunchInfo&);

struct OrthoPotentialGradientHessianLaunchInfo
{
    cuda_tool::CBufferView<Vector12> qs;
    cuda_tool::CBufferView<Float> volumes;
    cuda_tool::BufferView<Vector12> gradients;
    cuda_tool::BufferView<Matrix12x12> hessians;
    cuda_tool::CBufferView<Float> kappas;
    Float dt = 0;
    bool gradient_only = false;
};
UIPC_BACKEND_API void launch_ortho_potential_gradient_hessian(const OrthoPotentialGradientHessianLaunchInfo&);
}  // namespace uipc::backend::cuda
