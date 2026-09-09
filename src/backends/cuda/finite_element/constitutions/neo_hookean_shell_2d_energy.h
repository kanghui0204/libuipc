#pragma once

#include <cuda_tool/buffer.h>
#include <type_define.h>

namespace uipc::backend::cuda
{
struct NeoHookeanShell2DEnergyLaunchInfo
{
    cuda_tool::CBufferView<Float>     lambdas;
    cuda_tool::CBufferView<Float>     mus;
    cuda_tool::CBufferView<Float>     rest_areas;
    cuda_tool::CBufferView<Float>     thicknesses;
    cuda_tool::BufferView<Float>      energies;
    cuda_tool::CBufferView<Vector3i>  indices;
    cuda_tool::CBufferView<Vector3>   positions;
    cuda_tool::CBufferView<Matrix2x2> inverse_rest_shape_matrices;
    Float                             dt = 0.0;
};

// Production launch path for Neo-Hookean shell energy. A zero-sized indices
// view is a no-op.
void launch_neo_hookean_shell_2d_energy(
    const NeoHookeanShell2DEnergyLaunchInfo& info);
}  // namespace uipc::backend::cuda
