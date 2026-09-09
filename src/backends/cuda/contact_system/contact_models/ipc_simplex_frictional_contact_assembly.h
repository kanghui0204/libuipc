#pragma once

#include <uipc/backend/macro.h>
#include <contact_system/contact_coeff.h>
#include <cuda_tool/buffer.h>
#include <cuda_tool/linear_system/views.h>

namespace uipc::backend::cuda
{
struct IPCSimplexFrictionalContactAssemblyLaunchInfo
{
    cuda_tool::CBuffer2DView<ContactCoeff> contact_tabular;
    cuda_tool::CBufferView<IndexT>         contact_element_ids;
    cuda_tool::CBufferView<Vector3>        positions;
    cuda_tool::CBufferView<Vector3>        prev_positions;
    cuda_tool::CBufferView<Vector3>        rest_positions;
    cuda_tool::CBufferView<Float>          thicknesses;
    cuda_tool::CBufferView<Float>          d_hats;

    cuda_tool::CBufferView<Vector4i> PTs;
    cuda_tool::CBufferView<Vector4i> EEs;
    cuda_tool::CBufferView<Vector3i> PEs;
    cuda_tool::CBufferView<Vector2i> PPs;

    cuda_tool::DoubletVectorView<Float, 3> PT_gradients;
    cuda_tool::TripletMatrixView<Float, 3> PT_hessians;
    cuda_tool::DoubletVectorView<Float, 3> EE_gradients;
    cuda_tool::TripletMatrixView<Float, 3> EE_hessians;
    cuda_tool::DoubletVectorView<Float, 3> PE_gradients;
    cuda_tool::TripletMatrixView<Float, 3> PE_hessians;
    cuda_tool::DoubletVectorView<Float, 3> PP_gradients;
    cuda_tool::TripletMatrixView<Float, 3> PP_hessians;

    Float eps_velocity  = 0.0;
    Float dt            = 0.0;
    bool  gradient_only = false;
};

// The production dispatch, including its CTA12/Pitch16 full-Hessian kernel.
// Empty total is a no-op. Gradient-only accepts empty Hessian views.
UIPC_BACKEND_API void launch_ipc_simplex_frictional_contact_assembly(
    const IPCSimplexFrictionalContactAssemblyLaunchInfo& info);
}  // namespace uipc::backend::cuda
