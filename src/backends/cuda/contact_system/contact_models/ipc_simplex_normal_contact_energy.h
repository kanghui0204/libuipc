#pragma once

#include <contact_system/contact_coeff.h>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/buffer/buffer_view.h>
#include <type_define.h>

namespace uipc::backend::cuda
{
struct IPCSimplexNormalContactEnergyLaunchInfo
{
    muda::CBuffer2DView<ContactCoeff> contact_tabular;
    muda::CBufferView<IndexT>         contact_element_ids;
    muda::CBufferView<Vector3>        positions;
    muda::CBufferView<Vector3>        rest_positions;
    muda::CBufferView<Float>          thicknesses;
    muda::CBufferView<Float>          d_hats;

    muda::CBufferView<Vector4i> PTs;
    muda::CBufferView<Vector4i> EEs;
    muda::CBufferView<Vector3i> PEs;
    muda::CBufferView<Vector2i> PPs;

    muda::BufferView<Float> PT_energies;
    muda::BufferView<Float> EE_energies;
    muda::BufferView<Float> PE_energies;
    muda::BufferView<Float> PP_energies;

    Float dt = 0.0;
};

// Launches one kernel over the contiguous PT | EE | PE | PP index range.
// A zero total count is a no-op; empty interior ranges are skipped by the
// cumulative end offsets.
void launch_ipc_simplex_normal_contact_energy(
    const IPCSimplexNormalContactEnergyLaunchInfo& info);
}  // namespace uipc::backend::cuda
