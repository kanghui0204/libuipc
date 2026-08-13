#pragma once

#include <type_define.h>
#include <cuda_runtime_api.h>

namespace uipc::backend::cuda
{
enum class FusedPcgStatus : IndexT
{
    Running   = 0,
    Converged = 1,
};

struct alignas(16) FusedPcgCheckState
{
    Float  rz                 = 0.0;
    IndexT status             = static_cast<IndexT>(FusedPcgStatus::Running);
    IndexT iteration_in_chunk = 0;
};

struct alignas(16) FusedPcgDeviceParams
{
    Float  tolerance         = 0.0;
    IndexT triplet_count     = 0;
    IndexT active_iterations = 0;
};

struct FusedPcgGraphSignature
{
    const void* matrix_rows              = nullptr;
    const void* matrix_cols              = nullptr;
    const void* matrix_values            = nullptr;
    const void* x                        = nullptr;
    const void* r                        = nullptr;
    const void* z                        = nullptr;
    const void* p                        = nullptr;
    const void* Ap_0                     = nullptr;
    const void* Ap_1                     = nullptr;
    const void* pAp_0                    = nullptr;
    const void* pAp_1                    = nullptr;
    const void* rz_old_0                 = nullptr;
    const void* rz_old_1                 = nullptr;
    const void* rz_new_0                 = nullptr;
    const void* rz_new_1                 = nullptr;
    const void* device_params            = nullptr;
    const void* status                   = nullptr;
    const void* check_state              = nullptr;
    const void* beta                     = nullptr;
    SizeT       scalar_dof_count         = 0;
    SizeT       triplet_bucket           = 0;
    SizeT       preconditioner_signature = 0;
    SizeT       check_interval           = 0;

    friend bool operator==(const FusedPcgGraphSignature&,
                           const FusedPcgGraphSignature&) = default;
};
}  // namespace uipc::backend::cuda
