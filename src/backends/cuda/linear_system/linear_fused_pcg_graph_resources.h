#pragma once

#include <linear_system/linear_fused_pcg.h>

#include <array>

namespace uipc::backend::cuda
{
struct LinearFusedPCG::GraphResources
{
    std::array<DeviceDenseVector, 2>      graph_Ap;
    std::array<muda::DeviceVar<Float>, 2> graph_pAp;
    std::array<muda::DeviceVar<Float>, 2> graph_rz_old;
    std::array<muda::DeviceVar<Float>, 2> graph_rz_new;

    muda::DeviceVar<Float>                alpha;
    muda::DeviceVar<Float>                beta;
    muda::DeviceVar<FusedPcgDeviceParams> device_params;
    muda::DeviceVar<FusedPcgCheckState>   check_state;

    cudaStream_t           capture_stream     = nullptr;
    cudaGraphExec_t        graph_start_slot_0 = nullptr;
    cudaGraphExec_t        graph_start_slot_1 = nullptr;
    FusedPcgGraphSignature signature{};
    bool                   signature_valid = false;
    SizeT                  triplet_bucket  = 0;
    SizeT                  generation      = 0;

    ~GraphResources()
    {
        if(graph_start_slot_0)
            cudaGraphExecDestroy(graph_start_slot_0);
        if(graph_start_slot_1)
            cudaGraphExecDestroy(graph_start_slot_1);
        if(capture_stream)
            cudaStreamDestroy(capture_stream);
    }
};
}  // namespace uipc::backend::cuda
