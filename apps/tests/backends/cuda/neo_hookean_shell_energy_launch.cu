#include <app/app.h>
#include <cuda_tool/cuda_tool.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>

#include <array>
#include <cmath>
#include <vector>

using namespace uipc;
using namespace uipc::backend::cuda;
using namespace uipc::backend::cuda_tool;

namespace
{
namespace NH = sym::neo_hookean_shell_2d;

__device__ Float sample_energy(int I)
{
    const Float s = Float(I % 17) * Float{1.0e-4};
    Vector9     X;
    X << Float{0.0}, Float{0.0}, Float{0.0}, Float{1.0} + s, Float{0.02},
        Float{0.0}, Float{0.01}, Float{1.0} - s, Float{0.0};

    Matrix2x2 IB = Matrix2x2::Identity();
    Float     E;
    NH::E(E, Float{2.0} + s, Float{3.0} - s, X, IB);
    return E;
}

__global__ void sample_energy_kernel(BufferView<Float> output, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I < n)
        output(I) = sample_energy(I);
}

void compare_default_and_cta64(int count)
{
    DeviceBuffer<Float> default_energy(count);
    DeviceBuffer<Float> cta64_energy(count);
    auto                kernel = sample_energy_kernel;
    kernel<<<best_grid_dim(count, kernel), best_block_dim(kernel), 0, nullptr>>>(
        default_energy.view(), count);
    kernel<<<(count + 63) / 64, 64, 0, nullptr>>>(cta64_energy.view(), count);

    std::vector<Float> default_host;
    std::vector<Float> cta64_host;
    default_energy.copy_to(default_host);
    cta64_energy.copy_to(cta64_host);
    REQUIRE(default_host == cta64_host);
    for(Float E : cta64_host)
        REQUIRE(std::isfinite(E));
}
}  // namespace

TEST_CASE("Neo-Hookean shell energy is invariant to CTA64 launch geometry",
          "[cuda][line_search][neo_energy_launch]")
{
    for(int count : std::array{1, 63, 64, 65, 127, 128, 129, 257, 4097})
        compare_default_and_cta64(count);
}
