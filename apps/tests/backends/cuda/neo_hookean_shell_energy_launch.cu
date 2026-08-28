#include <app/app.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>
#include <muda/buffer/device_buffer.h>

#include <array>
#include <cmath>
#include <vector>

using namespace uipc;
using namespace uipc::backend::cuda;

namespace
{
namespace NH = sym::neo_hookean_shell_2d;

__device__ Float sample_energy(int I)
{
    const Float s = Float(I % 17) * Float{1.0e-4};

    Vector9 X;
    X << Float{0.0}, Float{0.0}, Float{0.0},
        Float{1.0} + s, Float{0.02}, Float{0.0},
        Float{0.01}, Float{1.0} - s, Float{0.0};

    Matrix2x2 IB = Matrix2x2::Identity();
    Float     E;
    NH::E(E, Float{2.0} + s, Float{3.0} - s, X, IB);
    return E;
}

void compare_default_and_cta128(int count)
{
    muda::DeviceBuffer<Float> default_energy(count);
    muda::DeviceBuffer<Float> cta128_energy(count);

    muda::ParallelFor()
        .kernel_name("neo_shell_energy_default")
        .apply(count,
               [out = default_energy.viewer().name("default_energy")]
                   __device__(int I) mutable { out(I) = sample_energy(I); });

    muda::ParallelFor(128)
        .kernel_name("neo_shell_energy_cta128")
        .apply(count,
               [out = cta128_energy.viewer().name("cta128_energy")]
                   __device__(int I) mutable { out(I) = sample_energy(I); });

    std::vector<Float> default_host(count);
    std::vector<Float> cta128_host(count);
    default_energy.copy_to(default_host);
    cta128_energy.copy_to(cta128_host);

    REQUIRE(default_host == cta128_host);
    for(const Float E : cta128_host)
        REQUIRE(std::isfinite(E));
}
}  // namespace

TEST_CASE("Neo-Hookean shell energy is invariant to CTA128 launch geometry",
          "[cuda][line_search][neo_energy_launch]")
{
    for(const int count : std::array{1, 127, 128, 129, 257, 4097})
        compare_default_and_cta128(count);
}
