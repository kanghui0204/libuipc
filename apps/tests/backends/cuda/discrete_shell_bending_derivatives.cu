#include <app/app.h>
#include <finite_element/constitutions/discrete_shell_bending_function.h>

using namespace uipc;
using namespace uipc::backend::cuda;

TEST_CASE("discrete shell bending combined derivatives",
          "[cuda][discrete_shell_bending]")
{
    namespace DSB = sym::discrete_shell_bending;
    const Vector3 x0{0.0, 1.0, 0.2};
    const Vector3 x1{-1.0, 0.0, 0.0};
    const Vector3 x2{1.0, 0.0, 0.0};
    const Vector3 x3{0.0, -1.0, -0.3};

    Vector12    separate_G;
    Vector12    combined_G;
    Matrix12x12 separate_H;
    Matrix12x12 combined_H;

    DSB::dEdx(separate_G, x0, x1, x2, x3, 1.2, 0.8, 0.15, 2.0);
    DSB::ddEddx(separate_H, x0, x1, x2, x3, 1.2, 0.8, 0.15, 2.0);
    DSB::d2Edx2(combined_G,
                combined_H,
                x0,
                x1,
                x2,
                x3,
                1.2,
                0.8,
                0.15,
                2.0);

    CHECK((combined_G.array() == separate_G.array()).all());
    CHECK((combined_H.array() == separate_H.array()).all());
}
