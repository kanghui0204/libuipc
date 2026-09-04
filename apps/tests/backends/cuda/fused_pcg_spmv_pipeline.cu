#include <app/app.h>
#include <linear_system/spmv.h>

#include <array>

namespace cuda_tool = uipc::backend::cuda_tool;
using namespace uipc;
using namespace uipc::backend::cuda;

TEST_CASE("fused PCG pipelined SpMV preserves result and clears the next slot",
          "[build_solve_focused][fused_pcg][spmv_pipeline]")
{
    constexpr int BlockRows = 2;
    constexpr int Dofs      = BlockRows * 3;

    const std::array<int, 3> rows = {0, 0, 1};
    const std::array<int, 3> cols = {0, 1, 1};
    std::array<Matrix3x3, 3> values;
    values[0] = Float{2} * Matrix3x3::Identity();
    values[1] = Matrix3x3::Identity();
    values[2] = Float{3} * Matrix3x3::Identity();

    cuda_tool::DeviceBCOOMatrix<Float, 3> A;
    A.resize(BlockRows, BlockRows, rows.size());
    A.row_indices().copy_from(rows.data());
    A.col_indices().copy_from(cols.data());
    A.values().copy_from(values.data());

    const std::array<Float, Dofs> x_host = {1, 2, 3, 4, 5, 6};
    cuda_tool::DeviceDenseVector<Float> x;
    cuda_tool::DeviceDenseVector<Float> y;
    cuda_tool::DeviceDenseVector<Float> next_y;
    x.resize(Dofs);
    y.resize(Dofs);
    next_y.resize(Dofs);
    x.buffer_view().copy_from(x_host.data());
    y.buffer_view().fill(Float{0});
    next_y.buffer_view().fill(Float{7});

    cuda_tool::DeviceVar<Float>  dot;
    cuda_tool::DeviceVar<Float>  next_dot;
    cuda_tool::DeviceVar<IndexT> converged;
    cuda_tool::DeviceVar<IndexT> triplet_count;
    dot           = Float{0};
    next_dot      = Float{11};
    converged     = IndexT{0};
    triplet_count = static_cast<IndexT>(rows.size());

    Spmv().rbk_sym_spmv_dot_pipelined(Float{1},
                                      A.cview(),
                                      x.cview(),
                                      y.view(),
                                      dot.view(),
                                      next_y.view(),
                                      next_dot.view(),
                                      converged.view(),
                                      triplet_count.cviewer(),
                                      A.triplet_capacity(),
                                      nullptr);

    const std::array<Float, Dofs> expected_y = {6, 9, 12, 13, 17, 21};
    std::array<Float, Dofs>       actual_y{};
    std::array<Float, Dofs>       actual_next_y{};
    y.buffer_view().copy_to(actual_y.data());
    next_y.buffer_view().copy_to(actual_next_y.data());
    for(int i = 0; i < Dofs; ++i)
    {
        REQUIRE(actual_y[i] == Catch::Approx(expected_y[i]).margin(1e-12));
        REQUIRE(actual_next_y[i] == Float{0});
    }
    REQUIRE(static_cast<Float>(dot) == Catch::Approx(323.0).margin(1e-12));
    REQUIRE(static_cast<Float>(next_dot) == Float{0});

    // Once converged, neither the output nor the next-slot clear is allowed
    // to run. This is the skip used by later inactive iterations in a replay.
    y.buffer_view().fill(Float{13});
    next_y.buffer_view().fill(Float{17});
    dot       = Float{19};
    next_dot  = Float{23};
    converged = IndexT{1};

    Spmv().rbk_sym_spmv_dot_pipelined(Float{1},
                                      A.cview(),
                                      x.cview(),
                                      y.view(),
                                      dot.view(),
                                      next_y.view(),
                                      next_dot.view(),
                                      converged.view(),
                                      triplet_count.cviewer(),
                                      A.triplet_capacity(),
                                      nullptr);

    y.buffer_view().copy_to(actual_y.data());
    next_y.buffer_view().copy_to(actual_next_y.data());
    for(int i = 0; i < Dofs; ++i)
    {
        REQUIRE(actual_y[i] == Float{13});
        REQUIRE(actual_next_y[i] == Float{17});
    }
    REQUIRE(static_cast<Float>(dot) == Float{19});
    REQUIRE(static_cast<Float>(next_dot) == Float{23});
}
