#include <app/app.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <muda/buffer/buffer_launch.h>
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/device_bcoo_vector.h>
#include <muda/ext/linear_system/device_doublet_vector.h>
#include <muda/launch/parallel_for.h>
#include <backends/cuda/dytopo_effect_system/dytopo_distribution.h>

namespace
{
using namespace uipc;
using namespace muda;
namespace distribution = uipc::backend::cuda::dytopo_distribution;

using Query  = distribution::DistributionQuery;
using Result = distribution::DistributionResult;

struct GradientWorkspace
{
    DeviceBuffer<Query>  queries;
    DeviceBuffer<Result> results;
    std::vector<Result>  host_results;
    std::vector<DeviceDoubletVector<Float, 3>> classified_gradients;
};

void check_queries(GradientWorkspace&          workspace,
                   IndexT                     extent,
                   const std::vector<IndexT>& indices,
                   const std::vector<Vector2i>& ranges,
                   bool                       has_gradient_work = true)
{
    DeviceBCOOVector<Float, 3> sorted_gradient;
    sorted_gradient.resize(extent, indices.size());

    std::vector<Vector3> values(indices.size());
    for(IndexT i = 0; i < static_cast<IndexT>(values.size()); ++i)
    {
        values[i] = Vector3{Float(i) + 0.25, Float(i) + 0.5, Float(i) + 0.75};
    }

    if(!indices.empty())
    {
        sorted_gradient.indices().copy_from(indices.data());
        sorted_gradient.values().copy_from(values.data());
    }

    std::vector<Query> queries;
    queries.reserve(ranges.size());
    for(const auto& range : ranges)
    {
        queries.push_back(distribution::make_distribution_query(
            range, Vector2i::Zero(), Vector2i::Zero()));
    }

    workspace.queries.resize(queries.size());
    workspace.results.resize(queries.size());
    workspace.host_results.assign(queries.size(),
                                  distribution::empty_distribution_result());
    workspace.classified_gradients.resize(queries.size());

    const bool launch_metadata = has_gradient_work && !indices.empty()
                                 && !queries.empty();
    if(launch_metadata)
    {
        BufferLaunch().copy(workspace.queries.view(), queries.data());

        ParallelFor()
            .kernel_name("dytopo_distribution_gradient_metadata_oracle")
            .apply(queries.size(),
                   [gradient_indices = sorted_gradient.indices().cviewer(),
                    queries = workspace.queries.cviewer(),
                    results = workspace.results.viewer(),
                    count = static_cast<IndexT>(indices.size())] __device__(IndexT I) mutable
                   {
                       auto result = distribution::empty_distribution_result();
                       result.gradient_entry_range =
                           distribution::query_sorted_gradient_range(
                               gradient_indices,
                               count,
                               queries(I).gradient_range);
                       results(I) = result;
                   });

        workspace.results.view().copy_to(workspace.host_results.data());
    }

    const auto full_view = std::as_const(sorted_gradient).view();
    REQUIRE(full_view.extent() == extent);
    REQUIRE(full_view.total_extent() == extent);
    REQUIRE(full_view.doublet_count() == indices.size());
    REQUIRE(full_view.total_doublet_count() == indices.size());

    for(IndexT q = 0; q < static_cast<IndexT>(ranges.size()); ++q)
    {
        const auto begin = std::lower_bound(indices.begin(),
                                            indices.end(),
                                            ranges[q].x());
        const auto end = std::lower_bound(indices.begin(),
                                          indices.end(),
                                          ranges[q].y());
        const auto expected_begin =
            launch_metadata ? static_cast<IndexT>(begin - indices.begin()) : 0;
        const auto expected_end =
            launch_metadata ? static_cast<IndexT>(end - indices.begin()) : 0;

        INFO("query=" << q << ", range=[" << ranges[q].x() << ", "
                      << ranges[q].y() << ")");
        const auto actual = workspace.host_results[q].gradient_entry_range;
        REQUIRE(workspace.host_results[q].hessian_selection_range
                == Vector2i::Zero());
        REQUIRE(actual.x() == expected_begin);
        REQUIRE(actual.y() == expected_end);

        auto& classified = workspace.classified_gradients[q];
        classified.resize(extent, expected_end - expected_begin);
        if(expected_begin != expected_end)
        {
            ParallelFor()
                .kernel_name("dytopo_distribution_gradient_materialize_oracle")
                .apply(expected_end - expected_begin,
                       [source = full_view.cviewer(),
                        destination = classified.viewer(),
                        begin = expected_begin] __device__(IndexT I) mutable
                       {
                           auto&& [index, value] =
                               source(begin + I);
                           destination(I).write(index, value);
                       });
        }

        const auto classified_view = std::as_const(classified).view();
        REQUIRE(classified_view.extent() == extent);
        REQUIRE(classified_view.total_extent() == extent);
        REQUIRE(classified_view.doublet_count() == expected_end - expected_begin);
        REQUIRE(classified_view.total_doublet_count()
                == expected_end - expected_begin);

        if(expected_begin == expected_end)
            continue;

        std::vector<IndexT>  selected_indices(expected_end - expected_begin);
        std::vector<Vector3> selected_values(expected_end - expected_begin);
        classified_view.indices().copy_to(selected_indices.data());
        classified_view.values().copy_to(selected_values.data());

        for(IndexT i = 0; i < expected_end - expected_begin; ++i)
        {
            REQUIRE(selected_indices[i] == indices[expected_begin + i]);
            for(IndexT axis = 0; axis < 3; ++axis)
                REQUIRE(selected_values[i](axis) == values[expected_begin + i](axis));
        }
    }
}
}  // namespace

TEST_CASE("DyTopo gradient batched lower-bound queries preserve classified buffers",
          "[cuda][dytopo]")
{
    GradientWorkspace workspace;

    check_queries(workspace,
                  32,
                  {},
                  {Vector2i{0, 0}, Vector2i{-5, 5}, Vector2i{7, 9}});

    check_queries(workspace,
                  32,
                  {1, 4, 8, 9, 20},
                  {
                      Vector2i{0, 32},   // all
                      Vector2i{1, 21},   // exact outer bounds
                      Vector2i{2, 8},    // gap and exclusive upper bound
                      Vector2i{8, 10},   // adjacent stored indices
                      Vector2i{4, 9},    // exact lower and upper bounds
                      Vector2i{5, 7},    // none inside a gap
                      Vector2i{21, 30},  // none after all entries
                      Vector2i{-5, 1},   // none before all entries
                      Vector2i{20, 21},  // one entry
                      Vector2i{9, 9},    // empty range
                      Vector2i{1, 9},    // overlap query A
                      Vector2i{4, 21},   // overlap query B
                  });

    check_queries(workspace,
                  8,
                  {3},
                  {Vector2i{0, 8}, Vector2i{3, 4}, Vector2i{0, 3}, Vector2i{4, 8}});

    // Reuse the same workspace through no-metadata work and then restore the
    // original full input. Independent receiver buffers must not expose stale
    // entries from the first call.
    check_queries(workspace,
                  32,
                  {1, 4, 8, 9, 20},
                  {Vector2i{0, 32}, Vector2i{4, 9}},
                  false);
    check_queries(workspace,
                  32,
                  {1, 4, 8, 9, 20},
                  {Vector2i{0, 32}, Vector2i{4, 9}});
}
