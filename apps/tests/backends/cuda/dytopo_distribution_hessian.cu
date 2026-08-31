#include <app/app.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include <cub/iterator/counting_input_iterator.cuh>
#include <muda/buffer/buffer_launch.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <muda/cub/device/device_select.h>
#include <muda/ext/linear_system/device_bcoo_matrix.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>
#include <muda/launch/parallel_for.h>
#include <backends/cuda/dytopo_effect_system/dytopo_distribution.h>

namespace
{
using namespace uipc;
using namespace muda;
namespace distribution = uipc::backend::cuda::dytopo_distribution;

using Query  = distribution::DistributionQuery;
using Result = distribution::DistributionResult;

struct SelectionWorkspace
{
    DeviceBuffer<Query>  queries;
    DeviceBuffer<Result> results;
    DeviceBuffer<IndexT> selected_virtual_indices;
    DeviceVar<IndexT>    selected_count;

    std::vector<Result> host_results;
    std::vector<DeviceTripletMatrix<Float, 3>> classified_hessians;
};

template <typename T>
void loose_resize(DeviceBuffer<T>& buffer, SizeT size)
{
    if(size > buffer.capacity())
        buffer.reserve(size);
    buffer.resize(size);
}

void loose_resize_triplets(DeviceTripletMatrix<Float, 3>& matrix, SizeT size)
{
    if(size > matrix.triplet_capacity())
        matrix.reserve_triplets(size);
    matrix.resize_triplets(size);
}

bool query_is_empty(const Query& query)
{
    return query.hessian_i_range.x() == query.hessian_i_range.y()
           || query.hessian_j_range.x() == query.hessian_j_range.y();
}

bool query_contains(const Query& query, IndexT row, IndexT col)
{
    return row >= query.hessian_i_range.x()
           && row < query.hessian_i_range.y()
           && col >= query.hessian_j_range.x()
           && col < query.hessian_j_range.y();
}

std::vector<Query> standard_queries(IndexT extent)
{
    const IndexT split = (extent + 1) / 2;
    return {
        distribution::make_distribution_query(
            Vector2i::Zero(), {0, extent}, {0, extent}),
        distribution::make_distribution_query(
            Vector2i::Zero(), {0, split}, {0, split}),
        distribution::make_distribution_query(
            Vector2i::Zero(), {0, split}, {split, extent}),
        distribution::make_distribution_query(
            Vector2i::Zero(), {extent / 4, extent}, {0, 3 * extent / 4}),
        distribution::make_distribution_query(
            Vector2i::Zero(), {5, 5}, {0, extent}),
        distribution::make_distribution_query(
            Vector2i::Zero(), {extent + 3, extent + 7}, {0, extent}),
    };
}

void make_source(IndexT               extent,
                 IndexT               count,
                 std::vector<IndexT>& rows,
                 std::vector<IndexT>& cols,
                 std::vector<Matrix3x3>& values)
{
    rows.resize(count);
    cols.resize(count);
    values.resize(count);

    for(IndexT k = 0; k < count; ++k)
    {
        // Deliberately not lexicographically sorted. The batched selection must
        // preserve source order regardless of the row/column pattern.
        rows[k] = (7 * k + 3) % extent;
        cols[k] = (11 * k + 1) % extent;

        values[k] = Matrix3x3::Zero();
        for(IndexT row = 0; row < 3; ++row)
        {
            for(IndexT col = 0; col < 3; ++col)
                values[k](row, col) =
                    static_cast<Float>(100 * k + 10 * row + col);
        }
    }
}

void check_case(SelectionWorkspace&       workspace,
                IndexT                   extent,
                const std::vector<IndexT>& rows,
                const std::vector<IndexT>& cols,
                const std::vector<Matrix3x3>& values,
                const std::vector<Query>& queries,
                bool                     gradient_only)
{
    REQUIRE(rows.size() == cols.size());
    REQUIRE(rows.size() == values.size());

    const IndexT hessian_count = static_cast<IndexT>(rows.size());
    const SizeT  receiver_count_size = queries.size();
    IndexT       virtual_count        = -1;
    REQUIRE(distribution::checked_hessian_virtual_count(
        receiver_count_size, hessian_count, virtual_count));
    const IndexT receiver_count = static_cast<IndexT>(receiver_count_size);

    DeviceBCOOMatrix<Float, 3> source;
    source.resize(extent, extent, hessian_count);
    if(hessian_count > 0)
    {
        source.row_indices().copy_from(rows.data());
        source.col_indices().copy_from(cols.data());
        source.values().copy_from(values.data());
    }

    workspace.queries.resize(receiver_count_size);
    workspace.results.resize(receiver_count_size);
    workspace.host_results.assign(receiver_count_size,
                                  distribution::empty_distribution_result());
    workspace.classified_hessians.resize(receiver_count_size);

    const bool has_hessian_receiver =
        std::any_of(queries.begin(), queries.end(), [](const Query& query)
                    { return !query_is_empty(query); });
    const bool has_hessian_work = !gradient_only && has_hessian_receiver
                                  && hessian_count > 0
                                  && receiver_count > 0;

    loose_resize(workspace.selected_virtual_indices,
                 has_hessian_work ? virtual_count : 0);

    if(has_hessian_work)
    {
        BufferLaunch().copy(workspace.queries.view(), queries.data());

        const auto source_view = std::as_const(source).view();
        cub::CountingInputIterator<IndexT> virtual_indices{0};
        DeviceSelect().If(
            virtual_indices,
            workspace.selected_virtual_indices.data(),
            workspace.selected_count.data(),
            virtual_count,
            distribution::HessianRangePredicate{
                source_view.row_indices().data(),
                source_view.col_indices().data(),
                workspace.queries.data(),
                hessian_count});

        ParallelFor()
            .kernel_name("dytopo_hessian_batched_range_oracle")
            .apply(receiver_count,
                   [selected = workspace.selected_virtual_indices.cviewer(),
                    selected_count = workspace.selected_count.cviewer(),
                    results = workspace.results.viewer(),
                    hessian_count] __device__(IndexT receiver_index) mutable
                   {
                       auto result = distribution::empty_distribution_result();
                       result.hessian_selection_range =
                           distribution::query_selected_hessian_range(
                               selected,
                               *selected_count,
                               receiver_index,
                               hessian_count);
                       results(receiver_index) = result;
                   });

        workspace.results.view().copy_to(workspace.host_results.data());
    }

    std::vector<IndexT> expected_virtual_indices;
    std::vector<Vector2i> expected_ranges(receiver_count_size,
                                          Vector2i::Zero());
    if(has_hessian_work)
    {
        for(IndexT receiver = 0; receiver < receiver_count; ++receiver)
        {
            const IndexT begin =
                static_cast<IndexT>(expected_virtual_indices.size());
            for(IndexT k = 0; k < hessian_count; ++k)
            {
                if(query_contains(queries[receiver], rows[k], cols[k]))
                    expected_virtual_indices.push_back(receiver * hessian_count + k);
            }
            expected_ranges[receiver] = {
                begin, static_cast<IndexT>(expected_virtual_indices.size())};
        }
    }

    for(IndexT receiver = 0; receiver < receiver_count; ++receiver)
    {
        INFO("receiver=" << receiver << ", N=" << hessian_count
                          << ", gradient_only=" << gradient_only);
        REQUIRE(workspace.host_results[receiver].gradient_entry_range
                == Vector2i::Zero());
        REQUIRE(workspace.host_results[receiver].hessian_selection_range
                == expected_ranges[receiver]);
    }

    if(!expected_virtual_indices.empty())
    {
        std::vector<IndexT> actual_virtual_indices(expected_virtual_indices.size());
        workspace.selected_virtual_indices
            .view(0, expected_virtual_indices.size())
            .copy_to(actual_virtual_indices.data());
        REQUIRE(actual_virtual_indices == expected_virtual_indices);
    }

    // Mirror production's second stage: each receiver owns a separate output
    // buffer, and a selected q maps back to k = q - receiver*N.
    const auto source_view = std::as_const(source).view();
    for(IndexT receiver = 0; receiver < receiver_count; ++receiver)
    {
        auto& output = workspace.classified_hessians[receiver];
        output.reshape(extent, extent);

        const auto range = expected_ranges[receiver];
        const auto count = range.y() - range.x();
        loose_resize_triplets(output, count);

        if(count > 0)
        {
            ParallelFor()
                .kernel_name("dytopo_hessian_materialize_oracle")
                .apply(count,
                       [selected = workspace.selected_virtual_indices.cviewer(),
                        source = source_view.cviewer(),
                        output = output.viewer(),
                        begin = range.x(),
                        receiver,
                        hessian_count] __device__(IndexT I) mutable
                       {
                           const IndexT virtual_index = selected(begin + I);
                           const IndexT source_index =
                               virtual_index - receiver * hessian_count;
                           auto&& [row, col, value] = source(source_index);
                           output(I).write(row, col, value);
                       });
        }

        REQUIRE(output.rows() == extent);
        REQUIRE(output.cols() == extent);
        REQUIRE(output.triplet_count() == count);

        if(count == 0)
            continue;

        std::vector<IndexT> output_rows(count);
        std::vector<IndexT> output_cols(count);
        std::vector<Matrix3x3> output_values(count);
        output.row_indices().copy_to(output_rows.data());
        output.col_indices().copy_to(output_cols.data());
        output.values().copy_to(output_values.data());

        for(IndexT local_index = 0; local_index < count; ++local_index)
        {
            const IndexT virtual_index =
                expected_virtual_indices[range.x() + local_index];
            const IndexT source_index =
                virtual_index - receiver * hessian_count;
            REQUIRE(output_rows[local_index] == rows[source_index]);
            REQUIRE(output_cols[local_index] == cols[source_index]);
            for(IndexT row = 0; row < 3; ++row)
            {
                for(IndexT col = 0; col < 3; ++col)
                {
                    REQUIRE(output_values[local_index](row, col)
                            == values[source_index](row, col));
                }
            }
        }
    }
}
}  // namespace

TEST_CASE("DyTopo Hessian batched DeviceSelect is stable across boundaries and reuse",
          "[cuda][dytopo]")
{
    constexpr IndexT Extent = 17;
    const auto       queries = standard_queries(Extent);

    SelectionWorkspace workspace;
    std::vector<IndexT> rows;
    std::vector<IndexT> cols;
    std::vector<Matrix3x3> values;

    for(const IndexT count : {1, 31, 32, 33, 255, 256, 257, 1023})
    {
        make_source(Extent, count, rows, cols, values);
        check_case(workspace, Extent, rows, cols, values, queries, false);
    }

    const auto large_capacity = workspace.selected_virtual_indices.capacity();
    REQUIRE(large_capacity >= queries.size() * 1023);

    // Reuse one workspace through N=0 and gradient_only. Both paths must reset
    // logical state without shrinking capacity or exposing stale selections.
    make_source(Extent, 0, rows, cols, values);
    check_case(workspace, Extent, rows, cols, values, queries, false);
    REQUIRE(workspace.selected_virtual_indices.size() == 0);
    REQUIRE(workspace.selected_virtual_indices.capacity() == large_capacity);

    make_source(Extent, 33, rows, cols, values);
    check_case(workspace, Extent, rows, cols, values, queries, true);
    REQUIRE(workspace.selected_virtual_indices.size() == 0);
    REQUIRE(workspace.selected_virtual_indices.capacity() == large_capacity);

    make_source(Extent, 1, rows, cols, values);
    check_case(workspace, Extent, rows, cols, values, queries, false);
    REQUIRE(workspace.selected_virtual_indices.capacity() == large_capacity);

    // R=0 is a strict no-launch/no-transfer metadata path.
    make_source(Extent, 33, rows, cols, values);
    check_case(workspace, Extent, rows, cols, values, {}, false);
    REQUIRE(workspace.selected_virtual_indices.size() == 0);
    REQUIRE(workspace.selected_virtual_indices.capacity() == large_capacity);
}

TEST_CASE("DyTopo Hessian virtual count rejects signed and product overflow",
          "[cuda][dytopo]")
{
    constexpr IndexT Max = std::numeric_limits<IndexT>::max();
    IndexT           result = -1;

    REQUIRE(distribution::checked_hessian_virtual_count(0, Max, result));
    REQUIRE(result == 0);

    REQUIRE(distribution::checked_hessian_virtual_count(1, Max, result));
    REQUIRE(result == Max);

    REQUIRE_FALSE(distribution::checked_hessian_virtual_count(2, Max, result));
    REQUIRE(result == 0);

    REQUIRE_FALSE(distribution::checked_hessian_virtual_count(
        static_cast<SizeT>(Max) + SizeT{1}, 1, result));
    REQUIRE(result == 0);

    REQUIRE_FALSE(distribution::checked_hessian_virtual_count(1, -1, result));
    REQUIRE(result == 0);
}
