#pragma once
#include <limits>
#include <type_define.h>

namespace uipc::backend::cuda::dytopo_distribution
{
struct DistributionQuery
{
    Vector2i gradient_range;
    Vector2i hessian_i_range;
    Vector2i hessian_j_range;
};

struct DistributionResult
{
    Vector2i gradient_entry_range;
    Vector2i hessian_selection_range;
};

inline bool checked_hessian_virtual_count(SizeT   receiver_count,
                                          IndexT hessian_count,
                                          IndexT& virtual_count) noexcept
{
    virtual_count = 0;

    if(hessian_count < 0)
        return false;

    constexpr auto MaxIndexT = std::numeric_limits<IndexT>::max();
    if(receiver_count > static_cast<SizeT>(MaxIndexT))
        return false;

    if(hessian_count != 0
       && receiver_count
              > static_cast<SizeT>(MaxIndexT) / static_cast<SizeT>(hessian_count))
        return false;

    virtual_count = static_cast<IndexT>(receiver_count)
                    * static_cast<IndexT>(hessian_count);
    return true;
}

template <typename IndexViewer>
MUDA_GENERIC IndexT lower_bound_sorted_index(const IndexViewer& indices,
                                             IndexT             count,
                                             IndexT             value) noexcept
{
    IndexT first  = 0;
    IndexT length = count;

    while(length > 0)
    {
        const IndexT half = length / 2;
        const IndexT mid  = first + half;

        if(indices(mid) < value)
        {
            first = mid + 1;
            length -= half + 1;
        }
        else
        {
            length = half;
        }
    }

    return first;
}

template <typename IndexViewer>
MUDA_GENERIC Vector2i query_sorted_gradient_range(const IndexViewer& indices,
                                                  IndexT             count,
                                                  const Vector2i&    range) noexcept
{
    return {lower_bound_sorted_index(indices, count, range.x()),
            lower_bound_sorted_index(indices, count, range.y())};
}

template <typename SelectedIndexViewer>
MUDA_GENERIC Vector2i query_selected_hessian_range(
    const SelectedIndexViewer& selected_virtual_indices,
    IndexT                    selected_count,
    IndexT                    receiver_index,
    IndexT                    hessian_count) noexcept
{
    const IndexT begin_key = receiver_index * hessian_count;
    const IndexT end_key   = begin_key + hessian_count;

    return {lower_bound_sorted_index(
                selected_virtual_indices, selected_count, begin_key),
            lower_bound_sorted_index(
                selected_virtual_indices, selected_count, end_key)};
}

struct HessianRangePredicate
{
    const IndexT*             row_indices   = nullptr;
    const IndexT*             col_indices   = nullptr;
    const DistributionQuery* queries       = nullptr;
    IndexT                    hessian_count = 0;

    MUDA_GENERIC bool operator()(IndexT virtual_index) const noexcept
    {
        const IndexT receiver_index = virtual_index / hessian_count;
        const IndexT source_index =
            virtual_index - receiver_index * hessian_count;

        const auto& query = queries[receiver_index];
        const auto  i     = row_indices[source_index];
        const auto  j     = col_indices[source_index];

        return i >= query.hessian_i_range.x()
               && i < query.hessian_i_range.y()
               && j >= query.hessian_j_range.x()
               && j < query.hessian_j_range.y();
    }
};

MUDA_GENERIC inline DistributionQuery make_distribution_query(
    const Vector2i& gradient_range,
    const Vector2i& hessian_i_range,
    const Vector2i& hessian_j_range) noexcept
{
    return {gradient_range, hessian_i_range, hessian_j_range};
}

MUDA_GENERIC inline DistributionResult empty_distribution_result() noexcept
{
    return {Vector2i::Zero(), Vector2i::Zero()};
}
}  // namespace uipc::backend::cuda::dytopo_distribution
