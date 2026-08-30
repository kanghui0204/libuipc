#include <type_define.h>
#include <collision_detection/info_stackless_bvh.h>
#include <muda/buffer.h>
#include <algorithm>
#include <vector>

using namespace muda;
using namespace uipc;
using namespace uipc::backend::cuda;

namespace test_info_stackless_bvh_refit
{
struct PairSet
{
    std::vector<Vector2i> values;

    void sort()
    {
        std::ranges::sort(values,
                          [](const Vector2i& lhs, const Vector2i& rhs)
                          {
                              return lhs[0] < rhs[0]
                                     || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
                          });
    }
};

void check_same_pairs(PairSet lhs, PairSet rhs)
{
    lhs.sort();
    rhs.sort();
    REQUIRE(lhs.values.size() == rhs.values.size());
    for(SizeT i = 0; i < lhs.values.size(); ++i)
    {
        CHECK(lhs.values[i][0] == rhs.values[i][0]);
        CHECK(lhs.values[i][1] == rhs.values[i][1]);
    }
}

auto node_pred(muda::CBuffer2DView<IndexT> cmts)
{
    return [cmts = cmts.viewer().name("cmts")] __device__(
               const InfoStacklessBVH::NodePredInfo& info)
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        bool bid_cull = info.query_bid != invalid && info.node_bid != invalid
                        && info.query_bid == info.node_bid;
        bool cid_cull = info.query_cid != invalid && info.node_cid != invalid
                        && !cmts(info.query_cid, info.node_cid);
        return !(bid_cull || cid_cull);
    };
}

auto leaf_pred(muda::CBuffer2DView<IndexT> cmts)
{
    return [cmts = cmts.viewer().name("cmts")] __device__(
               const InfoStacklessBVH::LeafPredInfo& info)
    {
        return info.bid_i != info.bid_j && cmts(info.cid_i, info.cid_j);
    };
}

PairSet detect_pairs(InfoStacklessBVH&          bvh,
                     muda::CBuffer2DView<IndexT> cmts)
{
    InfoStacklessBVH::QueryBuffer pairs;
    pairs.reserve(1);  // force the production overflow/retry path
    bvh.detect(cmts, node_pred(cmts), leaf_pred(cmts), pairs);

    PairSet result;
    result.values.resize(pairs.size());
    if(!result.values.empty())
        pairs.view().copy_to(result.values.data());
    return result;
}

PairSet query_pairs(InfoStacklessBVH&          bvh,
                    muda::CBufferView<AABB>    query_aabbs,
                    muda::CBufferView<IndexT>  query_bids,
                    muda::CBufferView<IndexT>  query_cids,
                    muda::CBuffer2DView<IndexT> cmts)
{
    InfoStacklessBVH::QueryBuffer pairs;
    pairs.reserve(1);  // force the production overflow/retry path
    bvh.query(query_aabbs,
              query_bids,
              query_cids,
              cmts,
              node_pred(cmts),
              leaf_pred(cmts),
              pairs);

    PairSet result;
    result.values.resize(pairs.size());
    if(!result.values.empty())
        pairs.view().copy_to(result.values.data());
    return result;
}

AABB box_at(double x, double radius)
{
    AABB box;
    Vector3 lo{x - radius, -radius, -radius};
    Vector3 hi{x + radius, radius, radius};
    box.extend(lo.cast<float>()).extend(hi.cast<float>());
    return box;
}

struct Inputs
{
    std::vector<AABB>   aabbs;
    std::vector<IndexT> bids;
    std::vector<IndexT> cids;
};

Inputs initial_inputs(IndexT count)
{
    Inputs result;
    result.aabbs.reserve(count);
    result.bids.reserve(count);
    result.cids.reserve(count);
    for(IndexT i = 0; i < count; ++i)
    {
        result.aabbs.push_back(box_at(3.0 * i, 0.25));
        result.bids.push_back(i % 7);
        result.cids.push_back(i % 4);
    }
    return result;
}

Inputs swept_inputs(IndexT count, double phase)
{
    Inputs result;
    result.aabbs.reserve(count);
    result.bids.reserve(count);
    result.cids.reserve(count);
    for(IndexT i = 0; i < count; ++i)
    {
        // Reverse the spatial order and make many boxes overlap. This proves
        // that refit remains conservative even when the old Morton topology is
        // a poor fit for the swept boxes.
        double x = 0.015 * (count - 1 - i) + phase * ((i % 3) - 1);
        result.aabbs.push_back(box_at(x, 0.65));
        result.bids.push_back((i + 3) % 9);
        result.cids.push_back((i + 1) % 4);
    }
    return result;
}

void upload(const Inputs&            input,
            muda::DeviceBuffer<AABB>&   aabbs,
            muda::DeviceBuffer<IndexT>& bids,
            muda::DeviceBuffer<IndexT>& cids)
{
    aabbs.resize(input.aabbs.size());
    bids.resize(input.bids.size());
    cids.resize(input.cids.size());
    if(!input.aabbs.empty())
    {
        aabbs.view().copy_from(input.aabbs.data());
        bids.view().copy_from(input.bids.data());
        cids.view().copy_from(input.cids.data());
    }
}

void compare_full_build_and_refit(const Inputs& initial,
                                  const Inputs& swept,
                                  muda::CBuffer2DView<IndexT> cmts)
{
    DeviceBuffer<AABB>   refit_aabbs;
    DeviceBuffer<IndexT> refit_bids;
    DeviceBuffer<IndexT> refit_cids;
    upload(initial, refit_aabbs, refit_bids, refit_cids);

    InfoStacklessBVH refitted;
    refitted.build(refit_aabbs, refit_bids, refit_cids);

    upload(swept, refit_aabbs, refit_bids, refit_cids);
    REQUIRE(refitted.refit(refit_aabbs, refit_bids, refit_cids));

    DeviceBuffer<AABB>   build_aabbs;
    DeviceBuffer<IndexT> build_bids;
    DeviceBuffer<IndexT> build_cids;
    upload(swept, build_aabbs, build_bids, build_cids);

    InfoStacklessBVH rebuilt;
    rebuilt.build(build_aabbs, build_bids, build_cids);

    check_same_pairs(detect_pairs(refitted, cmts), detect_pairs(rebuilt, cmts));

    auto query = swept_inputs(11, 0.025);
    DeviceBuffer<AABB>   query_aabbs;
    DeviceBuffer<IndexT> query_bids;
    DeviceBuffer<IndexT> query_cids;
    upload(query, query_aabbs, query_bids, query_cids);
    check_same_pairs(query_pairs(
                         refitted, query_aabbs, query_bids, query_cids, cmts),
                     query_pairs(
                         rebuilt, query_aabbs, query_bids, query_cids, cmts));
}
}  // namespace test_info_stackless_bvh_refit

TEST_CASE("info_stackless_bvh_refit", "[LS11][line_search][bvh]")
{
    using namespace test_info_stackless_bvh_refit;

    constexpr IndexT cid_count = 4;
    std::vector<IndexT> cmts(cid_count * cid_count);
    for(IndexT i = 0; i < cid_count; ++i)
        for(IndexT j = 0; j < cid_count; ++j)
            cmts[i * cid_count + j] = ((i + j) % 3) != 0;
    DeviceBuffer2D<IndexT> d_cmts(Extent2D{cid_count, cid_count});
    d_cmts.view().copy_from(cmts.data());

    SECTION("empty")
    {
        DeviceBuffer<AABB>   aabbs;
        DeviceBuffer<IndexT> bids;
        DeviceBuffer<IndexT> cids;
        InfoStacklessBVH     bvh;
        bvh.build(aabbs, bids, cids);
        CHECK(bvh.refit(aabbs, bids, cids));
        CHECK(detect_pairs(bvh, d_cmts.view()).values.empty());
    }

    SECTION("single_primitive")
    {
        compare_full_build_and_refit(
            initial_inputs(1), swept_inputs(1, 0.0), d_cmts.view());
    }

    SECTION("swept_topology_and_metadata")
    {
        compare_full_build_and_refit(
            initial_inputs(33), swept_inputs(33, 0.0), d_cmts.view());
    }

    SECTION("repeated_refit")
    {
        auto initial = initial_inputs(33);
        auto swept0  = swept_inputs(33, 0.0);
        auto swept1  = swept_inputs(33, 0.04);

        DeviceBuffer<AABB>   aabbs;
        DeviceBuffer<IndexT> bids;
        DeviceBuffer<IndexT> cids;
        upload(initial, aabbs, bids, cids);
        InfoStacklessBVH refitted;
        refitted.build(aabbs, bids, cids);

        upload(swept0, aabbs, bids, cids);
        REQUIRE(refitted.refit(aabbs, bids, cids));
        upload(swept1, aabbs, bids, cids);
        REQUIRE(refitted.refit(aabbs, bids, cids));

        DeviceBuffer<AABB>   rebuilt_aabbs;
        DeviceBuffer<IndexT> rebuilt_bids;
        DeviceBuffer<IndexT> rebuilt_cids;
        upload(swept1, rebuilt_aabbs, rebuilt_bids, rebuilt_cids);
        InfoStacklessBVH rebuilt;
        rebuilt.build(rebuilt_aabbs, rebuilt_bids, rebuilt_cids);
        check_same_pairs(detect_pairs(refitted, d_cmts.view()),
                         detect_pairs(rebuilt, d_cmts.view()));
    }

    SECTION("count_change_fails_closed")
    {
        auto initial = initial_inputs(8);
        DeviceBuffer<AABB>   aabbs;
        DeviceBuffer<IndexT> bids;
        DeviceBuffer<IndexT> cids;
        upload(initial, aabbs, bids, cids);
        InfoStacklessBVH bvh;
        bvh.build(aabbs, bids, cids);

        auto changed = swept_inputs(9, 0.0);
        upload(changed, aabbs, bids, cids);
        CHECK_FALSE(bvh.refit(aabbs, bids, cids));
    }
}
