#include <type_define.h>
#include <app/app.h>
#include <app/asset_dir.h>
#include <collision_detection/info_stackless_bvh.h>
#include <uipc/geometry.h>
#include <uipc/uipc.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/timer.h>
#include <algorithm>
#include <array>
#include <iterator>
#include <list>
#include <utility>
#include <vector>

namespace cuda_tool = uipc::backend::cuda_tool;
using namespace cuda_tool;
using namespace uipc;
using namespace uipc::geometry;
using namespace uipc::backend::cuda;

namespace test_info_stackless_bvh
{
struct NodePred
{
    UIPC_GENERIC bool operator()(const InfoStacklessBVH::NodePredInfo&) const
    {
        return true;
    }
};

struct LeafPred
{
    UIPC_GENERIC bool operator()(const InfoStacklessBVH::LeafPredInfo& info) const
    {
        return ((info.i ^ info.j) & 1) == 0;
    }
};

struct CountingNodeCull
{
    int* calls;

    __device__ bool operator()(const InfoStacklessBVH::NodePredInfo&) const
    {
        atomicAdd(calls, 1);
        return false;
    }
};

struct CountingLeafPair
{
    int* leaf_calls;

    __device__ bool operator()(const InfoStacklessBVH::LeafPredInfo&) const
    {
        atomicAdd(leaf_calls, 1);
        return true;
    }
};

struct CullRateNodeCull
{
    int* calls;
    int* rejects;

    __device__ bool operator()(const InfoStacklessBVH::NodePredInfo& info) const
    {
        atomicAdd(calls, 1);
        bool keep = (info.query_id % 3) != 0;
        if(!keep)
            atomicAdd(rejects, 1);
        return keep;
    }
};

struct LeafPredTrue
{
    __device__ bool operator()(const InfoStacklessBVH::LeafPredInfo&) const
    {
        return true;
    }
};

struct BidOnlyNodeCull
{
    static constexpr IndexT invalid = static_cast<IndexT>(-1);

    cuda_tool::BufferView<IndexT> bids;
    cuda_tool::BufferView<IndexT> is_self_contact;
    IndexT                        self_contact_count;
    int*                          calls;
    int*                          rejects;
    int*                          invalid_cid_hits;

    __device__ bool operator()(const InfoStacklessBVH::NodePredInfo& info) const
    {
        atomicAdd(calls, 1);
        if(info.node_cid == invalid)
            atomicAdd(invalid_cid_hits, 1);
        auto qbid = bids(info.query_id);
        bool self_contact_disabled = (qbid != invalid) && (qbid < self_contact_count)
                                     && !is_self_contact(qbid);
        bool keep = !(info.node_bid != invalid && qbid != invalid
                      && info.node_bid == qbid && self_contact_disabled);
        if(!keep)
            atomicAdd(rejects, 1);
        return keep;
    }
};

struct CidOnlyNodeCull
{
    static constexpr IndexT invalid = static_cast<IndexT>(-1);

    cuda_tool::BufferView<IndexT> bids;
    cuda_tool::BufferView<IndexT> cids;
    cuda_tool::Dense2D<IndexT>    cmts;
    cuda_tool::BufferView<IndexT> is_self_contact;
    IndexT                        self_contact_count;
    int*                          rejects;

    __device__ bool operator()(const InfoStacklessBVH::NodePredInfo& info) const
    {
        auto qbid = bids(info.query_id);
        auto qcid = cids(info.query_id);
        bool bid_cull = info.node_bid != invalid && qbid != invalid && info.node_bid == qbid
                        && (qbid < self_contact_count) && !is_self_contact(qbid);
        bool cid_cull = info.node_cid != invalid && qcid != invalid
                        && !cmts(qcid, info.node_cid);
        bool keep = !(bid_cull || cid_cull);
        if(!keep)
            atomicAdd(rejects, 1);
        return keep;
    }
};

struct FallbackNodeCull
{
    static constexpr IndexT invalid = static_cast<IndexT>(-1);

    cuda_tool::BufferView<IndexT> bids;
    cuda_tool::BufferView<IndexT> cids;
    cuda_tool::Dense2D<IndexT>    cmts;
    cuda_tool::BufferView<IndexT> is_self_contact;
    IndexT                        self_contact_count;
    int*                          rejects;
    int*                          invalid_bid_hits;
    int*                          invalid_cid_hits;

    __device__ bool operator()(const InfoStacklessBVH::NodePredInfo& info) const
    {
        if(info.node_bid == invalid)
            atomicAdd(invalid_bid_hits, 1);
        if(info.node_cid == invalid)
            atomicAdd(invalid_cid_hits, 1);
        auto qbid = bids(info.query_id);
        auto qcid = cids(info.query_id);
        bool bid_cull = info.node_bid != invalid && qbid != invalid && info.node_bid == qbid
                        && (qbid < self_contact_count) && !is_self_contact(qbid);
        bool cid_cull = info.node_cid != invalid && qcid != invalid
                        && !cmts(qcid, info.node_cid);
        bool keep = !(bid_cull || cid_cull);
        if(!keep)
            atomicAdd(rejects, 1);
        return keep;
    }
};

struct CountingLeafPredFalse
{
    int* leaf_calls;

    __device__ bool operator()(InfoStacklessBVH::LeafPredInfo) const
    {
        atomicAdd(leaf_calls, 1);
        return false;
    }
};

void check_cp_conservative(span<Vector2i> test, span<Vector2i> gt)
{
    auto compare = [](const Vector2i& lhs, const Vector2i& rhs)
    { return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]); };

    std::ranges::sort(test, compare);
    std::ranges::sort(gt, compare);

    std::list<Vector2i> diff;
    std::set_difference(
        gt.begin(), gt.end(), test.begin(), test.end(), std::back_inserter(diff), compare);

    CHECK(diff.empty());
}

void check_cp_exact(std::vector<Vector2i> test, std::vector<Vector2i> gt)
{
    auto compare = [](const Vector2i& lhs, const Vector2i& rhs)
    { return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]); };

    std::ranges::sort(test, compare);
    std::ranges::sort(gt, compare);
    REQUIRE(test.size() == gt.size());
    for(size_t i = 0; i < gt.size(); ++i)
    {
        CAPTURE(i);
        CHECK(test[i][0] == gt[i][0]);
        CHECK(test[i][1] == gt[i][1]);
    }
}

bool allow_contact(span<const IndexT> cmts, IndexT cid_count, IndexT lhs, IndexT rhs)
{
    return cmts[lhs * cid_count + rhs] != 0;
}

std::vector<Vector2i> brute_force_detect(span<const AABB>   aabbs,
                                         span<const IndexT> bids,
                                         span<const IndexT> cids,
                                         span<const IndexT> cmts,
                                         IndexT             cid_count)
{
    std::vector<Vector2i> pairs;
    LeafPred              lp;

    for(IndexT i = 0; i < static_cast<IndexT>(aabbs.size()); ++i)
    {
        for(IndexT j = i + 1; j < static_cast<IndexT>(aabbs.size()); ++j)
        {
            if(!aabbs[i].intersects(aabbs[j]))
                continue;
            if(bids[i] == bids[j])
                continue;
            if(!allow_contact(cmts, cid_count, cids[i], cids[j]))
                continue;
            if(!lp(InfoStacklessBVH::LeafPredInfo{i, j, bids[i], cids[i], bids[j], cids[j]}))
                continue;

            pairs.emplace_back(i, j);
        }
    }

    return pairs;
}

std::vector<Vector2i> brute_force_query(span<const AABB>   query_aabbs,
                                        span<const IndexT> query_bids,
                                        span<const IndexT> query_cids,
                                        span<const AABB>   tree_aabbs,
                                        span<const IndexT> tree_bids,
                                        span<const IndexT> tree_cids,
                                        span<const IndexT> cmts,
                                        IndexT             cid_count)
{
    std::vector<Vector2i> pairs;
    LeafPred              lp;

    for(IndexT i = 0; i < static_cast<IndexT>(query_aabbs.size()); ++i)
    {
        for(IndexT j = 0; j < static_cast<IndexT>(tree_aabbs.size()); ++j)
        {
            if(!query_aabbs[i].intersects(tree_aabbs[j]))
                continue;
            if(query_bids[i] == tree_bids[j])
                continue;
            if(!allow_contact(cmts, cid_count, query_cids[i], tree_cids[j]))
                continue;
            if(!lp(InfoStacklessBVH::LeafPredInfo{
                   i, j, query_bids[i], query_cids[i], tree_bids[j], tree_cids[j]}))
                continue;

            pairs.emplace_back(i, j);
        }
    }

    return pairs;
}

void run_info_stackless_bvh_test(const SimplicialComplex& mesh)
{
    auto pos_view = mesh.positions().view();
    auto tri_view = mesh.triangles().topo().view();

    std::vector<AABB>   aabbs(tri_view.size());
    std::vector<IndexT> bids(tri_view.size());
    std::vector<IndexT> cids(tri_view.size());

    for(auto&& [i, tri] : enumerate(tri_view))
    {
        auto p0 = pos_view[tri[0]];
        auto p1 = pos_view[tri[1]];
        auto p2 = pos_view[tri[2]];
        aabbs[i].extend(p0.cast<float>()).extend(p1.cast<float>()).extend(p2.cast<float>());

        bids[i] = static_cast<IndexT>(i % 7);
        cids[i] = static_cast<IndexT>(i % 4);
    }

    constexpr IndexT    cid_count = 4;
    std::vector<IndexT> cmts(cid_count * cid_count, 1);
    for(IndexT i = 0; i < cid_count; ++i)
    {
        for(IndexT j = 0; j < cid_count; ++j)
        {
            cmts[i * cid_count + j] = (((i + j) % 3) != 0) ? 1 : 0;
        }
    }

    DeviceBuffer<AABB> d_aabbs(aabbs.size());
    d_aabbs.view().copy_from(aabbs.data());
    DeviceBuffer<IndexT> d_bids(bids.size());
    d_bids.view().copy_from(bids.data());
    DeviceBuffer<IndexT> d_cids(cids.size());
    d_cids.view().copy_from(cids.data());
    DeviceBuffer2D<IndexT> d_cmts(Extent2D{static_cast<size_t>(cid_count),
                                           static_cast<size_t>(cid_count)});
    d_cmts.view().copy_from(cmts.data());

    InfoStacklessBVH bvh;
    bvh.build(d_aabbs, d_bids, d_cids);

    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.reserve(1024);

    {
        Timer timer("info_stackless_bvh detect");
        bvh.detect(d_cmts.view(), NodePred{}, LeafPred{}, qbuffer);
    }

    std::vector<Vector2i> detect_pairs(qbuffer.size());
    qbuffer.view().copy_to(detect_pairs.data());
    auto detect_gt = brute_force_detect(aabbs, bids, cids, cmts, cid_count);
    check_cp_conservative(detect_pairs, detect_gt);

    // Exercise the launch/finalize split and its overflow retry with a queue
    // deliberately smaller than the expected result.
    InfoStacklessBVH::QueryBuffer batched_qbuffer;
    batched_qbuffer.m_pairs.release();
    batched_qbuffer.reserve(1);
    bvh.launch_detect(d_cmts.view(), NodePred{}, LeafPred{}, batched_qbuffer);
    int  batched_count = batched_qbuffer.m_cpNum;
    bool retry = bvh.prepare_query_result(batched_qbuffer, batched_count);
    if(retry)
        bvh.launch_detect(d_cmts.view(), NodePred{}, LeafPred{}, batched_qbuffer);

    std::vector<Vector2i> batched_pairs(batched_qbuffer.size());
    batched_qbuffer.view().copy_to(batched_pairs.data());
    check_cp_conservative(batched_pairs, detect_gt);
    CHECK(retry == (batched_count > 1));

    std::vector<AABB>   query_aabbs = aabbs;
    std::vector<IndexT> query_bids  = bids;
    std::vector<IndexT> query_cids  = cids;

    for(IndexT i = 0; i < static_cast<IndexT>(query_bids.size()); ++i)
    {
        query_bids[i] = (query_bids[i] + 3) % 11;
    }

    DeviceBuffer<AABB> d_query_aabbs(query_aabbs.size());
    d_query_aabbs.view().copy_from(query_aabbs.data());
    DeviceBuffer<IndexT> d_query_bids(query_bids.size());
    d_query_bids.view().copy_from(query_bids.data());
    DeviceBuffer<IndexT> d_query_cids(query_cids.size());
    d_query_cids.view().copy_from(query_cids.data());

    {
        Timer timer("info_stackless_bvh query");
        bvh.query(d_query_aabbs.view(),
                  d_query_bids.view(),
                  d_query_cids.view(),
                  d_cmts.view(),
                  NodePred{},
                  LeafPred{},
                  qbuffer);
    }

    std::vector<Vector2i> query_pairs(qbuffer.size());
    qbuffer.view().copy_to(query_pairs.data());
    auto query_gt = brute_force_query(
        query_aabbs, query_bids, query_cids, aabbs, bids, cids, cmts, cid_count);
    check_cp_conservative(query_pairs, query_gt);
}

void run_internal_cull_proof_case()
{
    constexpr IndexT    n = 64;
    std::vector<AABB>   aabbs(n);
    std::vector<IndexT> bids(n, 3);
    std::vector<IndexT> cids(n, 0);
    for(IndexT i = 0; i < n; ++i)
    {
        double  x  = static_cast<double>(i) * 1.0e-3;
        Vector3 p0 = Vector3{x, x, x};
        Vector3 p1 = Vector3{x + 1.0, x + 1.0, x + 1.0};
        AABB    box;
        box.extend(p0.cast<float>()).extend(p1.cast<float>());
        aabbs[i] = box;
    }

    DeviceBuffer<AABB> d_aabbs(aabbs.size());
    d_aabbs.view().copy_from(aabbs.data());
    DeviceBuffer<IndexT> d_bids(bids.size());
    d_bids.view().copy_from(bids.data());
    DeviceBuffer<IndexT> d_cids(cids.size());
    d_cids.view().copy_from(cids.data());
    InfoStacklessBVH::Impl impl;
    impl.build(d_aabbs.view(), d_bids.view(), d_cids.view());

    DeviceVar<int>         cp_num;
    DeviceBuffer<int>      node_cull_calls(1);
    DeviceBuffer<int>      leaf_pair_calls(1);
    DeviceBuffer<Vector2i> pairs(16);
    BufferLaunch().fill(cp_num.view(), 0);
    BufferLaunch().fill(node_cull_calls.view(), 0);
    BufferLaunch().fill(leaf_pair_calls.view(), 0);

    impl.stacklessSelf(CountingNodeCull{node_cull_calls.data()},
                       CountingLeafPair{leaf_pair_calls.data()},
                       cp_num.view(),
                       pairs.view());

    int h_node_cull_calls = 0;
    int h_leaf_pair_calls = 0;
    node_cull_calls.view(0, 1).copy_to(&h_node_cull_calls);
    leaf_pair_calls.view(0, 1).copy_to(&h_leaf_pair_calls);
    int h_pairs = cp_num;

    fmt::println("internal-cull proof stats: node_cull_calls={}, leaf_pair_calls={}, pairs={}",
                 h_node_cull_calls,
                 h_leaf_pair_calls,
                 h_pairs);

    // The final Morton-rank query skips the root because no greater-ranked
    // leaf can exist below it. All other queries reach and reject the root.
    CHECK(h_node_cull_calls == n - 1);
    CHECK(h_leaf_pair_calls == 0);
    CHECK(h_pairs == 0);
}

void run_self_rank_reordered_node_case()
{
    // Raw ids 0,1,2,3 have Morton order 1,3,2,0. Distinct centers avoid any
    // dependence on radix-sort tie order; the common extent overlaps all pairs.
    constexpr std::array<float, 4> x_by_raw_id = {3.0f, 0.0f, 2.0f, 1.0f};
    std::vector<AABB>              aabbs;
    for(float x : x_by_raw_id)
    {
        AABB box;
        box.extend(Vector3{x, 0.0, 0.0}.cast<float>());
        box.extend(Vector3{x + 4.0, 4.0, 4.0}.cast<float>());
        aabbs.push_back(box);
    }

    std::vector<IndexT>  bids(aabbs.size(), 0);
    std::vector<IndexT>  cids(aabbs.size(), 0);
    DeviceBuffer<AABB>   d_aabbs(aabbs.size());
    DeviceBuffer<IndexT> d_bids(bids.size());
    DeviceBuffer<IndexT> d_cids(cids.size());
    d_aabbs.view().copy_from(aabbs.data());
    d_bids.view().copy_from(bids.data());
    d_cids.view().copy_from(cids.data());

    DeviceBuffer2D<IndexT> d_cmts(Extent2D{1, 1});
    IndexT                  allow = 1;
    d_cmts.view().copy_from(&allow);

    InfoStacklessBVH bvh;
    bvh.build(d_aabbs.view(), d_bids.view(), d_cids.view());
    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.reserve(8);
    bvh.detect(d_cmts.view(), NodePred{}, LeafPredTrue{}, qbuffer);

    std::vector<Vector2i> actual(qbuffer.size());
    qbuffer.view().copy_to(actual.data());
    std::vector<Vector2i> expected;
    for(IndexT i = 0; i < static_cast<IndexT>(aabbs.size()); ++i)
        for(IndexT j = i + 1; j < static_cast<IndexT>(aabbs.size()); ++j)
            expected.emplace_back(i, j);
    check_cp_exact(std::move(actual), std::move(expected));
}

std::vector<AABB> make_all_overlapping_aabbs(size_t count)
{
    std::vector<AABB> aabbs(count);
    for(size_t i = 0; i < count; ++i)
    {
        const float shift = static_cast<float>(i) * 1.0e-4f;
        aabbs[i].extend(Vector3{shift, shift, shift}.cast<float>());
        aabbs[i].extend(Vector3{1.0 + shift, 1.0 + shift, 1.0 + shift}.cast<float>());
    }
    return aabbs;
}

std::vector<Vector2i> run_self_all_pairs(const std::vector<AABB>& aabbs,
                                         bool&                    retried)
{
    std::vector<IndexT>  bids(aabbs.size(), 0);
    std::vector<IndexT>  cids(aabbs.size(), 0);
    DeviceBuffer<AABB>   d_aabbs(aabbs.size());
    DeviceBuffer<IndexT> d_bids(bids.size());
    DeviceBuffer<IndexT> d_cids(cids.size());
    if(!aabbs.empty())
    {
        d_aabbs.view().copy_from(aabbs.data());
        d_bids.view().copy_from(bids.data());
        d_cids.view().copy_from(cids.data());
    }

    DeviceBuffer2D<IndexT> d_cmts(Extent2D{1, 1});
    IndexT                  allow = 1;
    d_cmts.view().copy_from(&allow);

    InfoStacklessBVH bvh;
    bvh.build(d_aabbs.view(), d_bids.view(), d_cids.view());
    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.m_pairs.release();
    qbuffer.reserve(1);
    bvh.launch_detect(d_cmts.view(), NodePred{}, LeafPredTrue{}, qbuffer);
    int count = qbuffer.m_cpNum;
    retried   = bvh.prepare_query_result(qbuffer, count);
    if(retried)
        bvh.launch_detect(d_cmts.view(), NodePred{}, LeafPredTrue{}, qbuffer);

    std::vector<Vector2i> pairs(qbuffer.size());
    if(!pairs.empty())
        qbuffer.view().copy_to(pairs.data());
    return pairs;
}

std::vector<Vector2i> run_other_all_pairs(const std::vector<AABB>& query_aabbs,
                                          const std::vector<AABB>& tree_aabbs,
                                          bool&                    retried)
{
    std::vector<IndexT> query_bids(query_aabbs.size(), 0);
    std::vector<IndexT> query_cids(query_aabbs.size(), 0);
    std::vector<IndexT> tree_bids(tree_aabbs.size(), 0);
    std::vector<IndexT> tree_cids(tree_aabbs.size(), 0);

    DeviceBuffer<AABB>   d_query_aabbs(query_aabbs.size());
    DeviceBuffer<IndexT> d_query_bids(query_bids.size());
    DeviceBuffer<IndexT> d_query_cids(query_cids.size());
    if(!query_aabbs.empty())
    {
        d_query_aabbs.view().copy_from(query_aabbs.data());
        d_query_bids.view().copy_from(query_bids.data());
        d_query_cids.view().copy_from(query_cids.data());
    }

    DeviceBuffer<AABB>   d_tree_aabbs(tree_aabbs.size());
    DeviceBuffer<IndexT> d_tree_bids(tree_bids.size());
    DeviceBuffer<IndexT> d_tree_cids(tree_cids.size());
    d_tree_aabbs.view().copy_from(tree_aabbs.data());
    d_tree_bids.view().copy_from(tree_bids.data());
    d_tree_cids.view().copy_from(tree_cids.data());

    DeviceBuffer2D<IndexT> d_cmts(Extent2D{1, 1});
    IndexT                  allow = 1;
    d_cmts.view().copy_from(&allow);

    InfoStacklessBVH bvh;
    bvh.build(d_tree_aabbs.view(), d_tree_bids.view(), d_tree_cids.view());
    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.m_pairs.release();
    qbuffer.reserve(1);
    bvh.launch_query(d_query_aabbs.view(),
                     d_query_bids.view(),
                     d_query_cids.view(),
                     d_cmts.view(),
                     NodePred{},
                     LeafPredTrue{},
                     qbuffer,
                     true);
    int count = qbuffer.m_cpNum;
    retried   = bvh.prepare_query_result(qbuffer, count);
    if(retried)
        bvh.launch_query(d_query_aabbs.view(),
                         d_query_bids.view(),
                         d_query_cids.view(),
                         d_cmts.view(),
                         NodePred{},
                         LeafPredTrue{},
                         qbuffer,
                         false);

    std::vector<Vector2i> pairs(qbuffer.size());
    if(!pairs.empty())
        qbuffer.view().copy_to(pairs.data());
    return pairs;
}

void run_cta_queue_boundary_cases()
{
    constexpr std::array<size_t, 14> counts = {
        0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257};

    for(size_t count : counts)
    {
        DYNAMIC_SECTION("Self N=" << count)
        {
            auto aabbs = make_all_overlapping_aabbs(count);
            std::vector<Vector2i> expected;
            for(IndexT i = 0; i < static_cast<IndexT>(count); ++i)
                for(IndexT j = i + 1; j < static_cast<IndexT>(count); ++j)
                    expected.emplace_back(i, j);
            bool retried = false;
            auto actual  = run_self_all_pairs(aabbs, retried);
            CHECK(retried == (expected.size() > 1));
            check_cp_exact(std::move(actual), std::move(expected));
        }
    }

    const auto tree_aabbs = make_all_overlapping_aabbs(65);
    for(size_t count : counts)
    {
        DYNAMIC_SECTION("Other N=" << count)
        {
            auto query_aabbs = make_all_overlapping_aabbs(count);
            std::vector<Vector2i> expected;
            for(IndexT i = 0; i < static_cast<IndexT>(count); ++i)
                for(IndexT j = 0; j < static_cast<IndexT>(tree_aabbs.size()); ++j)
                    expected.emplace_back(i, j);
            bool retried = false;
            auto actual = run_other_all_pairs(query_aabbs, tree_aabbs, retried);
            CHECK(retried == (expected.size() > 1));
            check_cp_exact(std::move(actual), std::move(expected));
        }
    }
}

struct RefitNodePred
{
    cuda_tool::CDense2D<IndexT> cmts;

    UIPC_GENERIC bool operator()(const InfoStacklessBVH::NodePredInfo& info) const
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        bool bid_cull = info.query_bid != invalid && info.node_bid != invalid
                        && info.query_bid == info.node_bid;
        bool cid_cull = info.query_cid != invalid && info.node_cid != invalid
                        && !cmts(info.query_cid, info.node_cid);
        return !(bid_cull || cid_cull);
    }
};

struct RefitLeafPred
{
    cuda_tool::CDense2D<IndexT> cmts;

    UIPC_GENERIC bool operator()(const InfoStacklessBVH::LeafPredInfo& info) const
    {
        return info.bid_i != info.bid_j && cmts(info.cid_i, info.cid_j);
    }
};

struct RefitInputs
{
    std::vector<AABB>   aabbs;
    std::vector<IndexT> bids;
    std::vector<IndexT> cids;
};

AABB refit_box_at(double x, double radius)
{
    AABB box;
    box.extend(Vector3{x - radius, -radius, -radius}.cast<float>());
    box.extend(Vector3{x + radius, radius, radius}.cast<float>());
    return box;
}

RefitInputs refit_initial_inputs(IndexT count)
{
    RefitInputs result;
    result.aabbs.reserve(count);
    result.bids.reserve(count);
    result.cids.reserve(count);
    for(IndexT i = 0; i < count; ++i)
    {
        result.aabbs.push_back(refit_box_at(3.0 * i, 0.25));
        result.bids.push_back(i % 7);
        result.cids.push_back(i % 4);
    }
    return result;
}

RefitInputs refit_swept_inputs(IndexT count, double phase)
{
    RefitInputs result;
    result.aabbs.reserve(count);
    result.bids.reserve(count);
    result.cids.reserve(count);
    for(IndexT i = 0; i < count; ++i)
    {
        // Reverse spatial order and overlap many leaves so a stale or
        // incompletely published parent becomes visible in the exact pairs.
        double x = 0.015 * (count - 1 - i) + phase * ((i % 3) - 1);
        result.aabbs.push_back(refit_box_at(x, 0.65));
        result.bids.push_back((i + 3) % 9);
        result.cids.push_back((i + 1) % 4);
    }
    return result;
}

enum class RefitMetadataPhase
{
    Homogeneous,
    DifferentHomogeneous,
    Mixed
};

RefitInputs refit_grouped_inputs(IndexT             count,
                                 RefitMetadataPhase phase,
                                 bool               reverse_groups)
{
    constexpr IndexT group_size = 3;
    REQUIRE(count % group_size == 0);

    RefitInputs result;
    result.aabbs.reserve(count);
    result.bids.reserve(count);
    result.cids.reserve(count);
    IndexT group_count = count / group_size;
    for(IndexT i = 0; i < count; ++i)
    {
        IndexT group         = i / group_size;
        IndexT local         = i % group_size;
        IndexT spatial_group = reverse_groups ? group_count - 1 - group : group;
        result.aabbs.push_back(refit_box_at(10.0 * spatial_group + 0.15 * local, 0.2));

        switch(phase)
        {
            case RefitMetadataPhase::Homogeneous:
                result.bids.push_back(17);
                result.cids.push_back(1);
                break;
            case RefitMetadataPhase::DifferentHomogeneous:
                result.bids.push_back(29);
                result.cids.push_back(2);
                break;
            case RefitMetadataPhase::Mixed:
                result.bids.push_back(i);
                result.cids.push_back(local);
                break;
        }
    }
    return result;
}

RefitInputs refit_grouped_queries(IndexT group_count, IndexT bid, IndexT cid)
{
    RefitInputs result;
    result.aabbs.reserve(group_count);
    result.bids.reserve(group_count);
    result.cids.reserve(group_count);
    for(IndexT group = 0; group < group_count; ++group)
    {
        result.aabbs.push_back(refit_box_at(10.0 * group + 0.15, 0.4));
        result.bids.push_back(bid);
        result.cids.push_back(cid);
    }
    return result;
}

void upload_refit(const RefitInputs&       input,
                  DeviceBuffer<AABB>&      aabbs,
                  DeviceBuffer<IndexT>&    bids,
                  DeviceBuffer<IndexT>&    cids)
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

std::vector<Vector2i> refit_detect_pairs(InfoStacklessBVH&          bvh,
                                         CBuffer2DView<IndexT>       cmts)
{
    InfoStacklessBVH::QueryBuffer pairs;
    pairs.m_pairs.release();
    pairs.reserve(1);
    bvh.detect(cmts, RefitNodePred{cmts.viewer()}, RefitLeafPred{cmts.viewer()}, pairs);

    std::vector<Vector2i> result(pairs.size());
    if(!result.empty())
        pairs.view().copy_to(result.data());
    return result;
}

std::vector<Vector2i> refit_query_pairs(InfoStacklessBVH&       bvh,
                                        CBufferView<AABB>       query_aabbs,
                                        CBufferView<IndexT>     query_bids,
                                        CBufferView<IndexT>     query_cids,
                                        CBuffer2DView<IndexT>   cmts)
{
    InfoStacklessBVH::QueryBuffer pairs;
    pairs.m_pairs.release();
    pairs.reserve(1);
    bvh.query(query_aabbs,
              query_bids,
              query_cids,
              cmts,
              RefitNodePred{cmts.viewer()},
              RefitLeafPred{cmts.viewer()},
              pairs);

    std::vector<Vector2i> result(pairs.size());
    if(!result.empty())
        pairs.view().copy_to(result.data());
    return result;
}

size_t check_refit_pairs(std::vector<Vector2i> refitted,
                         std::vector<Vector2i> rebuilt)
{
    size_t count = refitted.size();
    check_cp_exact(std::move(refitted), std::move(rebuilt));
    return count;
}

void compare_full_build_and_refit(const RefitInputs& initial,
                                  const RefitInputs& swept,
                                  CBuffer2DView<IndexT> cmts)
{
    DeviceBuffer<AABB>   refit_aabbs;
    DeviceBuffer<IndexT> refit_bids;
    DeviceBuffer<IndexT> refit_cids;
    upload_refit(initial, refit_aabbs, refit_bids, refit_cids);

    InfoStacklessBVH refitted;
    refitted.build(refit_aabbs, refit_bids, refit_cids);
    upload_refit(swept, refit_aabbs, refit_bids, refit_cids);
    REQUIRE(refitted.refit(refit_aabbs, refit_bids, refit_cids));

    DeviceBuffer<AABB>   rebuilt_aabbs;
    DeviceBuffer<IndexT> rebuilt_bids;
    DeviceBuffer<IndexT> rebuilt_cids;
    upload_refit(swept, rebuilt_aabbs, rebuilt_bids, rebuilt_cids);
    InfoStacklessBVH rebuilt;
    rebuilt.build(rebuilt_aabbs, rebuilt_bids, rebuilt_cids);

    check_refit_pairs(refit_detect_pairs(refitted, cmts),
                      refit_detect_pairs(rebuilt, cmts));

    auto query = refit_swept_inputs(11, 0.025);
    DeviceBuffer<AABB>   query_aabbs;
    DeviceBuffer<IndexT> query_bids;
    DeviceBuffer<IndexT> query_cids;
    upload_refit(query, query_aabbs, query_bids, query_cids);
    check_refit_pairs(refit_query_pairs(refitted,
                                        query_aabbs,
                                        query_bids,
                                        query_cids,
                                        cmts),
                      refit_query_pairs(rebuilt,
                                        query_aabbs,
                                        query_bids,
                                        query_cids,
                                        cmts));
}

void run_refit_cases()
{
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
        CHECK(refit_detect_pairs(bvh, d_cmts.view()).empty());
    }

    SECTION("single primitive")
    {
        compare_full_build_and_refit(
            refit_initial_inputs(1), refit_swept_inputs(1, 0.0), d_cmts.view());
    }

    SECTION("swept topology and metadata")
    {
        compare_full_build_and_refit(
            refit_initial_inputs(33), refit_swept_inputs(33, 0.0), d_cmts.view());
    }

    SECTION("repeated refit")
    {
        auto initial = refit_initial_inputs(33);
        auto swept0  = refit_swept_inputs(33, 0.0);
        auto swept1  = refit_swept_inputs(33, 0.04);

        DeviceBuffer<AABB>   aabbs;
        DeviceBuffer<IndexT> bids;
        DeviceBuffer<IndexT> cids;
        upload_refit(initial, aabbs, bids, cids);
        InfoStacklessBVH refitted;
        refitted.build(aabbs, bids, cids);
        upload_refit(swept0, aabbs, bids, cids);
        REQUIRE(refitted.refit(aabbs, bids, cids));
        upload_refit(swept1, aabbs, bids, cids);
        REQUIRE(refitted.refit(aabbs, bids, cids));

        DeviceBuffer<AABB>   rebuilt_aabbs;
        DeviceBuffer<IndexT> rebuilt_bids;
        DeviceBuffer<IndexT> rebuilt_cids;
        upload_refit(swept1, rebuilt_aabbs, rebuilt_bids, rebuilt_cids);
        InfoStacklessBVH rebuilt;
        rebuilt.build(rebuilt_aabbs, rebuilt_bids, rebuilt_cids);
        check_refit_pairs(refit_detect_pairs(refitted, d_cmts.view()),
                          refit_detect_pairs(rebuilt, d_cmts.view()));
    }

    SECTION("count change fails closed")
    {
        auto initial = refit_initial_inputs(8);
        DeviceBuffer<AABB>   aabbs;
        DeviceBuffer<IndexT> bids;
        DeviceBuffer<IndexT> cids;
        upload_refit(initial, aabbs, bids, cids);
        InfoStacklessBVH bvh;
        bvh.build(aabbs, bids, cids);
        auto changed = refit_swept_inputs(9, 0.0);
        upload_refit(changed, aabbs, bids, cids);
        CHECK_FALSE(bvh.refit(aabbs, bids, cids));
    }

    SECTION("cross CTA metadata transitions and overflow")
    {
        constexpr IndexT primitive_count = 1536;
        constexpr IndexT group_size      = 3;
        static_assert(primitive_count > 1024);

        DeviceBuffer<AABB>   refit_aabbs;
        DeviceBuffer<IndexT> refit_bids;
        DeviceBuffer<IndexT> refit_cids;
        upload_refit(refit_grouped_inputs(
                         primitive_count, RefitMetadataPhase::Homogeneous, false),
                     refit_aabbs,
                     refit_bids,
                     refit_cids);
        InfoStacklessBVH refitted;
        refitted.build(refit_aabbs, refit_bids, refit_cids);

        auto compare_refit_with_rebuild = [&](const RefitInputs& state,
                                              const RefitInputs& query)
        {
            upload_refit(state, refit_aabbs, refit_bids, refit_cids);
            REQUIRE(refitted.refit(refit_aabbs, refit_bids, refit_cids));

            DeviceBuffer<AABB>   rebuilt_aabbs;
            DeviceBuffer<IndexT> rebuilt_bids;
            DeviceBuffer<IndexT> rebuilt_cids;
            upload_refit(state, rebuilt_aabbs, rebuilt_bids, rebuilt_cids);
            InfoStacklessBVH rebuilt;
            rebuilt.build(rebuilt_aabbs, rebuilt_bids, rebuilt_cids);

            size_t self_count = check_refit_pairs(
                refit_detect_pairs(refitted, d_cmts.view()),
                refit_detect_pairs(rebuilt, d_cmts.view()));

            DeviceBuffer<AABB>   query_aabbs;
            DeviceBuffer<IndexT> query_bids;
            DeviceBuffer<IndexT> query_cids;
            upload_refit(query, query_aabbs, query_bids, query_cids);
            size_t other_count = check_refit_pairs(
                refit_query_pairs(refitted,
                                  query_aabbs,
                                  query_bids,
                                  query_cids,
                                  d_cmts.view()),
                refit_query_pairs(rebuilt,
                                  query_aabbs,
                                  query_bids,
                                  query_cids,
                                  d_cmts.view()));
            return std::pair{self_count, other_count};
        };

        auto different = refit_grouped_inputs(
            primitive_count, RefitMetadataPhase::DifferentHomogeneous, true);
        auto query_old_homogeneous =
            refit_grouped_queries(primitive_count / group_size, 17, 2);
        auto [different_self, different_other] =
            compare_refit_with_rebuild(different, query_old_homogeneous);
        CHECK(different_self == 0);
        CHECK(different_other > 1);

        auto mixed = refit_grouped_inputs(
            primitive_count, RefitMetadataPhase::Mixed, false);
        auto query_old_different =
            refit_grouped_queries(primitive_count / group_size, 29, 1);
        auto [mixed_self, mixed_other] =
            compare_refit_with_rebuild(mixed, query_old_different);
        CHECK(mixed_self > 1);
        CHECK(mixed_other > 1);
    }
}

void run_internal_cull_rate_case()
{
    constexpr IndexT    n = 96;
    std::vector<AABB>   aabbs(n);
    std::vector<IndexT> bids(n);
    std::vector<IndexT> cids(n);
    for(IndexT i = 0; i < n; ++i)
    {
        double  x  = static_cast<double>(i % 16) * 5.0e-3;
        double  y  = static_cast<double>(i / 16) * 5.0e-3;
        Vector3 p0 = Vector3{x, y, 0.0};
        Vector3 p1 = Vector3{x + 1.0, y + 1.0, 1.0};
        aabbs[i].extend(p0.cast<float>()).extend(p1.cast<float>());
        bids[i] = static_cast<IndexT>(i % 9);
        cids[i] = static_cast<IndexT>(i % 4);
    }

    DeviceBuffer<AABB> d_aabbs(aabbs.size());
    d_aabbs.view().copy_from(aabbs.data());
    DeviceBuffer<IndexT> d_bids(bids.size());
    d_bids.view().copy_from(bids.data());
    DeviceBuffer<IndexT> d_cids(cids.size());
    d_cids.view().copy_from(cids.data());
    InfoStacklessBVH::Impl impl;
    impl.build(d_aabbs.view(), d_bids.view(), d_cids.view());

    DeviceVar<int>         cp_num;
    DeviceBuffer<int>      node_cull_calls(1);
    DeviceBuffer<int>      node_cull_rejects(1);
    DeviceBuffer<Vector2i> pairs(2048);
    BufferLaunch().fill(cp_num.view(), 0);
    BufferLaunch().fill(node_cull_calls.view(), 0);
    BufferLaunch().fill(node_cull_rejects.view(), 0);

    impl.stacklessSelf(CullRateNodeCull{node_cull_calls.data(),
                                        node_cull_rejects.data()},
                       LeafPredTrue{},
                       cp_num.view(),
                       pairs.view());

    int h_node_cull_calls   = 0;
    int h_node_cull_rejects = 0;
    node_cull_calls.view(0, 1).copy_to(&h_node_cull_calls);
    node_cull_rejects.view(0, 1).copy_to(&h_node_cull_rejects);
    double node_cull_rate = (h_node_cull_calls > 0) ?
                                static_cast<double>(h_node_cull_rejects)
                                    / static_cast<double>(h_node_cull_calls) :
                                0.0;
    fmt::println("internal-cull rate stats: node_cull_calls={}, node_cull_rejects={}, node_cull_rate={:.4f}",
                 h_node_cull_calls,
                 h_node_cull_rejects,
                 node_cull_rate);

    CHECK(h_node_cull_calls > 0);
    CHECK(h_node_cull_rejects > 0);
    CHECK(node_cull_rate > 0.0);
    CHECK(node_cull_rate < 1.0);
}

void run_two_leaf_nodepred_cases()
{
    constexpr IndexT invalid = static_cast<IndexT>(-1);

    auto make_overlapping_aabbs = []()
    {
        std::vector<AABB> aabbs(2);
        Vector3           p00 = Vector3{0.0, 0.0, 0.0};
        Vector3           p01 = Vector3{1.0, 1.0, 1.0};
        Vector3           p10 = Vector3{0.25, 0.25, 0.25};
        Vector3           p11 = Vector3{1.25, 1.25, 1.25};
        aabbs[0].extend(p00.cast<float>()).extend(p01.cast<float>());
        aabbs[1].extend(p10.cast<float>()).extend(p11.cast<float>());
        return aabbs;
    };

    SECTION("two_leaf_bid_only_node_cull")
    {
        // Scenario: both leaves belong to body 0, and self-contact is disabled for body 0.
        // Their leaf CIDs are valid but different, so the internal node CID should be invalid (-1).
        // Expected: NodePred rejects the subtree using BID logic before any leaf-pair output.
        auto                aabbs           = make_overlapping_aabbs();
        std::vector<IndexT> bids            = {0, 0};
        std::vector<IndexT> cids            = {0, 1};
        std::vector<IndexT> is_self_contact = {0};
        std::vector<IndexT> cmts            = {1, 1, 1, 1};
        IndexT self_contact_count = static_cast<IndexT>(is_self_contact.size());
        DeviceBuffer<AABB> d_aabbs(aabbs.size());
        d_aabbs.view().copy_from(aabbs.data());
        DeviceBuffer<IndexT> d_bids(bids.size());
        d_bids.view().copy_from(bids.data());
        DeviceBuffer<IndexT> d_cids(cids.size());
        d_cids.view().copy_from(cids.data());
        DeviceBuffer<IndexT> d_is_self_contact(is_self_contact.size());
        d_is_self_contact.view().copy_from(is_self_contact.data());
        DeviceBuffer2D<IndexT> d_cmts(Extent2D{2, 2});
        d_cmts.view().copy_from(cmts.data());
        DeviceBuffer<int> node_calls(1);
        DeviceBuffer<int> node_rejects(1);
        DeviceBuffer<int> node_invalid_cid_hits(1);
        BufferLaunch().fill(node_calls.view(), 0);
        BufferLaunch().fill(node_rejects.view(), 0);
        BufferLaunch().fill(node_invalid_cid_hits.view(), 0);

        InfoStacklessBVH bvh;
        bvh.build(d_aabbs.view(), d_bids.view(), d_cids.view());
        InfoStacklessBVH::QueryBuffer qbuffer;
        qbuffer.reserve(8);
        bvh.detect(d_cmts.view(),
                   BidOnlyNodeCull{d_bids.viewer(),
                                   d_is_self_contact.viewer(),
                                   self_contact_count,
                                   node_calls.data(),
                                   node_rejects.data(),
                                   node_invalid_cid_hits.data()},
                   LeafPredTrue{},
                   qbuffer);

        int h_calls            = 0;
        int h_rejects          = 0;
        int h_invalid_cid_hits = 0;
        node_calls.view(0, 1).copy_to(&h_calls);
        node_rejects.view(0, 1).copy_to(&h_rejects);
        node_invalid_cid_hits.view(0, 1).copy_to(&h_invalid_cid_hits);
        CHECK(qbuffer.size() == 0);
        CHECK(h_calls > 0);
        CHECK(h_rejects > 0);
        CHECK(h_invalid_cid_hits > 0);
    }

    SECTION("two_leaf_cid_only_node_cull")
    {
        // Scenario: leaves have different bodies (0 and 1), so BID does not cull.
        // Their CIDs are both 1 and cmts(1,1)=0, so NodePred must cull by CID rule.
        auto                aabbs           = make_overlapping_aabbs();
        std::vector<IndexT> bids            = {0, 1};
        std::vector<IndexT> cids            = {1, 1};
        std::vector<IndexT> is_self_contact = {1, 1};
        std::vector<IndexT> cmts            = {1, 1, 1, 0};
        IndexT self_contact_count = static_cast<IndexT>(is_self_contact.size());
        DeviceBuffer<AABB> d_aabbs(aabbs.size());
        d_aabbs.view().copy_from(aabbs.data());
        DeviceBuffer<IndexT> d_bids(bids.size());
        d_bids.view().copy_from(bids.data());
        DeviceBuffer<IndexT> d_cids(cids.size());
        d_cids.view().copy_from(cids.data());
        DeviceBuffer<IndexT> d_is_self_contact(is_self_contact.size());
        d_is_self_contact.view().copy_from(is_self_contact.data());
        DeviceBuffer2D<IndexT> d_cmts(Extent2D{2, 2});
        d_cmts.view().copy_from(cmts.data());
        DeviceBuffer<int> node_rejects(1);
        BufferLaunch().fill(node_rejects.view(), 0);

        InfoStacklessBVH bvh;
        bvh.build(d_aabbs.view(), d_bids.view(), d_cids.view());
        InfoStacklessBVH::QueryBuffer qbuffer;
        qbuffer.reserve(8);
        bvh.detect(d_cmts.view(),
                   CidOnlyNodeCull{d_bids.viewer(),
                                   d_cids.viewer(),
                                   d_cmts.viewer(),
                                   d_is_self_contact.viewer(),
                                   self_contact_count,
                                   node_rejects.data()},
                   LeafPredTrue{},
                   qbuffer);

        int h_rejects = 0;
        node_rejects.view(0, 1).copy_to(&h_rejects);
        CHECK(qbuffer.size() == 0);
        CHECK(h_rejects > 0);
    }

    SECTION("two_leaf_both_invalid_leaf_fallback")
    {
        // Scenario: leaf BIDs/CIDs are legal values but different between the two leaves.
        // This forces internal node BID/CID to become invalid by the merge rule.
        // Expected: NodePred cannot cull, traversal reaches LeafPred fallback.
        auto                aabbs           = make_overlapping_aabbs();
        std::vector<IndexT> bids            = {0, 1};
        std::vector<IndexT> cids            = {0, 1};
        std::vector<IndexT> is_self_contact = {1, 1};
        std::vector<IndexT> cmts            = {1, 1, 1, 1};
        IndexT self_contact_count = static_cast<IndexT>(is_self_contact.size());
        DeviceBuffer<AABB> d_aabbs(aabbs.size());
        d_aabbs.view().copy_from(aabbs.data());
        DeviceBuffer<IndexT> d_bids(bids.size());
        d_bids.view().copy_from(bids.data());
        DeviceBuffer<IndexT> d_cids(cids.size());
        d_cids.view().copy_from(cids.data());
        DeviceBuffer<IndexT> d_is_self_contact(is_self_contact.size());
        d_is_self_contact.view().copy_from(is_self_contact.data());
        DeviceBuffer2D<IndexT> d_cmts(Extent2D{2, 2});
        d_cmts.view().copy_from(cmts.data());
        DeviceBuffer<int> node_rejects(1);
        DeviceBuffer<int> leaf_calls(1);
        DeviceBuffer<int> node_invalid_bid_hits(1);
        DeviceBuffer<int> node_invalid_cid_hits(1);
        BufferLaunch().fill(node_rejects.view(), 0);
        BufferLaunch().fill(leaf_calls.view(), 0);
        BufferLaunch().fill(node_invalid_bid_hits.view(), 0);
        BufferLaunch().fill(node_invalid_cid_hits.view(), 0);

        InfoStacklessBVH bvh;
        bvh.build(d_aabbs.view(), d_bids.view(), d_cids.view());
        InfoStacklessBVH::QueryBuffer qbuffer;
        qbuffer.reserve(8);
        bvh.detect(d_cmts.view(),
                   FallbackNodeCull{d_bids.viewer(),
                                    d_cids.viewer(),
                                    d_cmts.viewer(),
                                    d_is_self_contact.viewer(),
                                    self_contact_count,
                                    node_rejects.data(),
                                    node_invalid_bid_hits.data(),
                                    node_invalid_cid_hits.data()},
                   CountingLeafPredFalse{leaf_calls.data()},
                   qbuffer);

        int h_rejects          = 0;
        int h_leaf_calls       = 0;
        int h_invalid_bid_hits = 0;
        int h_invalid_cid_hits = 0;
        node_rejects.view(0, 1).copy_to(&h_rejects);
        leaf_calls.view(0, 1).copy_to(&h_leaf_calls);
        node_invalid_bid_hits.view(0, 1).copy_to(&h_invalid_bid_hits);
        node_invalid_cid_hits.view(0, 1).copy_to(&h_invalid_cid_hits);
        CHECK(qbuffer.size() == 0);
        CHECK(h_rejects == 0);
        CHECK(h_leaf_calls > 0);
        CHECK(h_invalid_bid_hits > 0);
        CHECK(h_invalid_cid_hits > 0);
    }
}

SimplicialComplex tet()
{
    std::vector<Vector3>  Vs = {Vector3{0.0, 0.0, 0.0},
                                Vector3{1.0, 0.0, 0.0},
                                Vector3{0.0, 1.0, 0.0},
                                Vector3{0.0, 0.0, 1.0}};
    std::vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};

    return tetmesh(Vs, Ts);
}
}  // namespace test_info_stackless_bvh

TEST_CASE("info_stackless_bvh", "[collision detection]")
{
    using namespace test_info_stackless_bvh;

    SECTION("tet")
    {
        fmt::println("tet:");
        run_info_stackless_bvh_test(tet());
    }

    SECTION("cube.obj")
    {
        fmt::println("cube.obj:");
        SimplicialComplexIO io;
        auto mesh = io.read(fmt::format("{}cube.obj", AssetDir::trimesh_path()));
        run_info_stackless_bvh_test(mesh);
    }

    SECTION("internal_cull_proof")
    {
        fmt::println("internal_cull_proof:");
        run_internal_cull_proof_case();
    }

    SECTION("internal_cull_rate")
    {
        fmt::println("internal_cull_rate:");
        run_internal_cull_rate_case();
    }

    SECTION("two_leaf_nodepred_cases")
    {
        fmt::println("two_leaf_nodepred_cases:");
        run_two_leaf_nodepred_cases();
    }
}

TEST_CASE("info_stackless_bvh self Morton-rank pruning",
          "[collision detection][self_rank_pruning]")
{
    using namespace test_info_stackless_bvh;
    run_internal_cull_proof_case();
    run_self_rank_reordered_node_case();
}

TEST_CASE("info_stackless_bvh CTA and queue boundaries",
          "[collision detection][bvh_cta_sweep]")
{
    using namespace test_info_stackless_bvh;
    fmt::println("Self CTA={} queue={}; Other CTA={} queue={}",
                 uipc::info_stackless_detail::K_SELF_THREADS,
                 uipc::info_stackless_detail::K_SELF_MAX_RES_PER_BLOCK,
                 uipc::info_stackless_detail::K_OTHER_THREADS,
                 uipc::info_stackless_detail::K_OTHER_MAX_RES_PER_BLOCK);
    run_cta_queue_boundary_cases();
}

TEST_CASE("info_stackless_bvh topology refit",
          "[collision detection][bvh_refit][line_search]")
{
    using namespace test_info_stackless_bvh;
    run_refit_cases();
}
