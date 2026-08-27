#include <type_define.h>
#include <app/app.h>
#include <collision_detection/info_stackless_bvh.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

using namespace muda;
using namespace uipc;
using namespace uipc::backend::cuda;

namespace test_info_stackless_bvh_trajectory_correctness
{
using PairList = std::vector<Vector2i>;

struct AllowNode
{
    MUDA_GENERIC bool operator()(const InfoStacklessBVH::NodePredInfo&) const
    {
        return true;
    }
};

struct AllowLeaf
{
    MUDA_GENERIC bool operator()(const InfoStacklessBVH::LeafPredInfo&) const
    {
        return true;
    }
};

bool pair_less(const Vector2i& lhs, const Vector2i& rhs)
{
    return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
}

void check_exact_pair_multiset(PairList actual, PairList expected)
{
    std::sort(actual.begin(), actual.end(), pair_less);
    std::sort(expected.begin(), expected.end(), pair_less);

    REQUIRE(actual.size() == expected.size());
    for(size_t i = 0; i < expected.size(); ++i)
    {
        CAPTURE(i);
        REQUIRE(actual[i][0] == expected[i][0]);
        REQUIRE(actual[i][1] == expected[i][1]);
    }
}

AABB make_box(float x, float y, float z, float extent = 0.75f)
{
    AABB box;
    box.extend(Eigen::Vector3f{x, y, z});
    box.extend(Eigen::Vector3f{x + extent, y + extent, z + extent});
    return box;
}

std::vector<AABB> make_sparse_cluster_boxes(size_t count)
{
    constexpr size_t bucket_count = 17;
    std::vector<AABB> boxes(count);
    for(size_t i = 0; i < count; ++i)
    {
        const float x = static_cast<float>(i % bucket_count) * 2.0f;
        boxes[i]      = make_box(x, 0.0f, 0.0f);
    }
    return boxes;
}

std::vector<AABB> make_overlapping_boxes(size_t count)
{
    return std::vector<AABB>(count, make_box(0.0f, 0.0f, 0.0f, 1.0f));
}

std::vector<AABB> make_reverse_morton_overlapping_boxes(size_t count)
{
    std::vector<AABB> boxes(count);
    for(size_t i = 0; i < count; ++i)
    {
        const float x = static_cast<float>(count - i) * 0.01f;
        boxes[i]      = make_box(x, 0.0f, 0.0f, 1.0f);
    }
    return boxes;
}

PairList brute_force_self(const std::vector<AABB>& boxes)
{
    PairList pairs;
    for(size_t i = 0; i < boxes.size(); ++i)
    {
        for(size_t j = i + 1; j < boxes.size(); ++j)
        {
            if(boxes[i].intersects(boxes[j]))
                pairs.emplace_back(static_cast<IndexT>(i),
                                   static_cast<IndexT>(j));
        }
    }
    return pairs;
}

PairList brute_force_other(const std::vector<AABB>& query_boxes,
                           const std::vector<AABB>& tree_boxes)
{
    PairList pairs;
    for(size_t i = 0; i < query_boxes.size(); ++i)
    {
        for(size_t j = 0; j < tree_boxes.size(); ++j)
        {
            if(query_boxes[i].intersects(tree_boxes[j]))
                pairs.emplace_back(static_cast<IndexT>(i),
                                   static_cast<IndexT>(j));
        }
    }
    return pairs;
}

PairList download_pairs(const InfoStacklessBVH::QueryBuffer& qbuffer)
{
    PairList pairs(qbuffer.size());
    if(!pairs.empty())
        qbuffer.view().copy_to(pairs.data());
    return pairs;
}

PairList run_self_allow_all(const std::vector<AABB>& boxes,
                            size_t                   initial_capacity)
{
    std::vector<IndexT> bids(boxes.size(), 0);
    std::vector<IndexT> cids(boxes.size(), 0);

    DeviceBuffer<AABB> d_boxes(boxes.size());
    DeviceBuffer<IndexT> d_bids(bids.size());
    DeviceBuffer<IndexT> d_cids(cids.size());
    if(!boxes.empty())
    {
        d_boxes.view().copy_from(boxes.data());
        d_bids.view().copy_from(bids.data());
        d_cids.view().copy_from(cids.data());
    }

    std::array<IndexT, 1> cmts = {1};
    DeviceBuffer2D<IndexT> d_cmts(Extent2D{1, 1});
    d_cmts.view().copy_from(cmts.data());

    InfoStacklessBVH bvh;
    bvh.build(d_boxes.view(), d_bids.view(), d_cids.view());

    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.reserve(std::max<size_t>(initial_capacity, 1));
    bvh.detect(d_cmts.view(), AllowNode{}, AllowLeaf{}, qbuffer);
    return download_pairs(qbuffer);
}

PairList run_other_allow_all(const std::vector<AABB>& query_boxes,
                             const std::vector<AABB>& tree_boxes,
                             size_t                   initial_capacity)
{
    std::vector<IndexT> query_bids(query_boxes.size(), 0);
    std::vector<IndexT> query_cids(query_boxes.size(), 0);
    std::vector<IndexT> tree_bids(tree_boxes.size(), 0);
    std::vector<IndexT> tree_cids(tree_boxes.size(), 0);

    DeviceBuffer<AABB> d_query_boxes(query_boxes.size());
    DeviceBuffer<IndexT> d_query_bids(query_bids.size());
    DeviceBuffer<IndexT> d_query_cids(query_cids.size());
    if(!query_boxes.empty())
    {
        d_query_boxes.view().copy_from(query_boxes.data());
        d_query_bids.view().copy_from(query_bids.data());
        d_query_cids.view().copy_from(query_cids.data());
    }

    DeviceBuffer<AABB> d_tree_boxes(tree_boxes.size());
    DeviceBuffer<IndexT> d_tree_bids(tree_bids.size());
    DeviceBuffer<IndexT> d_tree_cids(tree_cids.size());
    if(!tree_boxes.empty())
    {
        d_tree_boxes.view().copy_from(tree_boxes.data());
        d_tree_bids.view().copy_from(tree_bids.data());
        d_tree_cids.view().copy_from(tree_cids.data());
    }

    std::array<IndexT, 1> cmts = {1};
    DeviceBuffer2D<IndexT> d_cmts(Extent2D{1, 1});
    d_cmts.view().copy_from(cmts.data());

    InfoStacklessBVH bvh;
    bvh.build(d_tree_boxes.view(), d_tree_bids.view(), d_tree_cids.view());

    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.reserve(std::max<size_t>(initial_capacity, 1));
    bvh.query(d_query_boxes.view(),
              d_query_bids.view(),
              d_query_cids.view(),
              d_cmts.view(),
              AllowNode{},
              AllowLeaf{},
              qbuffer);
    return download_pairs(qbuffer);
}

bool mask_allows(const std::vector<IndexT>& mask,
                 size_t                     dimension,
                 IndexT                     i,
                 IndexT                     j)
{
    return mask[static_cast<size_t>(i) * dimension + static_cast<size_t>(j)]
           != 0;
}

bool mask_allows_ee(const std::vector<IndexT>& mask,
                    size_t                     dimension,
                    const Vector4i&            ids)
{
    return mask_allows(mask, dimension, ids[0], ids[2])
           && mask_allows(mask, dimension, ids[0], ids[3])
           && mask_allows(mask, dimension, ids[1], ids[2])
           && mask_allows(mask, dimension, ids[1], ids[3]);
}

bool mask_allows_pt(const std::vector<IndexT>& mask,
                    size_t                     dimension,
                    const Vector4i&            ids)
{
    return mask_allows(mask, dimension, ids[0], ids[1])
           && mask_allows(mask, dimension, ids[0], ids[2])
           && mask_allows(mask, dimension, ids[0], ids[3]);
}

struct MaskCoverage
{
    bool topology = false;
    bool body      = false;
    bool contact   = false;
    bool subscene  = false;
};

struct EEMaskCase
{
    static constexpr size_t mask_dimension = 3;

    std::vector<AABB>     boxes;
    std::vector<Vector2i> edges;
    std::vector<IndexT>   edge_bids;
    std::vector<IndexT>   edge_cids;
    std::vector<IndexT>   vertex_cids;
    std::vector<IndexT>   vertex_scids;
    std::vector<IndexT>   body_self_collision;
    std::vector<IndexT>   contact_mask;
    std::vector<IndexT>   subscene_mask;
};

EEMaskCase make_ee_mask_case()
{
    EEMaskCase data;
    data.edges = {{0, 1},
                  {1, 2},
                  {3, 4},
                  {5, 6},
                  {7, 8},
                  {9, 10},
                  {11, 12},
                  {13, 14}};
    data.boxes = make_reverse_morton_overlapping_boxes(data.edges.size());

    std::vector<IndexT> vertex_bids(15, 1);
    for(IndexT vertex : {IndexT{3}, IndexT{4}, IndexT{5}, IndexT{6}})
        vertex_bids[vertex] = 0;
    for(IndexT vertex : {IndexT{9}, IndexT{10}, IndexT{13}, IndexT{14}})
        vertex_bids[vertex] = 2;

    data.vertex_cids.assign(15, 0);
    data.vertex_cids[7]  = 1;
    data.vertex_cids[8]  = 1;
    data.vertex_cids[9]  = 2;
    data.vertex_cids[10] = 2;

    data.vertex_scids.assign(15, 0);
    data.vertex_scids[11] = 1;
    data.vertex_scids[12] = 1;
    data.vertex_scids[13] = 2;
    data.vertex_scids[14] = 2;

    data.edge_bids.resize(data.edges.size());
    data.edge_cids.resize(data.edges.size());
    for(size_t i = 0; i < data.edges.size(); ++i)
    {
        data.edge_bids[i] = vertex_bids[data.edges[i][0]];
        data.edge_cids[i] = data.vertex_cids[data.edges[i][0]];
    }

    data.body_self_collision = {0, 1, 1};
    data.contact_mask.assign(
        EEMaskCase::mask_dimension * EEMaskCase::mask_dimension, 1);
    data.contact_mask[1 * EEMaskCase::mask_dimension + 2] = 0;
    data.contact_mask[2 * EEMaskCase::mask_dimension + 1] = 0;
    data.subscene_mask.assign(
        EEMaskCase::mask_dimension * EEMaskCase::mask_dimension, 1);
    data.subscene_mask[1 * EEMaskCase::mask_dimension + 2] = 0;
    data.subscene_mask[2 * EEMaskCase::mask_dimension + 1] = 0;
    return data;
}

PairList brute_force_ee_masked(const EEMaskCase& data,
                               MaskCoverage&     coverage)
{
    constexpr IndexT invalid = static_cast<IndexT>(-1);
    PairList          pairs;
    for(size_t i = 0; i < data.edges.size(); ++i)
    {
        for(size_t j = i + 1; j < data.edges.size(); ++j)
        {
            if(!data.boxes[i].intersects(data.boxes[j]))
                continue;

            const auto& lhs = data.edges[i];
            const auto& rhs = data.edges[j];
            const IndexT lhs_bid = data.edge_bids[i];
            const bool topology_reject =
                lhs[0] == rhs[0] || lhs[0] == rhs[1] || lhs[1] == rhs[0]
                || lhs[1] == rhs[1];
            const bool body_reject =
                lhs_bid == data.edge_bids[j] && lhs_bid != invalid
                && !data.body_self_collision[lhs_bid];
            const Vector4i cids = {data.vertex_cids[lhs[0]],
                                   data.vertex_cids[lhs[1]],
                                   data.vertex_cids[rhs[0]],
                                   data.vertex_cids[rhs[1]]};
            const Vector4i scids = {data.vertex_scids[lhs[0]],
                                    data.vertex_scids[lhs[1]],
                                    data.vertex_scids[rhs[0]],
                                    data.vertex_scids[rhs[1]]};
            const bool contact_reject =
                !mask_allows_ee(data.contact_mask,
                                EEMaskCase::mask_dimension,
                                cids);
            const bool subscene_reject =
                !mask_allows_ee(data.subscene_mask,
                                EEMaskCase::mask_dimension,
                                scids);

            coverage.topology |= topology_reject;
            coverage.body |= body_reject;
            coverage.contact |= contact_reject;
            coverage.subscene |= subscene_reject;

            if(!(topology_reject || body_reject || contact_reject
                 || subscene_reject))
                pairs.emplace_back(static_cast<IndexT>(i),
                                   static_cast<IndexT>(j));
        }
    }
    return pairs;
}

PairList run_ee_masked(const EEMaskCase& data, size_t initial_capacity)
{
    DeviceBuffer<AABB> d_boxes(data.boxes.size());
    d_boxes.view().copy_from(data.boxes.data());
    DeviceBuffer<Vector2i> d_edges(data.edges.size());
    d_edges.view().copy_from(data.edges.data());
    DeviceBuffer<IndexT> d_edge_bids(data.edge_bids.size());
    d_edge_bids.view().copy_from(data.edge_bids.data());
    DeviceBuffer<IndexT> d_edge_cids(data.edge_cids.size());
    d_edge_cids.view().copy_from(data.edge_cids.data());
    DeviceBuffer<IndexT> d_vertex_cids(data.vertex_cids.size());
    d_vertex_cids.view().copy_from(data.vertex_cids.data());
    DeviceBuffer<IndexT> d_vertex_scids(data.vertex_scids.size());
    d_vertex_scids.view().copy_from(data.vertex_scids.data());
    DeviceBuffer<IndexT> d_body_self_collision(data.body_self_collision.size());
    d_body_self_collision.view().copy_from(data.body_self_collision.data());
    DeviceBuffer2D<IndexT> d_contact_mask(
        Extent2D{EEMaskCase::mask_dimension, EEMaskCase::mask_dimension});
    d_contact_mask.view().copy_from(data.contact_mask.data());
    DeviceBuffer2D<IndexT> d_subscene_mask(
        Extent2D{EEMaskCase::mask_dimension, EEMaskCase::mask_dimension});
    d_subscene_mask.view().copy_from(data.subscene_mask.data());

    InfoStacklessBVH bvh;
    bvh.build(d_boxes.view(), d_edge_bids.view(), d_edge_cids.view());
    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.reserve(std::max<size_t>(initial_capacity, 1));
    bvh.detect(
        d_contact_mask.view(),
        [body_self_collision =
             d_body_self_collision.viewer().name("body_self_collision"),
         contact_mask =
             d_contact_mask.viewer().name("contact_mask")] __device__(
            InfoStacklessBVH::NodePredInfo info)
        {
            constexpr IndexT invalid = static_cast<IndexT>(-1);
            const bool bid_cull =
                info.node_bid != invalid && info.query_bid != invalid
                && info.node_bid == info.query_bid
                && !body_self_collision(info.query_bid);
            const bool cid_cull =
                info.node_cid != invalid && info.query_cid != invalid
                && !contact_mask(info.query_cid, info.node_cid);
            return !(bid_cull || cid_cull);
        },
        [edges = d_edges.viewer().name("edges"),
         edge_bids = d_edge_bids.viewer().name("edge_bids"),
         edge_cids = d_edge_cids.viewer().name("edge_cids"),
         vertex_cids = d_vertex_cids.viewer().name("vertex_cids"),
         vertex_scids = d_vertex_scids.viewer().name("vertex_scids"),
         body_self_collision =
             d_body_self_collision.viewer().name("body_self_collision"),
         contact_mask = d_contact_mask.viewer().name("contact_mask"),
         subscene_mask =
             d_subscene_mask.viewer().name("subscene_mask")] __device__(
            InfoStacklessBVH::LeafPredInfo info)
        {
            constexpr IndexT invalid = static_cast<IndexT>(-1);
            const auto       lhs     = edges(info.i);
            const auto       rhs     = edges(info.j);
            if(info.bid_i != edge_bids(info.i)
               || info.cid_i != edge_cids(info.i)
               || info.bid_j != edge_bids(info.j)
               || info.cid_j != edge_cids(info.j))
                return false;
            if(lhs[0] == rhs[0] || lhs[0] == rhs[1] || lhs[1] == rhs[0]
               || lhs[1] == rhs[1])
                return false;
            if(info.bid_i == info.bid_j && info.bid_i != invalid
               && !body_self_collision(info.bid_i))
                return false;

            const Vector4i cids = {vertex_cids(lhs[0]),
                                   vertex_cids(lhs[1]),
                                   vertex_cids(rhs[0]),
                                   vertex_cids(rhs[1])};
            if(!(contact_mask(cids[0], cids[2])
                 && contact_mask(cids[0], cids[3])
                 && contact_mask(cids[1], cids[2])
                 && contact_mask(cids[1], cids[3])))
                return false;

            const Vector4i scids = {vertex_scids(lhs[0]),
                                    vertex_scids(lhs[1]),
                                    vertex_scids(rhs[0]),
                                    vertex_scids(rhs[1])};
            return subscene_mask(scids[0], scids[2])
                   && subscene_mask(scids[0], scids[3])
                   && subscene_mask(scids[1], scids[2])
                   && subscene_mask(scids[1], scids[3]);
        },
        qbuffer);
    return download_pairs(qbuffer);
}

struct PTMaskCase
{
    static constexpr size_t mask_dimension = 3;

    std::vector<AABB>     point_boxes;
    std::vector<AABB>     triangle_boxes;
    std::vector<IndexT>   points;
    std::vector<Vector3i> triangles;
    std::vector<IndexT>   point_bids;
    std::vector<IndexT>   point_cids;
    std::vector<IndexT>   triangle_bids;
    std::vector<IndexT>   triangle_cids;
    std::vector<IndexT>   vertex_cids;
    std::vector<IndexT>   vertex_scids;
    std::vector<IndexT>   body_self_collision;
    std::vector<IndexT>   contact_mask;
    std::vector<IndexT>   subscene_mask;
};

PTMaskCase make_pt_mask_case()
{
    PTMaskCase data;
    data.points    = {0, 18, 19, 20, 21};
    data.triangles = {{0, 1, 2},
                      {3, 4, 5},
                      {6, 7, 8},
                      {9, 10, 11},
                      {12, 13, 14},
                      {15, 16, 17}};
    data.point_boxes =
        make_reverse_morton_overlapping_boxes(data.points.size());
    data.triangle_boxes =
        make_reverse_morton_overlapping_boxes(data.triangles.size());

    std::vector<IndexT> vertex_bids(22, 1);
    for(IndexT vertex : {IndexT{3}, IndexT{4}, IndexT{5}, IndexT{18}})
        vertex_bids[vertex] = 0;
    for(IndexT vertex : {IndexT{6},
                         IndexT{7},
                         IndexT{8},
                         IndexT{9},
                         IndexT{10},
                         IndexT{11},
                         IndexT{12},
                         IndexT{13},
                         IndexT{14}})
        vertex_bids[vertex] = 2;

    data.vertex_cids.assign(22, 0);
    data.vertex_cids[6]  = 2;
    data.vertex_cids[7]  = 2;
    data.vertex_cids[8]  = 2;
    data.vertex_cids[19] = 1;

    data.vertex_scids.assign(22, 0);
    data.vertex_scids[9]  = 2;
    data.vertex_scids[10] = 2;
    data.vertex_scids[11] = 2;
    data.vertex_scids[20] = 1;

    data.point_bids.resize(data.points.size());
    data.point_cids.resize(data.points.size());
    for(size_t i = 0; i < data.points.size(); ++i)
    {
        data.point_bids[i] = vertex_bids[data.points[i]];
        data.point_cids[i] = data.vertex_cids[data.points[i]];
    }

    data.triangle_bids.resize(data.triangles.size());
    data.triangle_cids.resize(data.triangles.size());
    for(size_t i = 0; i < data.triangles.size(); ++i)
    {
        data.triangle_bids[i] = vertex_bids[data.triangles[i][0]];
        data.triangle_cids[i] = data.vertex_cids[data.triangles[i][0]];
    }

    data.body_self_collision = {0, 1, 1};
    data.contact_mask.assign(
        PTMaskCase::mask_dimension * PTMaskCase::mask_dimension, 1);
    data.contact_mask[1 * PTMaskCase::mask_dimension + 2] = 0;
    data.contact_mask[2 * PTMaskCase::mask_dimension + 1] = 0;
    data.subscene_mask.assign(
        PTMaskCase::mask_dimension * PTMaskCase::mask_dimension, 1);
    data.subscene_mask[1 * PTMaskCase::mask_dimension + 2] = 0;
    data.subscene_mask[2 * PTMaskCase::mask_dimension + 1] = 0;
    return data;
}

PairList brute_force_pt_masked(const PTMaskCase& data,
                               MaskCoverage&     coverage)
{
    constexpr IndexT invalid = static_cast<IndexT>(-1);
    PairList          pairs;
    for(size_t i = 0; i < data.points.size(); ++i)
    {
        for(size_t j = 0; j < data.triangles.size(); ++j)
        {
            if(!data.point_boxes[i].intersects(data.triangle_boxes[j]))
                continue;

            const IndexT   point    = data.points[i];
            const Vector3i triangle = data.triangles[j];
            const IndexT   point_bid = data.point_bids[i];
            const bool topology_reject =
                point == triangle[0] || point == triangle[1]
                || point == triangle[2];
            const bool body_reject =
                point_bid == data.triangle_bids[j] && point_bid != invalid
                && !data.body_self_collision[point_bid];
            const Vector4i cids = {data.vertex_cids[point],
                                   data.vertex_cids[triangle[0]],
                                   data.vertex_cids[triangle[1]],
                                   data.vertex_cids[triangle[2]]};
            const Vector4i scids = {data.vertex_scids[point],
                                    data.vertex_scids[triangle[0]],
                                    data.vertex_scids[triangle[1]],
                                    data.vertex_scids[triangle[2]]};
            const bool contact_reject =
                !mask_allows_pt(data.contact_mask,
                                PTMaskCase::mask_dimension,
                                cids);
            const bool subscene_reject =
                !mask_allows_pt(data.subscene_mask,
                                PTMaskCase::mask_dimension,
                                scids);

            coverage.topology |= topology_reject;
            coverage.body |= body_reject;
            coverage.contact |= contact_reject;
            coverage.subscene |= subscene_reject;

            if(!(topology_reject || body_reject || contact_reject
                 || subscene_reject))
                pairs.emplace_back(static_cast<IndexT>(i),
                                   static_cast<IndexT>(j));
        }
    }
    return pairs;
}

PairList run_pt_masked(const PTMaskCase& data, size_t initial_capacity)
{
    DeviceBuffer<AABB> d_point_boxes(data.point_boxes.size());
    d_point_boxes.view().copy_from(data.point_boxes.data());
    DeviceBuffer<AABB> d_triangle_boxes(data.triangle_boxes.size());
    d_triangle_boxes.view().copy_from(data.triangle_boxes.data());
    DeviceBuffer<IndexT> d_points(data.points.size());
    d_points.view().copy_from(data.points.data());
    DeviceBuffer<Vector3i> d_triangles(data.triangles.size());
    d_triangles.view().copy_from(data.triangles.data());
    DeviceBuffer<IndexT> d_point_bids(data.point_bids.size());
    d_point_bids.view().copy_from(data.point_bids.data());
    DeviceBuffer<IndexT> d_point_cids(data.point_cids.size());
    d_point_cids.view().copy_from(data.point_cids.data());
    DeviceBuffer<IndexT> d_triangle_bids(data.triangle_bids.size());
    d_triangle_bids.view().copy_from(data.triangle_bids.data());
    DeviceBuffer<IndexT> d_triangle_cids(data.triangle_cids.size());
    d_triangle_cids.view().copy_from(data.triangle_cids.data());
    DeviceBuffer<IndexT> d_vertex_cids(data.vertex_cids.size());
    d_vertex_cids.view().copy_from(data.vertex_cids.data());
    DeviceBuffer<IndexT> d_vertex_scids(data.vertex_scids.size());
    d_vertex_scids.view().copy_from(data.vertex_scids.data());
    DeviceBuffer<IndexT> d_body_self_collision(data.body_self_collision.size());
    d_body_self_collision.view().copy_from(data.body_self_collision.data());
    DeviceBuffer2D<IndexT> d_contact_mask(
        Extent2D{PTMaskCase::mask_dimension, PTMaskCase::mask_dimension});
    d_contact_mask.view().copy_from(data.contact_mask.data());
    DeviceBuffer2D<IndexT> d_subscene_mask(
        Extent2D{PTMaskCase::mask_dimension, PTMaskCase::mask_dimension});
    d_subscene_mask.view().copy_from(data.subscene_mask.data());

    InfoStacklessBVH bvh;
    bvh.build(d_triangle_boxes.view(),
              d_triangle_bids.view(),
              d_triangle_cids.view());
    InfoStacklessBVH::QueryBuffer qbuffer;
    qbuffer.reserve(std::max<size_t>(initial_capacity, 1));
    bvh.query(
        d_point_boxes.view(),
        d_point_bids.view(),
        d_point_cids.view(),
        d_contact_mask.view(),
        [body_self_collision =
             d_body_self_collision.viewer().name("body_self_collision"),
         contact_mask =
             d_contact_mask.viewer().name("contact_mask")] __device__(
            InfoStacklessBVH::NodePredInfo info)
        {
            constexpr IndexT invalid = static_cast<IndexT>(-1);
            const bool bid_cull =
                info.node_bid != invalid && info.query_bid != invalid
                && info.node_bid == info.query_bid
                && !body_self_collision(info.query_bid);
            const bool cid_cull =
                info.node_cid != invalid && info.query_cid != invalid
                && !contact_mask(info.query_cid, info.node_cid);
            return !(bid_cull || cid_cull);
        },
        [points = d_points.viewer().name("points"),
         triangles = d_triangles.viewer().name("triangles"),
         point_bids = d_point_bids.viewer().name("point_bids"),
         point_cids = d_point_cids.viewer().name("point_cids"),
         triangle_bids = d_triangle_bids.viewer().name("triangle_bids"),
         triangle_cids = d_triangle_cids.viewer().name("triangle_cids"),
         vertex_cids = d_vertex_cids.viewer().name("vertex_cids"),
         vertex_scids = d_vertex_scids.viewer().name("vertex_scids"),
         body_self_collision =
             d_body_self_collision.viewer().name("body_self_collision"),
         contact_mask = d_contact_mask.viewer().name("contact_mask"),
         subscene_mask =
             d_subscene_mask.viewer().name("subscene_mask")] __device__(
            InfoStacklessBVH::LeafPredInfo info)
        {
            constexpr IndexT invalid  = static_cast<IndexT>(-1);
            const IndexT    point    = points(info.i);
            const Vector3i  triangle = triangles(info.j);
            if(info.bid_i != point_bids(info.i)
               || info.cid_i != point_cids(info.i)
               || info.bid_j != triangle_bids(info.j)
               || info.cid_j != triangle_cids(info.j))
                return false;
            if(point == triangle[0] || point == triangle[1]
               || point == triangle[2])
                return false;
            if(info.bid_i == info.bid_j && info.bid_i != invalid
               && !body_self_collision(info.bid_i))
                return false;

            const Vector4i cids = {vertex_cids(point),
                                   vertex_cids(triangle[0]),
                                   vertex_cids(triangle[1]),
                                   vertex_cids(triangle[2])};
            if(!(contact_mask(cids[0], cids[1])
                 && contact_mask(cids[0], cids[2])
                 && contact_mask(cids[0], cids[3])))
                return false;

            const Vector4i scids = {vertex_scids(point),
                                    vertex_scids(triangle[0]),
                                    vertex_scids(triangle[1]),
                                    vertex_scids(triangle[2])};
            return subscene_mask(scids[0], scids[1])
                   && subscene_mask(scids[0], scids[2])
                   && subscene_mask(scids[0], scids[3]);
        },
        qbuffer);
    return download_pairs(qbuffer);
}
}  // namespace test_info_stackless_bvh_trajectory_correctness

TEST_CASE("T00 InfoStacklessBVH exact pairs at launch boundaries",
          "[collision detection][trajectory_bvh][T00]")
{
    using namespace test_info_stackless_bvh_trajectory_correctness;
    constexpr std::array<size_t, 11> counts = {
        0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257};

    for(size_t count : counts)
    {
        DYNAMIC_SECTION("stacklessSelf N=" << count)
        {
            auto boxes    = make_sparse_cluster_boxes(count);
            auto expected = brute_force_self(boxes);
            auto actual   = run_self_allow_all(boxes, expected.size() + 1);
            check_exact_pair_multiset(std::move(actual), std::move(expected));
        }
    }

    const auto tree_boxes = make_sparse_cluster_boxes(65);
    for(size_t count : counts)
    {
        DYNAMIC_SECTION("stacklessOther query N=" << count)
        {
            auto query_boxes = make_sparse_cluster_boxes(count);
            auto expected    = brute_force_other(query_boxes, tree_boxes);
            auto actual = run_other_allow_all(
                query_boxes, tree_boxes, expected.size() + 1);
            check_exact_pair_multiset(std::move(actual), std::move(expected));
        }
    }
}

TEST_CASE("T00 InfoStacklessBVH trajectory masks preserve exact pairs",
          "[collision detection][trajectory_bvh][T00]")
{
    using namespace test_info_stackless_bvh_trajectory_correctness;

    SECTION("EE-like stacklessSelf")
    {
        auto         data = make_ee_mask_case();
        MaskCoverage coverage;
        auto         expected = brute_force_ee_masked(data, coverage);
        REQUIRE(coverage.topology);
        REQUIRE(coverage.body);
        REQUIRE(coverage.contact);
        REQUIRE(coverage.subscene);
        REQUIRE(!expected.empty());

        auto actual = run_ee_masked(data, expected.size() + 1);
        check_exact_pair_multiset(std::move(actual), std::move(expected));
    }

    SECTION("PT-like stacklessOther")
    {
        auto         data = make_pt_mask_case();
        MaskCoverage coverage;
        auto         expected = brute_force_pt_masked(data, coverage);
        REQUIRE(coverage.topology);
        REQUIRE(coverage.body);
        REQUIRE(coverage.contact);
        REQUIRE(coverage.subscene);
        REQUIRE(!expected.empty());

        auto actual = run_pt_masked(data, expected.size() + 1);
        check_exact_pair_multiset(std::move(actual), std::move(expected));
    }
}

TEST_CASE("T00 InfoStacklessBVH retries undersized output capacity",
          "[collision detection][trajectory_bvh][T00]")
{
    using namespace test_info_stackless_bvh_trajectory_correctness;
    constexpr size_t initial_capacity = 7;

    SECTION("stacklessSelf spans multiple shared-buffer flushes")
    {
        auto boxes    = make_overlapping_boxes(65);
        auto expected = brute_force_self(boxes);
        REQUIRE(expected.size() == 2080);
        REQUIRE(expected.size() > 1024);
        REQUIRE(expected.size() > initial_capacity);

        auto actual = run_self_allow_all(boxes, initial_capacity);
        check_exact_pair_multiset(std::move(actual), std::move(expected));
    }

    SECTION("stacklessOther spans multiple shared-buffer flushes")
    {
        auto query_boxes = make_overlapping_boxes(33);
        auto tree_boxes  = make_overlapping_boxes(33);
        auto expected    = brute_force_other(query_boxes, tree_boxes);
        REQUIRE(expected.size() == 1089);
        REQUIRE(expected.size() > 1024);
        REQUIRE(expected.size() > initial_capacity);

        auto actual =
            run_other_allow_all(query_boxes, tree_boxes, initial_capacity);
        check_exact_pair_multiset(std::move(actual), std::move(expected));
    }
}
