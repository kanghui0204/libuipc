#include <cuda_device/builtin.h>
#include <muda/launch.h>

// Implementation of InfoStacklessBVH.
// All build/sort/reorder functions are identical to InfoStacklessBVH.
// The two traversal functions (stacklessSelf / stacklessOther) are the
// optimized variants: they pre-load per-query bid/cid into shared memory
// ONCE before the traversal loop, eliminating repeated global-memory reads
// of query_bid/query_cid inside the hot node-cull path.

namespace uipc::info_stackless_detail
{
using aabb     = uipc::backend::cuda::AABB;
using node_t   = uipc::backend::cuda::InfoStacklessBVH::Node;
using Vector2i = uipc::Vector2i;
using uint     = uint32_t;
using ullint   = unsigned long long;

constexpr int K_BUILD_THREADS = 256;
constexpr int K_BUILD_WARPS   = K_BUILD_THREADS >> 5;

constexpr int K_SELF_THREADS                = 64;
constexpr int K_SELF_QUEUE_SLOTS_PER_THREAD = 4;
constexpr int K_SELF_MAX_RES_PER_BLOCK =
    K_SELF_THREADS * K_SELF_QUEUE_SLOTS_PER_THREAD;

constexpr int K_OTHER_THREADS                = 64;
constexpr int K_OTHER_QUEUE_SLOTS_PER_THREAD = 4;
constexpr int K_OTHER_MAX_RES_PER_BLOCK =
    K_OTHER_THREADS * K_OTHER_QUEUE_SLOTS_PER_THREAD;

static_assert(K_BUILD_THREADS % 32 == 0);
static_assert(K_SELF_THREADS % 32 == 0);
static_assert(K_OTHER_THREADS % 32 == 0);
constexpr int  AABB_BITS         = 15;
constexpr uint AABB_MASK         = 0xFFFFFFFFu >> (32 - AABB_BITS);

struct PlainAABB
{
    float3 _min, _max;
};

MUDA_GENERIC MUDA_INLINE PlainAABB to_plain(const aabb& box)
{
    PlainAABB out;
    out._min = make_float3(box.min().x(), box.min().y(), box.min().z());
    out._max = make_float3(box.max().x(), box.max().y(), box.max().z());
    return out;
}

template <typename T>
MUDA_GENERIC MUDA_INLINE T mm_min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
MUDA_GENERIC MUDA_INLINE T mm_max(T a, T b)
{
    return a > b ? a : b;
}

MUDA_DEVICE MUDA_INLINE float atomic_minf(float* addr, float value)
{
    // Classify by the IEEE-754 sign bit: -0.0f compares >= 0.0f but must use
    // the negative-value integer ordering.
    const int value_bits = __float_as_int(value);
    return (value_bits >= 0) ?
               __int_as_float(atomicMin((int*)addr, value_bits)) :
               __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
}

MUDA_DEVICE MUDA_INLINE float atomic_maxf(float* addr, float value)
{
    // See atomic_minf: a numeric comparison misclassifies negative zero.
    const int value_bits = __float_as_int(value);
    return (value_bits >= 0) ?
               __int_as_float(atomicMax((int*)addr, value_bits)) :
               __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}

MUDA_GENERIC MUDA_INLINE uint expand_bits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

MUDA_GENERIC MUDA_INLINE uint morton3D(float x, float y, float z)
{
    x       = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
    y       = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
    z       = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
    uint xx = expand_bits((uint)x);
    uint yy = expand_bits((uint)y);
    uint zz = expand_bits((uint)z);
    return xx * 4 + yy * 2 + zz;
}

MUDA_GENERIC MUDA_INLINE Vector2i to_eigen(int2 v)
{
    return Vector2i{v.x, v.y};
}

MUDA_GENERIC MUDA_INLINE int2 ordered_pair(int a, int b)
{
    return (a < b) ? int2{a, b} : int2{b, a};
}

MUDA_GENERIC MUDA_INLINE float3 operator-(const float3& v0, const float3& v1)
{
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

MUDA_GENERIC MUDA_INLINE void safe_copy_to(int2*     shared_res,
                                           int       total_in_block,
                                           Vector2i* global_res,
                                           int       global_idx,
                                           int       max_res)
{
    if(global_idx >= max_res || total_in_block == 0)
        return;
    auto copy_count  = std::min(total_in_block, max_res - global_idx);
    int  full_blocks = (copy_count - 1) / (int)blockDim.x;
    for(int i = 0; i < full_blocks; ++i)
    {
        int offset                      = i * blockDim.x + threadIdx.x;
        global_res[global_idx + offset] = to_eigen(shared_res[offset]);
    }
    int offset = full_blocks * blockDim.x + threadIdx.x;
    if(offset < copy_count)
        global_res[global_idx + offset] = to_eigen(shared_res[offset]);
}
}  // namespace uipc::info_stackless_detail

namespace uipc::backend::cuda
{
using namespace info_stackless_detail;

// ---------------------------------------------------------------------------
// Build pipeline — identical to InfoStacklessBVH
// ---------------------------------------------------------------------------

inline void InfoStacklessBVH::Impl::calcMaxBVFromBox(muda::CBufferView<AABB> aabbs,
                                                     muda::VarView<AABB> scene_box)
{
    using namespace muda;

    // Reset in its own launch. A reset performed by one block inside the
    // reduction kernel can race with atomic updates from another block.
    Launch(1, 1)
        .file_line(__FILE__, __LINE__)
        .apply([out = scene_box.viewer().name("out")] __device__() { *out = AABB(); });

    if(aabbs.size() == 0)
        return;

    auto num  = aabbs.size();
    auto grid = (num + K_BUILD_THREADS - 1) / K_BUILD_THREADS;

    Launch(grid, K_BUILD_THREADS)
        .file_line(__FILE__, __LINE__)
        .apply(
            [size = aabbs.size(),
             box  = aabbs.viewer().name("box"),
             out  = scene_box.viewer().name("out")] __device__()
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                __shared__ PlainAABB warp_boxes[K_BUILD_WARPS];
                int                  warp_tid = threadIdx.x & 31;
                int                  warp_id  = threadIdx.x >> 5;

                PlainAABB temp;
                if(idx < size)
                {
                    temp = to_plain(box(idx));
                }
                else
                {
                    constexpr float max_float = 3.402823466e+38F;
                    temp._min = make_float3(max_float, max_float, max_float);
                    temp._max = make_float3(-max_float, -max_float, -max_float);
                }

                float minx = temp._min.x, miny = temp._min.y, minz = temp._min.z;
                float maxx = temp._max.x, maxy = temp._max.y, maxz = temp._max.z;
                for(int offset = 16; offset > 0; offset >>= 1)
                {
                    minx = mm_min(minx, __shfl_down_sync(0xffffffff, minx, offset));
                    miny = mm_min(miny, __shfl_down_sync(0xffffffff, miny, offset));
                    minz = mm_min(minz, __shfl_down_sync(0xffffffff, minz, offset));
                    maxx = mm_max(maxx, __shfl_down_sync(0xffffffff, maxx, offset));
                    maxy = mm_max(maxy, __shfl_down_sync(0xffffffff, maxy, offset));
                    maxz = mm_max(maxz, __shfl_down_sync(0xffffffff, maxz, offset));
                }
                if(warp_tid == 0)
                {
                    warp_boxes[warp_id]._min = make_float3(minx, miny, minz);
                    warp_boxes[warp_id]._max = make_float3(maxx, maxy, maxz);
                }
                __syncthreads();

                if(warp_id == 0)
                {
                    constexpr float max_float = 3.402823466e+38F;
                    if(warp_tid < K_BUILD_WARPS)
                        temp = warp_boxes[warp_tid];
                    else
                    {
                        temp._min = make_float3(max_float, max_float, max_float);
                        temp._max = make_float3(-max_float, -max_float, -max_float);
                    }

                    minx = temp._min.x;
                    miny = temp._min.y;
                    minz = temp._min.z;
                    maxx = temp._max.x;
                    maxy = temp._max.y;
                    maxz = temp._max.z;
                    for(int offset = 16; offset > 0; offset >>= 1)
                    {
                        minx = mm_min(minx, __shfl_down_sync(0xffffffff, minx, offset));
                        miny = mm_min(miny, __shfl_down_sync(0xffffffff, miny, offset));
                        minz = mm_min(minz, __shfl_down_sync(0xffffffff, minz, offset));
                        maxx = mm_max(maxx, __shfl_down_sync(0xffffffff, maxx, offset));
                        maxy = mm_max(maxy, __shfl_down_sync(0xffffffff, maxy, offset));
                        maxz = mm_max(maxz, __shfl_down_sync(0xffffffff, maxz, offset));
                    }
                    if(warp_tid == 0)
                    {
                        atomic_minf(&out->min().x(), minx);
                        atomic_minf(&out->min().y(), miny);
                        atomic_minf(&out->min().z(), minz);
                        atomic_maxf(&out->max().x(), maxx);
                        atomic_maxf(&out->max().y(), maxy);
                        atomic_maxf(&out->max().z(), maxz);
                    }
                }
            });
}

inline void InfoStacklessBVH::Impl::calcMCsFromBox(muda::CBufferView<AABB> aabbs,
                                                   muda::CVarView<AABB> scene_box,
                                                   muda::BufferView<uint32_t> codes)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(aabbs.size(),
               [box   = aabbs.viewer().name("box"),
                scene = scene_box.viewer().name("scene"),
                codes = codes.viewer().name("codes")] __device__(int idx)
               {
                   auto   bv     = box(idx);
                   auto   center = bv.center();
                   float3 c = make_float3(center.x(), center.y(), center.z());
                   auto   scene_min = scene->min();
                   float3 smin =
                       make_float3(scene_min.x(), scene_min.y(), scene_min.z());
                   auto   scene_size = scene->sizes();
                   float3 off        = c - smin;
                   float nx = scene_size.x() > 0.0f ? off.x / scene_size.x() : 0.0f;
                   float ny = scene_size.y() > 0.0f ? off.y / scene_size.y() : 0.0f;
                   float nz = scene_size.z() > 0.0f ? off.z / scene_size.z() : 0.0f;
                   codes(idx) = morton3D(nx, ny, nz);
               });
}

inline void InfoStacklessBVH::Impl::calcInverseMapping()
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sorted_id.size(),
               [map = sorted_id.viewer().name("map"),
                inv = primMap.viewer().name("inv")] __device__(int idx)
               { inv(map(idx)) = idx; });
}

inline void InfoStacklessBVH::Impl::buildPrimitivesFromBox(muda::CBufferView<AABB> aabbs)
{
    using namespace muda;
    constexpr IndexT invalid = static_cast<IndexT>(-1);
    bool has_info = bids.size() == aabbs.size() && cids.size() == aabbs.size();
    ParallelFor().apply(aabbs.size(),
                        [_prim_idx = ext_idx.viewer().name("prim_idx"),
                         _prim_box = ext_aabb.viewer().name("prim_box"),
                         _prim_map = primMap.viewer().name("prim_map"),
                         _ext_bid  = ext_bid.viewer().name("ext_bid"),
                         _ext_cid  = ext_cid.viewer().name("ext_cid"),
                         _bids     = bids.viewer().name("bids"),
                         _cids     = cids.viewer().name("cids"),
                         has_info,
                         box = aabbs.viewer().name("box")] __device__(int idx)
                        {
                            int new_idx        = _prim_map(idx);
                            _prim_idx(new_idx) = idx;
                            _prim_box(new_idx) = box(idx);
                            if(has_info)
                            {
                                _ext_bid(new_idx) = _bids(idx);
                                _ext_cid(new_idx) = _cids(idx);
                            }
                            else
                            {
                                _ext_bid(new_idx) = invalid;
                                _ext_cid(new_idx) = invalid;
                            }
                        });
}

inline void InfoStacklessBVH::Impl::calcExtNodeSplitMetrics()
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(mtcode.size(),
               [n       = mtcode.size(),
                codes   = mtcode.viewer().name("codes"),
                metrics = metric.viewer().name("metrics")] __device__(int idx)
               {
                   metrics(idx) =
                       idx != n - 1 ? 32 - __clz(codes(idx) ^ codes(idx + 1)) : 33;
               });
}

inline void InfoStacklessBVH::Impl::buildIntNodes(int size)
{
    using namespace muda;
    constexpr IndexT invalid = static_cast<IndexT>(-1);
    auto             grid    = (size + 255) / 256;
    Launch(grid, 256)
        .file_line(__FILE__, __LINE__)
        .apply(
            [size,
             _depths      = count.viewer().name("depths"),
             _lvs_lca     = ext_lca.viewer().name("lvs_lca"),
             _lvs_metric  = metric.viewer().name("lvs_metric"),
             _lvs_par     = ext_par.viewer().name("lvs_par"),
             _lvs_box     = ext_aabb.viewer().name("lvs_box"),
             _lvs_bid     = ext_bid.viewer().name("lvs_bid"),
             _lvs_cid     = ext_cid.viewer().name("lvs_cid"),
             _tks_lc      = int_lc.viewer().name("tks_lc"),
             _tks_rc      = int_rc.viewer().name("tks_rc"),
             _tks_range_x = int_range_x.viewer().name("tks_range_x"),
             _tks_range_y = int_range_y.viewer().name("tks_range_y"),
             _tks_mark    = int_mark.viewer().name("tks_mark"),
             _tks_box     = int_aabb.viewer().name("tks_box"),
             _tks_bid     = int_bid.viewer().name("tks_bid"),
             _tks_cid     = int_cid.viewer().name("tks_cid"),
             _flag        = flags.viewer().name("flag"),
             _tks_par     = int_par.viewer().name("tks_par")] __device__()
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if(idx >= size)
                    return;

                _lvs_lca(idx) = -1;
                _depths(idx)  = 0;
                int l         = idx - 1;
                int r         = idx;
                bool mark = (l >= 0) ? (_lvs_metric(l) < _lvs_metric(r)) : false;
                int cur       = mark ? l : r;
                _lvs_par(idx) = cur;
                if(_flag.total_size() == 0)
                    return;

                if(mark)
                {
                    _tks_rc(cur)      = idx;
                    _tks_range_y(cur) = idx;
                    atomicOr(&_tks_mark(cur), 0x00000002);
                }
                else
                {
                    _tks_lc(cur)      = idx;
                    _tks_range_x(cur) = idx;
                    atomicOr(&_tks_mark(cur), 0x00000001);
                }
                __threadfence();

                while(atomicAdd(&_flag(cur), 1) == 1)
                {
                    int      chl = _tks_lc(cur);
                    int      chr = _tks_rc(cur);
                    uint32_t m   = _tks_mark(cur);
                    if(m & 1)
                        _tks_box(cur) = _lvs_box(chl);
                    else
                        _tks_box(cur) = _tks_box(chl);
                    if(m & 2)
                        _tks_box(cur).extend(_lvs_box(chr));
                    else
                        _tks_box(cur).extend(_tks_box(chr));

                    IndexT l_bid  = (m & 1) ? _lvs_bid(chl) : _tks_bid(chl);
                    IndexT r_bid  = (m & 2) ? _lvs_bid(chr) : _tks_bid(chr);
                    IndexT l_cid  = (m & 1) ? _lvs_cid(chl) : _tks_cid(chl);
                    IndexT r_cid  = (m & 2) ? _lvs_cid(chr) : _tks_cid(chr);
                    _tks_bid(cur) = (l_bid == r_bid) ? l_bid : invalid;
                    _tks_cid(cur) = (l_cid == r_cid) ? l_cid : invalid;

                    _tks_mark(cur) &= 0x00000007;
                    l               = _tks_range_x(cur) - 1;
                    r               = _tks_range_y(cur);
                    _lvs_lca(l + 1) = cur;
                    _depths(l + 1)++;
                    mark = (l >= 0) ? (_lvs_metric(l) < _lvs_metric(r)) : false;
                    if(l + 1 == 0 && r == size - 1)
                    {
                        _tks_par(cur) = -1;
                        _tks_mark(cur) &= 0xFFFFFFFB;
                        break;
                    }

                    int par       = mark ? l : r;
                    _tks_par(cur) = par;
                    if(mark)
                    {
                        _tks_rc(par)      = cur;
                        _tks_range_y(par) = r;
                        atomicAnd(&_tks_mark(par), 0xFFFFFFFD);
                        _tks_mark(cur) |= 0x00000004;
                    }
                    else
                    {
                        _tks_lc(par)      = cur;
                        _tks_range_x(par) = l + 1;
                        atomicAnd(&_tks_mark(par), 0xFFFFFFFE);
                        _tks_mark(cur) &= 0xFFFFFFFB;
                    }
                    __threadfence();
                    cur = par;
                }
            });
}

inline void InfoStacklessBVH::Impl::calcIntNodeOrders(int size)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(size,
               [_tks_lc  = int_lc.viewer().name("tks_lc"),
                _lcas    = ext_lca.viewer().name("lcas"),
                _depths  = count.viewer().name("depths"),
                _offsets = offsetTable.viewer().name("offsets"),
                _tkMap   = tkMap.viewer().name("tkMap")] __device__(int idx)
               {
                   int node  = _lcas(idx);
                   int depth = _depths(idx);
                   int id    = _offsets(idx);
                   if(node != -1)
                   {
                       for(; depth--; node = _tks_lc(node))
                           _tkMap(node) = id++;
                   }
               });
}

inline void InfoStacklessBVH::Impl::updateBvhExtNodeLinks(int size)
{
    using namespace muda;
    if(flags.size() == 0)
        return;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(size,
               [_map  = tkMap.viewer().name("map"),
                _lcas = ext_lca.viewer().name("lcas"),
                _pars = ext_par.viewer().name("pars")] __device__(int idx)
               {
                   _pars(idx) = _map(_pars(idx));
                   int ori    = _lcas(idx);
                   _lcas(idx) = (ori != -1) ? (_map(ori) << 1) : (idx << 1 | 1);
               });
}

inline void InfoStacklessBVH::Impl::reorderNode(int int_size)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(int_size + 1,
               [int_size,
                _lvs_lca     = ext_lca.viewer().name("lvs_lca"),
                _lvs_par     = ext_par.viewer().name("lvs_par"),
                _lvs_box     = ext_aabb.viewer().name("lvs_box"),
                _lvs_bid     = ext_bid.viewer().name("lvs_bid"),
                _lvs_cid     = ext_cid.viewer().name("lvs_cid"),
                _tk_map      = tkMap.viewer().name("tk_map"),
                _int_lc      = int_lc.viewer().name("int_lc"),
                _int_rc      = int_rc.viewer().name("int_rc"),
                _int_par     = int_par.viewer().name("int_par"),
                _int_mark    = int_mark.viewer().name("int_mark"),
                _int_range_y = int_range_y.viewer().name("int_range_y"),
                _self_max_rank = self_max_rank.viewer().name("self_max_rank"),
                _refit_parent = refit_parent.viewer().name("refit_parent"),
                _refit_right = refit_right_child.viewer().name("refit_right"),
                _int_box     = int_aabb.viewer().name("int_box"),
                _int_bid     = int_bid.viewer().name("int_bid"),
                _int_cid     = int_cid.viewer().name("int_cid"),
                _nodes       = nodes.viewer().name("nodes")] __device__(int idx)
               {
                   Node leaf;
                   leaf.lc    = -1;
                   int escape = _lvs_lca(idx + 1);
                   if(escape == -1)
                       leaf.escape = -1;
                   else
                   {
                       int b_leaf = escape & 1;
                       escape >>= 1;
                       leaf.escape = escape + (b_leaf ? int_size : 0);
                   }
                   leaf.bound             = _lvs_box(idx);
                   leaf.bid               = _lvs_bid(idx);
                   leaf.cid               = _lvs_cid(idx);
                   _nodes(idx + int_size) = leaf;
                   if(int_size == 0)
                       _refit_parent(idx + int_size) = -1;
                   else
                       _refit_parent(idx + int_size) =
                           static_cast<int>(_lvs_par(idx));

                   if(idx >= int_size)
                       return;

                   Node     n;
                   int      new_id = _tk_map(idx);
                   uint32_t m      = _int_mark(idx);
                   _self_max_rank(new_id) = _int_range_y(idx);
                   n.lc = (m & 1) ? _int_lc(idx) + int_size : _tk_map(_int_lc(idx));
                   _refit_right(new_id) =
                       (m & 2) ? _int_rc(idx) + int_size : _tk_map(_int_rc(idx));
                   int old_parent = _int_par(idx);
                   _refit_parent(new_id) =
                       old_parent == -1 ? -1 : _tk_map(old_parent);
                   n.bound = _int_box(idx);
                   int ie  = _lvs_lca(_int_range_y(idx) + 1);
                   if(ie == -1)
                       n.escape = -1;
                   else
                   {
                       int b_leaf = ie & 1;
                       ie >>= 1;
                       n.escape = ie + (b_leaf ? int_size : 0);
                   }
                   n.bid          = _int_bid(idx);
                   n.cid          = _int_cid(idx);
                   _nodes(new_id) = n;
               });
}

inline void InfoStacklessBVH::Impl::propagateInformativeMetadata(int) {}

inline void InfoStacklessBVH::Impl::build(muda::CBufferView<AABB>   aabbs,
                                          muda::CBufferView<IndexT> _bids,
                                          muda::CBufferView<IndexT> _cids)
{
    objs          = aabbs;
    bids          = _bids;
    cids          = _cids;
    auto num_objs = aabbs.size();
    auto num_internal = num_objs > 0 ? num_objs - 1 : 0;
    self_max_rank.resize(num_internal);
    if(num_objs == 0)
        return;

    auto num_nodes    = num_objs * 2 - 1;
    mtcode.resize(num_objs);
    sorted_id.resize(num_objs);
    primMap.resize(num_objs);
    ext_aabb.resize(num_objs);
    ext_idx.resize(num_objs);
    ext_lca.resize(num_objs + 1);
    ext_par.resize(num_objs);
    ext_mark.resize(num_objs);
    ext_bid.resize(num_objs);
    ext_cid.resize(num_objs);
    metric.resize(num_objs);
    tkMap.resize(num_objs);
    offsetTable.resize(num_objs);
    count.resize(num_objs);
    flags.resize(num_internal);
    int_lc.resize(num_internal);
    int_rc.resize(num_internal);
    int_par.resize(num_internal);
    int_range_x.resize(num_internal);
    int_range_y.resize(num_internal);
    int_mark.resize(num_internal);
    int_aabb.resize(num_internal);
    int_bid.resize(num_internal);
    int_cid.resize(num_internal);
    nodes.resize(num_nodes);
    refit_parent.resize(num_nodes);
    refit_right_child.resize(num_internal);
    refit_arrivals.resize(num_internal);

    thrust::fill(flags.begin(), flags.end(), 0);
    thrust::fill(thrust::device, ext_mark.begin(), ext_mark.end(), 7);
    thrust::fill(thrust::device, ext_lca.begin(), ext_lca.end(), 0);
    thrust::fill(thrust::device, ext_par.begin(), ext_par.end(), 0);
    thrust::fill(thrust::device, int_bid.begin(), int_bid.end(), static_cast<IndexT>(-1));
    thrust::fill(thrust::device, int_cid.begin(), int_cid.end(), static_cast<IndexT>(-1));

    calcMaxBVFromBox(aabbs, scene_box.view());
    calcMCsFromBox(aabbs, scene_box.view(), mtcode.view());
    auto null_stream = thrust::cuda::par_nosync.on(nullptr);
    thrust::sequence(null_stream, sorted_id.begin(), sorted_id.end());
    thrust::sort_by_key(null_stream, mtcode.begin(), mtcode.end(), sorted_id.begin());
    calcInverseMapping();
    buildPrimitivesFromBox(aabbs);
    calcExtNodeSplitMetrics();
    buildIntNodes(num_objs);
    thrust::exclusive_scan(null_stream, count.begin(), count.end(), offsetTable.begin());
    calcIntNodeOrders(num_objs);
    thrust::fill(null_stream, ext_lca.begin() + num_objs, ext_lca.begin() + num_objs + 1, -1);
    updateBvhExtNodeLinks(num_objs);
    reorderNode(num_internal);
}

inline bool InfoStacklessBVH::Impl::refit(muda::CBufferView<AABB>   aabbs,
                                          muda::CBufferView<IndexT> _bids,
                                          muda::CBufferView<IndexT> _cids)
{
    using namespace muda;
    constexpr IndexT invalid = static_cast<IndexT>(-1);

    auto num_objs = aabbs.size();
    if(num_objs != objs.size())
        return false;

    bool has_info = _bids.size() == num_objs && _cids.size() == num_objs;
    if(!has_info)
        return false;

    if(num_objs == 0)
    {
        objs = aabbs;
        bids = _bids;
        cids = _cids;
        return true;
    }

    auto num_internal = num_objs - 1;
    auto num_nodes    = num_objs * 2 - 1;
    if(nodes.size() != num_nodes || ext_idx.size() != num_objs
       || ext_aabb.size() != num_objs || ext_bid.size() != num_objs
       || ext_cid.size() != num_objs || refit_parent.size() != num_nodes
       || refit_right_child.size() != num_internal
       || refit_arrivals.size() != num_internal)
        return false;

    objs = aabbs;
    bids = _bids;
    cids = _cids;

    auto null_stream = thrust::cuda::par_nosync.on(nullptr);
    thrust::fill(null_stream, refit_arrivals.begin(), refit_arrivals.end(), 0);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(num_objs,
               [int_size = static_cast<int>(num_internal),
                _aabbs   = aabbs.viewer().name("aabbs"),
                _bids    = _bids.viewer().name("bids"),
                _cids    = _cids.viewer().name("cids"),
                _lvs_idx = ext_idx.viewer().name("lvs_idx"),
                _lvs_box = ext_aabb.viewer().name("lvs_box"),
                _lvs_bid = ext_bid.viewer().name("lvs_bid"),
                _lvs_cid = ext_cid.viewer().name("lvs_cid"),
                _nodes   = nodes.viewer().name("nodes"),
                _parent  = refit_parent.viewer().name("parent"),
                _right   = refit_right_child.viewer().name("right"),
                _arrivals = refit_arrivals.viewer().name("arrivals")] __device__(int rank)
               {
                   int raw_id  = _lvs_idx(rank);
                   int leaf_id = int_size + rank;

                   Node leaf  = _nodes(leaf_id);
                   leaf.bound = _aabbs(raw_id);
                   leaf.bid   = _bids(raw_id);
                   leaf.cid   = _cids(raw_id);
                   _nodes(leaf_id) = leaf;
                   _lvs_box(rank)  = leaf.bound;
                   _lvs_bid(rank)  = leaf.bid;
                   _lvs_cid(rank)  = leaf.cid;

                   __threadfence();
                   int parent = _parent(leaf_id);
                   while(parent != -1)
                   {
                       // The first completed child stops here. The second child
                       // observes both child nodes, updates the parent, and
                       // continues toward the root.
                       if(atomicAdd(&_arrivals(parent), 1) == 0)
                           break;

                       __threadfence();
                       Node node  = _nodes(parent);
                       Node left  = _nodes(node.lc);
                       Node right = _nodes(_right(parent));
                       node.bound = left.bound;
                       node.bound.extend(right.bound);
                       node.bid = left.bid == right.bid ? left.bid : invalid;
                       node.cid = left.cid == right.cid ? left.cid : invalid;
                       _nodes(parent) = node;

                       __threadfence();
                       parent = _parent(parent);
                   }
               });

    return true;
}

// ---------------------------------------------------------------------------
// OPTIMIZED: stacklessSelf
//   Pre-loads query_bid and query_cid for each thread into shared memory
//   before the traversal loop. The node_cull functor receives a NodePredInfo
//   with query_bid/query_cid already filled from SMem — no global reads
//   inside the hot loop.
// ---------------------------------------------------------------------------
template <typename NodeCull, typename PairPred>
void InfoStacklessBVH::Impl::stacklessSelf(NodeCull                   node_cull,
                                           PairPred                   pair_pred,
                                           muda::VarView<int>         cpNum,
                                           muda::BufferView<Vector2i> buffer)
{
    using namespace muda;
    auto num_query = static_cast<int>(ext_aabb.size());
    auto num_objs  = num_query;
    auto grid      = (num_query + K_SELF_THREADS - 1) / K_SELF_THREADS;

    constexpr IndexT invalid = static_cast<IndexT>(-1);
    bool has_info = bids.size() == (size_t)num_objs && cids.size() == (size_t)num_objs;

    Launch(grid, K_SELF_THREADS)
        .apply(
            [Size     = num_query,
             _box     = objs.viewer().name("box"),
             intSize  = num_objs - 1,
             numObjs  = num_objs,
             _lvs_idx = ext_idx.viewer().name("lvs_idx"),
             _nodes   = nodes.viewer().name("nodes"),
             _self_max_rank = self_max_rank.viewer().name("self_max_rank"),
             _bids    = bids.viewer().name("bids"),  // needed for SMem pre-load
             _cids    = cids.viewer().name("cids"),  // needed for SMem pre-load
             has_info,
             resCounter = cpNum.viewer().name("cp_num"),
             res        = buffer.viewer().name("res"),
             node_cull,
             pair_pred] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx    = -1;
                AABB bv;
                if(active)
                {
                    idx = _lvs_idx(tid);
                    bv  = _box(idx);
                }

                // -----------------------------------------------------------------
                // SMem: pre-load query bid/cid once per thread, before hot loop.
                // Shared memory layout scales with the Self CTA and queue.
                // The queue keeps four candidate-pair slots per thread.
                //   s_qbid[K_SELF_THREADS]
                //   s_qcid[K_SELF_THREADS]
                //   shared_res[K_SELF_MAX_RES_PER_BLOCK]
                //   shared_counter, shared_global_idx
                // Total shared memory scales with the selected configuration.
                // -----------------------------------------------------------------
                __shared__ IndexT s_qbid[K_SELF_THREADS];
                __shared__ IndexT s_qcid[K_SELF_THREADS];

                s_qbid[threadIdx.x] = (active && has_info) ? _bids(idx) : invalid;
                s_qcid[threadIdx.x] = (active && has_info) ? _cids(idx) : invalid;

                __shared__ int2 shared_res[K_SELF_MAX_RES_PER_BLOCK];
                __shared__ int  shared_counter;
                __shared__ int  shared_global_idx;
                if(threadIdx.x == 0)
                    shared_counter = 0;

                int       st       = 0;
                const int max_iter = numObjs * 2;
                while(true)
                {
                    // First __syncthreads also ensures SMem pre-loads above are
                    // visible to all threads in the block (though each thread
                    // only reads its own slot: s_qbid[threadIdx.x]).
                    __syncthreads();
                    if(active)
                    {
                        int inner_i = 0;
                        for(; inner_i < max_iter; ++inner_i)
                        {
                            if(st == -1)
                                break;
                            // The existing leaf gate only accepts ranks above tid.
                            // Skip an internal subtree when its maximum rank cannot pass.
                            if(st < intSize && _self_max_rank(st) <= tid)
                            {
                                st = _nodes(st).escape;
                                continue;
                            }
                            auto node = _nodes(st);
                            if(!node.bound.intersects(bv))
                            {
                                st = node.escape;
                                continue;
                            }
                            if(!node_cull(NodePredInfo{
                                   idx, s_qbid[threadIdx.x], s_qcid[threadIdx.x], node.bid, node.cid}))
                            {
                                st = node.escape;
                                continue;
                            }
                            if(node.lc == -1)
                            {
                                if(tid < st - intSize)
                                {
                                    int  leaf_raw = _lvs_idx(st - intSize);
                                    bool q_first  = (idx < leaf_raw);
                                    auto pair     = ordered_pair(idx, leaf_raw);
                                    LeafPredInfo leaf_info{
                                        pair.x,
                                        pair.y,
                                        q_first ? s_qbid[threadIdx.x] : node.bid,
                                        q_first ? s_qcid[threadIdx.x] : node.cid,
                                        q_first ? node.bid : s_qbid[threadIdx.x],
                                        q_first ? node.cid : s_qcid[threadIdx.x]};
                                    if(pair_pred(leaf_info))
                                    {
                                        int sidx = atomicAdd(&shared_counter, 1);
                                        if(sidx >= K_SELF_MAX_RES_PER_BLOCK)
                                            break;
                                        shared_res[sidx] = pair;
                                    }
                                }
                                st = node.escape;
                            }
                            else
                                st = node.lc;
                        }
                        MUDA_ASSERT(inner_i < max_iter, "Exceeded max stackless iteration");
                    }
                    __syncthreads();
                    int total = min(shared_counter, K_SELF_MAX_RES_PER_BLOCK);
                    if(threadIdx.x == 0)
                        shared_global_idx = atomicAdd(resCounter.data(), total);
                    __syncthreads();
                    int gidx = shared_global_idx;
                    if(threadIdx.x == 0)
                        shared_counter = 0;
                    bool done = total < K_SELF_MAX_RES_PER_BLOCK;
                    safe_copy_to(shared_res,
                                 total,
                                 res.data(),
                                 gidx,
                                 static_cast<int>(res.total_size()));
                    if(done)
                        break;
                }
            });
}

// ---------------------------------------------------------------------------
// OPTIMIZED: stacklessOther
//   Takes explicit query_bids / query_cids buffers and pre-loads them into
//   shared memory. node_cull receives a NodePredInfo with query_bid/query_cid
//   already filled from SMem — no global reads inside the hot loop.
// ---------------------------------------------------------------------------
template <typename NodeCull, typename PairPred>
void InfoStacklessBVH::Impl::stacklessOther(NodeCull                node_cull,
                                            PairPred                pair_pred,
                                            muda::CBufferView<AABB> query_aabbs,
                                            muda::CBufferView<IndexT> query_bids,
                                            muda::CBufferView<IndexT> query_cids,
                                            muda::CBufferView<int> query_sorted_id,
                                            muda::VarView<int>         cpNum,
                                            muda::BufferView<Vector2i> buffer)
{
    using namespace muda;
    auto num_query = static_cast<int>(query_aabbs.size());
    auto num_objs  = static_cast<int>(ext_aabb.size());
    auto grid      = (num_query + K_OTHER_THREADS - 1) / K_OTHER_THREADS;

    constexpr IndexT invalid   = static_cast<IndexT>(-1);
    bool             qhas_info = query_bids.size() == (size_t)num_query
                                 && query_cids.size() == (size_t)num_query;

    Launch(grid, K_OTHER_THREADS)
        .apply(
            [Size      = num_query,
             _box      = query_aabbs.viewer().name("qbox"),
             sortedIdx = query_sorted_id.viewer().name("sortedIdx"),
             intSize   = num_objs - 1,
             numObjs   = num_objs,
             _lvs_idx  = ext_idx.viewer().name("lvs_idx"),
             _nodes    = nodes.viewer().name("nodes"),
             _qbids = query_bids.viewer().name("qbids"),  // for SMem pre-load
             _qcids = query_cids.viewer().name("qcids"),  // for SMem pre-load
             qhas_info,
             resCounter = cpNum.viewer().name("cp_num"),
             res        = buffer.viewer().name("res"),
             node_cull,
             pair_pred] __device__()
            {
                int  tid    = blockIdx.x * blockDim.x + threadIdx.x;
                bool active = tid < Size;
                int  idx    = -1;
                AABB bv;
                if(active)
                {
                    idx = sortedIdx(tid);
                    bv  = _box(idx);
                }

                // -----------------------------------------------------------------
                // SMem: pre-load per-query bid/cid before the traversal loop.
                // -----------------------------------------------------------------
                __shared__ IndexT s_qbid[K_OTHER_THREADS];
                __shared__ IndexT s_qcid[K_OTHER_THREADS];

                s_qbid[threadIdx.x] = (active && qhas_info) ? _qbids(idx) : invalid;
                s_qcid[threadIdx.x] = (active && qhas_info) ? _qcids(idx) : invalid;

                __shared__ int2 shared_res[K_OTHER_MAX_RES_PER_BLOCK];
                __shared__ int  shared_counter;
                __shared__ int  shared_global_idx;
                if(threadIdx.x == 0)
                    shared_counter = 0;

                int       st       = 0;
                const int max_iter = numObjs * 2;
                while(true)
                {
                    __syncthreads();
                    if(active)
                    {
                        int inner_i = 0;
                        for(; inner_i < max_iter; ++inner_i)
                        {
                            if(st == -1)
                                break;
                            auto node = _nodes(st);
                            if(!node.bound.intersects(bv))
                            {
                                st = node.escape;
                                continue;
                            }
                            if(!node_cull(NodePredInfo{
                                   idx, s_qbid[threadIdx.x], s_qcid[threadIdx.x], node.bid, node.cid}))
                            {
                                st = node.escape;
                                continue;
                            }
                            if(node.lc == -1)
                            {
                                auto pair = int2{idx, _lvs_idx(st - intSize)};
                                // query side: SMem pre-loaded; leaf side: node.bid/cid
                                LeafPredInfo leaf_info{pair.x,
                                                       pair.y,
                                                       s_qbid[threadIdx.x],
                                                       s_qcid[threadIdx.x],
                                                       node.bid,
                                                       node.cid};
                                if(pair_pred(leaf_info))
                                {
                                    int sidx = atomicAdd(&shared_counter, 1);
                                    if(sidx >= K_OTHER_MAX_RES_PER_BLOCK)
                                        break;
                                    shared_res[sidx] = pair;
                                }
                                st = node.escape;
                            }
                            else
                                st = node.lc;
                        }
                        MUDA_ASSERT(inner_i < max_iter, "Exceeded max stackless iteration");
                    }

                    __syncthreads();
                    int total = min(shared_counter, K_OTHER_MAX_RES_PER_BLOCK);
                    if(threadIdx.x == 0)
                        shared_global_idx = atomicAdd(resCounter.data(), total);
                    __syncthreads();
                    int gidx = shared_global_idx;
                    if(threadIdx.x == 0)
                        shared_counter = 0;
                    __syncthreads();
                    bool done = total < K_OTHER_MAX_RES_PER_BLOCK;
                    safe_copy_to(shared_res,
                                 total,
                                 res.data(),
                                 gidx,
                                 static_cast<int>(res.total_size()));
                    if(done)
                        break;
                }
            });
}

// ---------------------------------------------------------------------------
// Public API — same signatures as InfoStacklessBVH
// ---------------------------------------------------------------------------

inline InfoStacklessBVH::InfoStacklessBVH(muda::Stream& stream) noexcept
{
    (void)stream;
}

inline void InfoStacklessBVH::QueryBuffer::build(muda::CBufferView<AABB> aabbs,
                                                 bool reuse_order)
{
    // Morton order is only a traversal scheduling choice. It is not part of
    // the pair predicate. During Line Search the query primitive identities
    // and count are unchanged, so the order prepared by the preceding DCD
    // pass remains a valid permutation even when the swept boxes move.
    if(reuse_order && m_querySortedId.size() == aabbs.size())
        return;

    m_queryMtCode.resize(aabbs.size());
    m_querySortedId.resize(aabbs.size());
    Impl::calcMaxBVFromBox(aabbs, m_querySceneBox);
    Impl::calcMCsFromBox(aabbs, m_querySceneBox, m_queryMtCode.view());
    auto null_stream = thrust::cuda::par_nosync.on(nullptr);
    auto n           = static_cast<int>(aabbs.size());
    auto d_codes     = m_queryMtCode.data();
    auto d_ids       = m_querySortedId.data();
    thrust::sequence(null_stream, d_ids, d_ids + n);
    thrust::sort_by_key(null_stream, d_codes, d_codes + n, d_ids);
}

inline void InfoStacklessBVH::build(muda::CBufferView<AABB>   aabbs,
                                    muda::CBufferView<IndexT> BIDs,
                                    muda::CBufferView<IndexT> CIDs)
{
    m_aabbs = aabbs;
    m_BIDs  = BIDs;
    m_CIDs  = CIDs;
    UIPC_ASSERT(m_aabbs.size() == m_BIDs.size(),
                "AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                m_aabbs.size(),
                m_BIDs.size());
    UIPC_ASSERT(m_aabbs.size() == m_CIDs.size(),
                "AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                m_aabbs.size(),
                m_CIDs.size());
    m_impl.build(aabbs, BIDs, CIDs);
}

inline void InfoStacklessBVH::build(muda::CBufferView<AABB> aabbs)
{
    m_aabbs = aabbs;
    m_BIDs  = {};
    m_CIDs  = {};
    m_impl.build(aabbs, {}, {});
}

inline bool InfoStacklessBVH::refit(muda::CBufferView<AABB>   aabbs,
                                    muda::CBufferView<IndexT> BIDs,
                                    muda::CBufferView<IndexT> CIDs)
{
    if(aabbs.size() != m_aabbs.size() || aabbs.size() != BIDs.size()
       || aabbs.size() != CIDs.size())
        return false;

    if(!m_impl.refit(aabbs, BIDs, CIDs))
        return false;

    m_aabbs = aabbs;
    m_BIDs  = BIDs;
    m_CIDs  = CIDs;
    return true;
}

// detect() with NodePred / LeafPred: wrapper passes pre-loaded bid/cid to NodePredInfo
template <typename NodePred, typename LeafPred>
inline void InfoStacklessBVH::detect(muda::CBuffer2DView<IndexT> cmts,
                                     NodePred                    np,
                                     LeafPred                    lp,
                                     QueryBuffer&                qbuffer)
{
    if(m_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }
    UIPC_ASSERT(m_aabbs.size() == m_BIDs.size(),
                "AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                m_aabbs.size(),
                m_BIDs.size());
    UIPC_ASSERT(m_aabbs.size() == m_CIDs.size(),
                "AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                m_aabbs.size(),
                m_CIDs.size());

    using namespace muda;
    auto do_query = [&]
    {
        BufferLaunch().fill(qbuffer.m_cpNum.view(), 0);
        m_impl.stacklessSelf(np, lp, qbuffer.m_cpNum.view(), qbuffer.m_pairs.view());
    };

    do_query();
    int h_cp_num = qbuffer.m_cpNum;
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        do_query();
    }
    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
}

// query() with NodePred / LeafPred: passes query BIDs/CIDs to stacklessOther for SMem pre-load
template <typename NodePred, typename LeafPred>
inline void InfoStacklessBVH::query(muda::CBufferView<AABB>     query_aabbs,
                                    muda::CBufferView<IndexT>   query_BIDs,
                                    muda::CBufferView<IndexT>   query_CIDs,
                                    muda::CBuffer2DView<IndexT> cmts,
                                    NodePred                    np,
                                    LeafPred                    lp,
                                    QueryBuffer&                qbuffer,
                                    bool                        reuse_query_order)
{
    if(m_aabbs.size() == 0 || query_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }
    UIPC_ASSERT(query_aabbs.size() == query_BIDs.size(),
                "Query AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                query_aabbs.size(),
                query_BIDs.size());
    UIPC_ASSERT(query_aabbs.size() == query_CIDs.size(),
                "Query AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                query_aabbs.size(),
                query_CIDs.size());

    using namespace muda;
    qbuffer.build(query_aabbs, reuse_query_order);
    auto do_query = [&]
    {
        BufferLaunch().fill(qbuffer.m_cpNum.view(), 0);
        m_impl.stacklessOther(np,
                              lp,
                              query_aabbs,
                              query_BIDs,
                              query_CIDs,
                              qbuffer.m_querySortedId.view(),
                              qbuffer.m_cpNum.view(),
                              qbuffer.m_pairs.view());
    };
    do_query();
    int h_cp_num = qbuffer.m_cpNum;
    if(h_cp_num > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.resize(h_cp_num * m_impl.config.reserve_ratio);
        do_query();
    }
    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = h_cp_num;
}

}  // namespace uipc::backend::cuda
