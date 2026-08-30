#pragma once
#include <type_define.h>
#include <muda/buffer/device_var.h>
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/device_doublet_vector.h>
#include <muda/ext/linear_system/device_bcoo_vector.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>
#include <muda/ext/linear_system/device_bcoo_matrix.h>
#include <muda/ext/linear_system/device_bsr_matrix.h>

namespace uipc::backend::cuda
{
struct MatrixConverterIntPair
{
    int x;
    int y;
};

struct MatrixConverterRadixKeyConfig
{
    int      col_bits;
    int      end_bit;
    uint64_t col_mask;
    bool     compact;
};

constexpr int matrix_converter_index_bits(int extent)
{
    auto value = extent > 1 ? static_cast<uint32_t>(extent - 1) : uint32_t{0};
    int  bits  = 1;
    while(value >>= 1)
        ++bits;
    return bits;
}

constexpr MatrixConverterRadixKeyConfig matrix_converter_radix_key_config(int rows,
                                                                           int cols)
{
    if(rows <= 0 || cols <= 0)
        return {32, 64, 0xFFFFFFFFull, false};

    const int col_bits = matrix_converter_index_bits(cols);
    const int end_bit  = col_bits + matrix_converter_index_bits(rows);
    if(col_bits >= 64 || end_bit > 64)
        return {32, 64, 0xFFFFFFFFull, false};

    return {col_bits, end_bit, (uint64_t{1} << col_bits) - uint64_t{1}, true};
}

inline UIPC_GENERIC constexpr uint64_t matrix_converter_pack_radix_key(
    int row, int col, const MatrixConverterRadixKeyConfig& config)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << config.col_bits)
           | static_cast<uint64_t>(static_cast<uint32_t>(col));
}

inline UIPC_GENERIC constexpr MatrixConverterIntPair matrix_converter_unpack_radix_key(
    uint64_t key, const MatrixConverterRadixKeyConfig& config)
{
    return {static_cast<int>(key >> config.col_bits),
            static_cast<int>(key & config.col_mask)};
}

constexpr bool operator==(const MatrixConverterIntPair& l, const MatrixConverterIntPair& r)
{
    return l.x == r.x && l.y == r.y;
}

template <typename T, int N>
class MatrixConverter
{
    using BlockMatrix   = muda::DeviceTripletMatrix<T, N>::ValueT;
    using SegmentVector = muda::DeviceDoubletVector<T, N>::ValueT;

    Float m_reserve_ratio = 1.5;

    muda::DeviceBuffer<int> col_counts_per_row;
    muda::DeviceBuffer<int> unique_indices;
    muda::DeviceBuffer<int> unique_counts;
    muda::DeviceVar<int>    count;

    muda::DeviceBuffer<int> sort_index_input;
    muda::DeviceBuffer<int> sort_index;

    muda::DeviceBuffer<int> offsets;

    muda::DeviceBuffer<MatrixConverterIntPair> ij_pairs;
    muda::DeviceBuffer<MatrixConverterIntPair> unique_ij_pairs;

    muda::DeviceBuffer<uint64_t> ij_hash_input;
    muda::DeviceBuffer<uint64_t> ij_hash;

    muda::DeviceBuffer<BlockMatrix> blocks_sorted;
    muda::DeviceBuffer<BlockMatrix> diag_blocks;


    muda::DeviceBuffer<int>           indices_sorted;
    muda::DeviceBuffer<SegmentVector> segments_sorted;


    muda::DeviceBuffer<int> sorted_partition_input;
    muda::DeviceBuffer<int> sorted_partition_output;

  public:
    void  reserve_ratio(Float ratio) { m_reserve_ratio = ratio; }
    Float reserve_ratio() const { return m_reserve_ratio; }


    // Triplet -> BCOO
    void convert(const muda::DeviceTripletMatrix<T, N>& from,
                 muda::DeviceBCOOMatrix<T, N>&          to);

    void _radix_sort_indices_and_blocks(const muda::DeviceTripletMatrix<T, N>& from,
                                        muda::DeviceBCOOMatrix<T, N>& to);

    void _radix_sort_indices_and_blocks(muda::DeviceBCOOMatrix<T, N>& to);

    void _make_unique_indices(const muda::DeviceTripletMatrix<T, N>& from,
                              muda::DeviceBCOOMatrix<T, N>&          to);

    void _make_unique_block_warp_reduction(const muda::DeviceTripletMatrix<T, N>& from,
                                           muda::DeviceBCOOMatrix<T, N>& to);

    // BCOO -> BSR
    void convert(const muda::DeviceBCOOMatrix<T, N>& from,
                 muda::DeviceBSRMatrix<T, N>&        to);

    void _calculate_block_offsets(const muda::DeviceBCOOMatrix<T, N>& from,
                                  muda::DeviceBSRMatrix<T, N>&        to);


    // Doublet -> BCOO
    void convert(const muda::DeviceDoubletVector<T, N>& from,
                 muda::DeviceBCOOVector<T, N>&          to);

    void _radix_sort_indices_and_segments(const muda::DeviceDoubletVector<T, N>& from,
                                          muda::DeviceBCOOVector<T, N>& to);

    void _make_unique_indices(const muda::DeviceDoubletVector<T, N>& from,
                              muda::DeviceBCOOVector<T, N>&          to);

    void _make_unique_segment_warp_reduction(const muda::DeviceDoubletVector<T, N>& from,
                                             muda::DeviceBCOOVector<T, N>& to);


    template <typename U>
    void loose_resize(muda::DeviceBuffer<U>& buf, size_t new_size)
    {
        if(buf.capacity() < new_size)
            buf.reserve(new_size * m_reserve_ratio);
        buf.resize(new_size);
    }

    void ge2sym(muda::DeviceBCOOMatrix<T, N>& to);

    void ge2sym(muda::DeviceTripletMatrix<T, N>& to);

    void sym2ge(const muda::DeviceBCOOMatrix<T, N>& from,
                muda::DeviceBCOOMatrix<T, N>&       to);
};
}  // namespace uipc::backend::cuda

#include "details/matrix_converter.inl"
