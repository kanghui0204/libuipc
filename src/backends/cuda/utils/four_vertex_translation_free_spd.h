#pragma once

#include <type_define.h>
#include <utils/fixed_bank_soa_evd.h>
#include <utils/make_spd.h>

namespace uipc::backend::cuda
{
namespace four_vertex_translation_free_spd_detail
{
    // The three non-translation columns of the normalized 4x4 Hadamard
    // matrix are
    //
    //   1/2 [ 1  1  1]
    //       [ 1 -1 -1]
    //       [-1  1 -1]
    //       [-1 -1  1].
    //
    // Keeping the signs explicit avoids materializing the 12x9 basis Q or
    // evaluating Q.transpose() * H * Q through generic matrix products.
    UIPC_GENERIC inline void four_to_relative(Float  x0,
                                               Float  x1,
                                               Float  x2,
                                               Float  x3,
                                               Float& r0,
                                               Float& r1,
                                               Float& r2)
    {
        r0 = x0 + x1 - x2 - x3;
        r1 = x0 - x1 + x2 - x3;
        r2 = x0 - x1 - x2 + x3;
    }

    UIPC_GENERIC inline void relative_to_four(Float  r0,
                                               Float  r1,
                                               Float  r2,
                                               Float& x0,
                                               Float& x1,
                                               Float& x2,
                                               Float& x3)
    {
        x0 = r0 + r1 + r2;
        x1 = r0 - r1 - r2;
        x2 = -r0 + r1 - r2;
        x3 = -r0 - r1 + r2;
    }
}  // namespace four_vertex_translation_free_spd_detail

/**
 * Project a four-vertex 12x12 Hessian to PSD after removing its three common
 * translation modes.
 *
 * The input must be invariant to translating all four 3D vertices together.
 * For such an H, H = Q (Q^T H Q) Q^T, where Q is the 12x9 orthonormal
 * relative-coordinate basis described above.  Thus clamping the eigenvalues
 * of the reduced 9x9 matrix is mathematically equivalent to a full 12x12 PSD
 * projection, with the three translation eigenvalues fixed to zero.
 *
 * This helper intentionally does not test the precondition.  Its production
 * caller must only use it for four-vertex translation-invariant energies.
 */
UIPC_GENERIC inline void make_spd_four_vertex_translation_free(Matrix12x12& H)
{
    using namespace four_vertex_translation_free_spd_detail;

    Matrix9x9 reduced;

    // reduced = Q^T H Q.  Each spatial (axis_a, axis_b) slice is a 4x4
    // vertex matrix.  Apply the unnormalized Hadamard signs first to its
    // columns and then to its rows; the two 1/2 normalizations give 1/4.
#pragma unroll
    for(int axis_a = 0; axis_a < 3; ++axis_a)
    {
#pragma unroll
        for(int axis_b = 0; axis_b < 3; ++axis_b)
        {
            Float column_modes[4][3];
#pragma unroll
            for(int vertex_a = 0; vertex_a < 4; ++vertex_a)
            {
                four_to_relative(H(3 * vertex_a + axis_a, axis_b),
                                 H(3 * vertex_a + axis_a, 3 + axis_b),
                                 H(3 * vertex_a + axis_a, 6 + axis_b),
                                 H(3 * vertex_a + axis_a, 9 + axis_b),
                                 column_modes[vertex_a][0],
                                 column_modes[vertex_a][1],
                                 column_modes[vertex_a][2]);
            }

#pragma unroll
            for(int mode_b = 0; mode_b < 3; ++mode_b)
            {
                Float value0;
                Float value1;
                Float value2;
                four_to_relative(column_modes[0][mode_b],
                                 column_modes[1][mode_b],
                                 column_modes[2][mode_b],
                                 column_modes[3][mode_b],
                                 value0,
                                 value1,
                                 value2);
                reduced(axis_a, 3 * mode_b + axis_b) = 0.25 * value0;
                reduced(3 + axis_a, 3 * mode_b + axis_b) = 0.25 * value1;
                reduced(6 + axis_a, 3 * mode_b + axis_b) = 0.25 * value2;
            }
        }
    }

    make_spd<9>(reduced);

    // H = Q reduced Q^T.  Apply the same signs in reverse.  Again, the two
    // normalized Hadamard factors contribute the final factor of 1/4.
#pragma unroll
    for(int axis_a = 0; axis_a < 3; ++axis_a)
    {
#pragma unroll
        for(int axis_b = 0; axis_b < 3; ++axis_b)
        {
            Float vertex_to_modes[4][3];
#pragma unroll
            for(int mode_b = 0; mode_b < 3; ++mode_b)
            {
                relative_to_four(reduced(axis_a, 3 * mode_b + axis_b),
                                 reduced(3 + axis_a, 3 * mode_b + axis_b),
                                 reduced(6 + axis_a, 3 * mode_b + axis_b),
                                 vertex_to_modes[0][mode_b],
                                 vertex_to_modes[1][mode_b],
                                 vertex_to_modes[2][mode_b],
                                 vertex_to_modes[3][mode_b]);
            }

#pragma unroll
            for(int vertex_a = 0; vertex_a < 4; ++vertex_a)
            {
                Float value0;
                Float value1;
                Float value2;
                Float value3;
                relative_to_four(vertex_to_modes[vertex_a][0],
                                 vertex_to_modes[vertex_a][1],
                                 vertex_to_modes[vertex_a][2],
                                 value0,
                                 value1,
                                 value2,
                                 value3);
                H(3 * vertex_a + axis_a, axis_b)     = 0.25 * value0;
                H(3 * vertex_a + axis_a, 3 + axis_b) = 0.25 * value1;
                H(3 * vertex_a + axis_a, 6 + axis_b) = 0.25 * value2;
                H(3 * vertex_a + axis_a, 9 + axis_b) = 0.25 * value3;
            }
        }
    }
}

template <int LanePitch, typename Workspace>
UIPC_DEVICE bool selfadjoint_evd_four_vertex_translation_free_fixed_bank(
    Workspace& H, Vector12& eigen_values)
{
    using namespace four_vertex_translation_free_spd_detail;

    // First form H * Q in the first nine logical columns.  Loading one full
    // row before overwriting it keeps the transform in the existing shared
    // workspace without adding a second 12x12 buffer.
#pragma unroll
    for(int row = 0; row < 12; ++row)
    {
        Float input[12];
#pragma unroll
        for(int col = 0; col < 12; ++col)
            input[col] = H(row, col);
#pragma unroll
        for(int axis = 0; axis < 3; ++axis)
        {
            Float r0;
            Float r1;
            Float r2;
            four_to_relative(input[axis],
                             input[3 + axis],
                             input[6 + axis],
                             input[9 + axis],
                             r0,
                             r1,
                             r2);
            H(row, axis)     = 0.5 * r0;
            H(row, 3 + axis) = 0.5 * r1;
            H(row, 6 + axis) = 0.5 * r2;
        }
    }

    FixedBankSoAMap<9, LanePitch> reduced(H.data());

    // Then form Q^T * (H * Q).  Compacting each completed column is safe:
    // its destination precedes every source column that has not been read.
#pragma unroll
    for(int col = 0; col < 9; ++col)
    {
        Float input[12];
#pragma unroll
        for(int row = 0; row < 12; ++row)
            input[row] = H(row, col);
#pragma unroll
        for(int axis = 0; axis < 3; ++axis)
        {
            Float r0;
            Float r1;
            Float r2;
            four_to_relative(input[axis],
                             input[3 + axis],
                             input[6 + axis],
                             input[9 + axis],
                             r0,
                             r1,
                             r2);
            reduced(axis, col)     = 0.5 * r0;
            reduced(3 + axis, col) = 0.5 * r1;
            reduced(6 + axis, col) = 0.5 * r2;
        }
    }

    Vector9 reduced_eigen_values;
    const bool success =
        selfadjoint_evd_fixed_bank_shared<9>(reduced, reduced_eigen_values);

    eigen_values.setZero();
#pragma unroll
    for(int i = 0; i < 9; ++i)
        eigen_values(i) = reduced_eigen_values(i);

    // Lift the nine relative-coordinate eigenvectors back to 12D.  Expand
    // columns in reverse so their 12-stride destinations cannot overwrite
    // a compact 9-stride source column that is still needed.
#pragma unroll
    for(int col = 8; col >= 0; --col)
    {
        Float input[9];
#pragma unroll
        for(int row = 0; row < 9; ++row)
            input[row] = reduced(row, col);
#pragma unroll
        for(int axis = 0; axis < 3; ++axis)
        {
            Float x0;
            Float x1;
            Float x2;
            Float x3;
            relative_to_four(input[axis],
                             input[3 + axis],
                             input[6 + axis],
                             x0,
                             x1,
                             x2,
                             x3);
            H(axis, col)     = 0.5 * x0;
            H(3 + axis, col) = 0.5 * x1;
            H(6 + axis, col) = 0.5 * x2;
            H(9 + axis, col) = 0.5 * x3;
        }
    }

#pragma unroll
    for(int col = 9; col < 12; ++col)
#pragma unroll
        for(int row = 0; row < 12; ++row)
            H(row, col) = 0.0;

    return success;
}
}  // namespace uipc::backend::cuda
