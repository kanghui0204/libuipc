#pragma once

#include <type_define.h>
#include <utils/fixed_bank_soa_evd.h>

namespace uipc::backend::cuda
{
namespace four_vertex_translation_free_spd_detail
{
    // Three non-translation columns of the normalized 4x4 Hadamard basis.
    // Keeping the signs explicit avoids materializing a 12x9 matrix.
    UIPC_DEVICE inline void four_to_relative(Float  x0,
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

    UIPC_DEVICE inline void relative_to_four(Float  r0,
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
 * Remove the three known common-translation zero modes from a four-vertex
 * Hessian, solve the remaining 9D PSD projection, and lift its eigenvectors
 * back to 12D.  The caller must guarantee common-translation invariance.
 *
 * H is an in-place 12x12 fixed-bank workspace.  On return its first nine
 * columns hold lifted eigenvectors, its final three columns and eigenvalues
 * are zero, so the normal triplet writer can keep its existing 12D contract.
 */
template <int LanePitch, typename Workspace>
UIPC_DEVICE bool selfadjoint_evd_four_vertex_translation_free_fixed_bank(
    Workspace& H, Vector12& eigen_values)
{
    using namespace four_vertex_translation_free_spd_detail;

    // Form H * Q in the first nine logical 12D columns.
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

    // Form Q^T * H * Q and compact to the 9-stride workspace.  A destination
    // column always precedes any source column that has not been consumed.
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

    // Lift the nine eigenvectors back to four vertices.  Reverse column order
    // keeps expanded 12-stride destinations from overwriting compact sources.
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
