#pragma once

#include <type_define.h>
#include <muda/ext/eigen/evd.h>
#include <cfloat>

namespace uipc::backend::cuda
{
// Experimental one-thread/instance EVD workspace. Logical matrix element e
// for lane l is stored at shared[e * LanePitch + l]. No thread cooperates
// with another thread; the layout only changes where each thread keeps its
// persistent Eigen workspace.
template <int N, int LanePitch>
using FixedBankSoAStride = Eigen::Stride<N * LanePitch, LanePitch>;

template <int N, int LanePitch>
using FixedBankSoAMap =
    Eigen::Map<Matrix<Float, N, N>, Eigen::Unaligned, FixedBankSoAStride<N, LanePitch>>;

namespace fixed_bank_soa_evd_detail
{
    template <int N, typename Workspace>
    UIPC_DEVICE void tridiagonalization_inplace_strided(Workspace& workspace,
                                                        Vector<Float, N - 1>& hcoeffs)
    {
#pragma unroll
        for(int step = 0; step < N - 1; ++step)
        {
            const int remaining = N - step - 1;
            Float     beta      = 0.0;
            Float     tau       = 0.0;
            auto      reflector = workspace.col(step).tail(remaining);
            reflector.makeHouseholderInPlace(tau, beta);
            workspace(step + 1, step) = 1.0;

            Vector<Float, N> work;
#pragma unroll
            for(int row = 0; row < N; ++row)
                work(row) = 0.0;

                // Eigen 3.4's self-adjoint lower matrix-vector product, expressed
                // through coefficients so a non-unit inner stride is respected.
#pragma unroll
            for(int row = 0; row < N; ++row)
            {
                if(row < remaining)
                {
                    Float sum = 0.0;
#pragma unroll
                    for(int col = 0; col < N; ++col)
                    {
                        if(col < remaining)
                        {
                            const Float a =
                                row >= col ?
                                    workspace(step + 1 + row, step + 1 + col) :
                                    workspace(step + 1 + col, step + 1 + row);
                            sum += a * reflector(col);
                        }
                    }
                    work(row) = tau * sum;
                }
            }

            Float dot = 0.0;
#pragma unroll
            for(int row = 0; row < N; ++row)
                if(row < remaining)
                    dot += work(row) * reflector(row);
            const Float correction = -0.5 * tau * dot;
#pragma unroll
            for(int row = 0; row < N; ++row)
                if(row < remaining)
                    work(row) += correction * reflector(row);

                    // Eigen 3.4 lower rank-2 update without the raw-pointer kernel that
                    // assumes innerStride()==1.
#pragma unroll
            for(int col = 0; col < N; ++col)
            {
                if(col < remaining)
                {
#pragma unroll
                    for(int row = 0; row < N; ++row)
                    {
                        if(row >= col && row < remaining)
                        {
                            workspace(step + 1 + row, step + 1 + col) -=
                                reflector(col) * work(row) + work(col) * reflector(row);
                        }
                    }
                }
            }

            workspace(step + 1, step) = beta;
            hcoeffs(step)             = tau;
        }
    }

    template <int N, typename Workspace>
    UIPC_DEVICE void tridiagonal_qr_step_strided(Vector<Float, N>&     diag,
                                                 Vector<Float, N - 1>& subdiag,
                                                 int                   start,
                                                 int                   end,
                                                 Workspace&            q)
    {
        Float td = (diag(end - 1) - diag(end)) * 0.5;
        Float e  = subdiag(end - 1);
        Float mu = diag(end);
        if(td == 0.0)
            mu -= Eigen::numext::abs(e);
        else if(e != 0.0)
        {
            const Float e2 = Eigen::numext::abs2(e);
            const Float h  = Eigen::numext::hypot(td, e);
            if(e2 == 0.0)
                mu -= e / ((td + (td > 0.0 ? h : -h)) / e);
            else
                mu -= e2 / (td + (td > 0.0 ? h : -h));
        }

        Float x = diag(start) - mu;
        Float z = subdiag(start);
        for(int k = start; k < end && z != 0.0; ++k)
        {
            Eigen::JacobiRotation<Float> rotation;
            rotation.makeGivens(x, z);

            const Float sdk = rotation.s() * diag(k) + rotation.c() * subdiag(k);
            const Float dkp1 = rotation.s() * subdiag(k) + rotation.c() * diag(k + 1);
            diag(k) =
                rotation.c() * (rotation.c() * diag(k) - rotation.s() * subdiag(k))
                - rotation.s() * (rotation.c() * subdiag(k) - rotation.s() * diag(k + 1));
            diag(k + 1) = rotation.s() * sdk + rotation.c() * dkp1;
            subdiag(k)  = rotation.c() * sdk - rotation.s() * dkp1;

            if(k > start)
                subdiag(k - 1) = rotation.c() * subdiag(k - 1) - rotation.s() * z;

            x = subdiag(k);
            if(k < end - 1)
            {
                z              = -rotation.s() * subdiag(k + 1);
                subdiag(k + 1) = rotation.c() * subdiag(k + 1);
            }
            // Unlike Eigen 3.4's raw q.data() re-Map, this applies the same
            // rotation directly to the strided Eigen expression.
            q.applyOnTheRight(k, k + 1, rotation);
        }
    }

    template <int N, typename Workspace>
    UIPC_DEVICE Eigen::ComputationInfo compute_from_tridiagonal_strided(
        Vector<Float, N>& diag, Vector<Float, N - 1>& subdiag, Workspace& q)
    {
        int             end              = N - 1;
        int             start            = 0;
        int             iter             = 0;
        constexpr int   MaxIterations    = 30;
        constexpr Float ConsiderAsZero   = DBL_MIN;
        constexpr Float PrecisionInverse = 1.0 / DBL_EPSILON;

        while(end > 0)
        {
            for(int i = start; i < end; ++i)
            {
                if(Eigen::numext::abs(subdiag(i)) < ConsiderAsZero)
                    subdiag(i) = 0.0;
                else
                {
                    const Float scaled_subdiag = PrecisionInverse * subdiag(i);
                    if(scaled_subdiag * scaled_subdiag
                       <= Eigen::numext::abs(diag(i)) + Eigen::numext::abs(diag(i + 1)))
                        subdiag(i) = 0.0;
                }
            }

            while(end > 0 && subdiag(end - 1) == 0.0)
                --end;
            if(end <= 0)
                break;
            if(++iter > MaxIterations * N)
                break;

            start = end - 1;
            while(start > 0 && subdiag(start - 1) != 0.0)
                --start;
            tridiagonal_qr_step_strided<N>(diag, subdiag, start, end, q);
        }

        const Eigen::ComputationInfo info =
            iter <= MaxIterations * N ? Eigen::Success : Eigen::NoConvergence;
        if(info == Eigen::Success)
        {
            for(int i = 0; i < N - 1; ++i)
            {
                int   selected = i;
                Float value    = diag(i);
                for(int j = i + 1; j < N; ++j)
                {
                    if(diag(j) < value)
                    {
                        value    = diag(j);
                        selected = j;
                    }
                }
                if(selected != i)
                {
                    const Float temporary = diag(i);
                    diag(i)               = diag(selected);
                    diag(selected)        = temporary;
                    q.col(i).swap(q.col(selected));
                }
            }
        }
        return info;
    }
}  // namespace fixed_bank_soa_evd_detail

template <int N, typename Workspace>
UIPC_DEVICE bool selfadjoint_evd_fixed_bank_shared(Workspace& eigen_vectors,
                                                   Vector<Float, N>& eigen_values)
{
    Vector<Float, N - 1> subdiag;
    Vector<Float, N - 1> hcoeffs;

    // SelfAdjointEigenSolver treats only the lower triangle as authoritative.
    for(int row = 0; row < N; ++row)
        for(int col = row + 1; col < N; ++col)
            eigen_vectors(row, col) = 0.0;

    Float scale = eigen_vectors.cwiseAbs().maxCoeff();
    if(scale == 0.0)
        scale = 1.0;
    eigen_vectors.template triangularView<Eigen::Lower>() /= scale;

    fixed_bank_soa_evd_detail::tridiagonalization_inplace_strided<N>(eigen_vectors, hcoeffs);
    eigen_values = eigen_vectors.diagonal();
    subdiag      = eigen_vectors.template diagonal<-1>();
    Eigen::HouseholderSequence<Workspace, Vector<Float, N - 1>> householder(eigen_vectors, hcoeffs);
    eigen_vectors = householder.setLength(N - 1).setShift(1);
    const Eigen::ComputationInfo info =
        fixed_bank_soa_evd_detail::compute_from_tridiagonal_strided<N>(
            eigen_values, subdiag, eigen_vectors);

    eigen_values *= scale;
    for(int i = 0; i < N; ++i)
    {
        auto& value = eigen_values(i);
        value       = value < 0.0 ? 0.0 : value;
    }
    return info == Eigen::Success;
}

template <int N, typename Workspace, typename Output>
UIPC_DEVICE bool make_spd_fixed_bank_shared_upper_fma(Workspace& eigen_vectors, Output& projected)
{
    Vector<Float, N> eigen_values;
    const bool success = selfadjoint_evd_fixed_bank_shared<N>(eigen_vectors, eigen_values);

    for(int row = 0; row < N; ++row)
    {
        for(int col = row; col < N; ++col)
        {
            Float value = 0.0;
#pragma unroll
            for(int k = 0; k < N; ++k)
            {
                const Float weighted_col = eigen_values(k) * eigen_vectors(col, k);
                value = fma(eigen_vectors(row, k), weighted_col, value);
            }
            projected(row, col) = value;
            projected(col, row) = value;
        }
    }
    return success;
}
}  // namespace uipc::backend::cuda
