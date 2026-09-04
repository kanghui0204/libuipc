#include <app/app.h>

#include <utils/fixed_bank_soa_evd.h>
#include <cuda_tool/stream.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

using namespace uipc;
using namespace uipc::backend::cuda;

namespace
{
template <typename T>
class DeviceAllocation
{
  public:
    explicit DeviceAllocation(std::size_t count)
        : m_count{count}
    {
        CUDA_TOOL_CHECK(cudaMalloc(&m_data, count * sizeof(T)));
    }

    DeviceAllocation(const DeviceAllocation&)            = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    ~DeviceAllocation()
    {
        if(m_data)
            cudaFree(m_data);
    }

    T* data() const { return m_data; }

  private:
    T*          m_data  = nullptr;
    std::size_t m_count = 0;
};

template <int N, int LanePitch>
__global__ void fixed_size_evd_test_kernel(const Float* input,
                                            Float*       output_psd,
                                            Float*       output_eigenvalues,
                                            int*         output_status,
                                            int          count)
{
    __shared__ Float shared_workspace[N * N * LanePitch];

    const int matrix_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(matrix_id >= count)
        return;

    FixedBankSoAMap<N, LanePitch> workspace(shared_workspace + threadIdx.x);
    const std::size_t input_offset =
        static_cast<std::size_t>(matrix_id) * N * N;

#pragma unroll
    for(int col = 0; col < N; ++col)
    {
#pragma unroll
        for(int row = 0; row < N; ++row)
        {
            workspace(row, col) =
                row >= col ? input[input_offset + col * N + row] : Float(0);
        }
    }

    Vector<Float, N> eigenvalues;
    if(!selfadjoint_evd_fixed_bank_shared<N>(workspace, eigenvalues))
    {
        output_status[matrix_id] = 1;
        return;
    }

    bool finite = true;
#pragma unroll
    for(int k = 0; k < N; ++k)
    {
        output_eigenvalues[static_cast<std::size_t>(matrix_id) * N + k] =
            eigenvalues(k);
        finite = finite && isfinite(eigenvalues(k));
    }

#pragma unroll
    for(int col = 0; col < N; ++col)
    {
#pragma unroll
        for(int row = 0; row < N; ++row)
        {
            Float value = 0;
#pragma unroll
            for(int k = 0; k < N; ++k)
            {
                value = fma(workspace(row, k),
                            eigenvalues(k) * workspace(col, k),
                            value);
            }
            output_psd[input_offset + col * N + row] = value;
            finite = finite && isfinite(value);
        }
    }

    output_status[matrix_id] = finite ? 0 : 2;
}

template <int N>
void run_fixed_size_evd_test()
{
    static_assert(std::is_same_v<Float, double>);

    constexpr int Count     = 257;
    constexpr int BlockSize = 16;
    constexpr int LanePitch = 16;
    using MatrixN = Eigen::Matrix<Float, N, N>;
    using VectorN = Eigen::Matrix<Float, N, 1>;

    std::mt19937_64                       generator(0x5eed0000ULL + N);
    std::uniform_real_distribution<Float> distribution(-1.0, 1.0);

    std::vector<Float>   input(static_cast<std::size_t>(Count) * N * N);
    std::vector<MatrixN> expected_psd(Count);
    std::vector<VectorN> expected_eigenvalues(Count);

    for(int matrix_id = 0; matrix_id < Count; ++matrix_id)
    {
        MatrixN matrix;
        for(int col = 0; col < N; ++col)
        {
            for(int row = 0; row < N; ++row)
                matrix(row, col) = distribution(generator);
        }
        matrix = ((matrix + matrix.transpose()) * Float(0.5)).eval();

        if(matrix_id == 0)
        {
            matrix.setZero();
        }
        else if(matrix_id == 1)
        {
            matrix.setZero();
            for(int i = 0; i < N; ++i)
            {
                matrix(i, i) =
                    i < N / 3 ? Float(-2) : Float(1);  // repeated eigenvalues
            }
        }
        else if(matrix_id == 2)
        {
            matrix *= Float(1e-12);
        }
        else if(matrix_id == 3)
        {
            matrix *= Float(1e6);
        }

        const std::size_t offset =
            static_cast<std::size_t>(matrix_id) * N * N;
        for(int col = 0; col < N; ++col)
        {
            for(int row = 0; row < N; ++row)
                input[offset + col * N + row] = matrix(row, col);
        }

        Eigen::SelfAdjointEigenSolver<MatrixN> solver(matrix);
        REQUIRE(solver.info() == Eigen::Success);
        expected_eigenvalues[matrix_id] =
            solver.eigenvalues().cwiseMax(Float(0));
        expected_psd[matrix_id] =
            solver.eigenvectors()
            * expected_eigenvalues[matrix_id].asDiagonal()
            * solver.eigenvectors().transpose();
    }

    DeviceAllocation<Float> device_input(input.size());
    DeviceAllocation<Float> device_psd(input.size());
    DeviceAllocation<Float> device_eigenvalues(
        static_cast<std::size_t>(Count) * N);
    DeviceAllocation<int> device_status(Count);

    CUDA_TOOL_CHECK(cudaMemcpy(device_input.data(),
                               input.data(),
                               input.size() * sizeof(Float),
                               cudaMemcpyHostToDevice));
    CUDA_TOOL_CHECK(cudaMemset(device_status.data(), 0xff, Count * sizeof(int)));

    fixed_size_evd_test_kernel<N, LanePitch>
        <<<(Count + BlockSize - 1) / BlockSize, BlockSize>>>(
            device_input.data(),
            device_psd.data(),
            device_eigenvalues.data(),
            device_status.data(),
            Count);
    CUDA_TOOL_CHECK(cudaGetLastError());
    CUDA_TOOL_CHECK(cudaDeviceSynchronize());

    std::vector<Float> output_psd(input.size());
    std::vector<Float> output_eigenvalues(
        static_cast<std::size_t>(Count) * N);
    std::vector<int> output_status(Count);
    CUDA_TOOL_CHECK(cudaMemcpy(output_psd.data(),
                               device_psd.data(),
                               output_psd.size() * sizeof(Float),
                               cudaMemcpyDeviceToHost));
    CUDA_TOOL_CHECK(cudaMemcpy(output_eigenvalues.data(),
                               device_eigenvalues.data(),
                               output_eigenvalues.size() * sizeof(Float),
                               cudaMemcpyDeviceToHost));
    CUDA_TOOL_CHECK(cudaMemcpy(output_status.data(),
                               device_status.data(),
                               output_status.size() * sizeof(int),
                               cudaMemcpyDeviceToHost));

    for(int matrix_id = 0; matrix_id < Count; ++matrix_id)
    {
        REQUIRE(output_status[matrix_id] == 0);
        const std::size_t matrix_offset =
            static_cast<std::size_t>(matrix_id) * N * N;
        const std::size_t eigen_offset =
            static_cast<std::size_t>(matrix_id) * N;

        Eigen::Map<const MatrixN> actual_psd(output_psd.data() + matrix_offset);
        Eigen::Map<const VectorN> actual_eigenvalues(
            output_eigenvalues.data() + eigen_offset);

        const Float psd_scale =
            std::max<Float>(Float(1), expected_psd[matrix_id].norm());
        const Float eigen_scale = std::max<Float>(
            Float(1), expected_eigenvalues[matrix_id].norm());
        REQUIRE((actual_psd - expected_psd[matrix_id]).norm() / psd_scale
                <= Float(1e-12));
        REQUIRE((actual_eigenvalues - expected_eigenvalues[matrix_id]).norm()
                    / eigen_scale
                <= Float(1e-12));
        REQUIRE((actual_psd - actual_psd.transpose()).norm() / psd_scale
                <= Float(1e-12));
    }
}
}  // namespace

TEST_CASE("fixed-bank shared EVD matches CPU Eigen", "[cuda][fixed_size_evd]")
{
    CUDA_TOOL_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024));
    run_fixed_size_evd_test<6>();
    run_fixed_size_evd_test<9>();
    run_fixed_size_evd_test<12>();
}
