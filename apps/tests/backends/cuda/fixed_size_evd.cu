#include <app/app.h>

#include <utils/fixed_bank_soa_evd.h>
#include <utils/contact_type_block_layout.h>

#include <muda/check/check_cuda_errors.h>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
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
        checkCudaErrors(cudaMalloc(&m_data, count * sizeof(T)));
    }

    DeviceAllocation(const DeviceAllocation&)            = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    ~DeviceAllocation()
    {
        if(m_data)
            cudaFree(m_data);
    }

    T*          data() const { return m_data; }
    std::size_t size() const { return m_count; }

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
    const std::size_t input_offset = static_cast<std::size_t>(matrix_id) * N * N;

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

    std::mt19937_64                     generator(0x5eed0000ULL + N);
    std::uniform_real_distribution<Float> distribution(-1.0, 1.0);

    std::vector<Float> input(static_cast<std::size_t>(Count) * N * N);
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
                matrix(i, i) = i < N / 3 ? Float(-2) : (i < 2 * N / 3 ? Float(1) : Float(1));
        }
        else if(matrix_id == 2)
        {
            matrix *= Float(1e-12);
        }
        else if(matrix_id == 3)
        {
            matrix *= Float(1e6);
        }

        const std::size_t offset = static_cast<std::size_t>(matrix_id) * N * N;
        for(int col = 0; col < N; ++col)
        {
            for(int row = 0; row < N; ++row)
                input[offset + col * N + row] = matrix(row, col);
        }

        Eigen::SelfAdjointEigenSolver<MatrixN> solver(matrix);
        REQUIRE(solver.info() == Eigen::Success);
        expected_eigenvalues[matrix_id] = solver.eigenvalues().cwiseMax(Float(0));
        expected_psd[matrix_id] =
            solver.eigenvectors() * expected_eigenvalues[matrix_id].asDiagonal()
            * solver.eigenvectors().transpose();
    }

    DeviceAllocation<Float> device_input(input.size());
    DeviceAllocation<Float> device_psd(input.size());
    DeviceAllocation<Float> device_eigenvalues(static_cast<std::size_t>(Count) * N);
    DeviceAllocation<int>   device_status(Count);

    checkCudaErrors(cudaMemcpy(device_input.data(),
                               input.data(),
                               input.size() * sizeof(Float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(device_status.data(), 0xff, Count * sizeof(int)));

    fixed_size_evd_test_kernel<N, LanePitch>
        <<<(Count + BlockSize - 1) / BlockSize, BlockSize>>>(device_input.data(),
                                                            device_psd.data(),
                                                            device_eigenvalues.data(),
                                                            device_status.data(),
                                                            Count);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<Float> output_psd(input.size());
    std::vector<Float> output_eigenvalues(static_cast<std::size_t>(Count) * N);
    std::vector<int>   output_status(Count);
    checkCudaErrors(cudaMemcpy(output_psd.data(),
                               device_psd.data(),
                               output_psd.size() * sizeof(Float),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(output_eigenvalues.data(),
                               device_eigenvalues.data(),
                               output_eigenvalues.size() * sizeof(Float),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(output_status.data(),
                               device_status.data(),
                               output_status.size() * sizeof(int),
                               cudaMemcpyDeviceToHost));

    for(int matrix_id = 0; matrix_id < Count; ++matrix_id)
    {
        REQUIRE(output_status[matrix_id] == 0);
        const std::size_t matrix_offset = static_cast<std::size_t>(matrix_id) * N * N;
        const std::size_t eigen_offset  = static_cast<std::size_t>(matrix_id) * N;

        Eigen::Map<const MatrixN> actual_psd(output_psd.data() + matrix_offset);
        Eigen::Map<const VectorN> actual_eigenvalues(output_eigenvalues.data() + eigen_offset);

        const Float psd_scale = std::max<Float>(Float(1), expected_psd[matrix_id].norm());
        const Float eigen_scale =
            std::max<Float>(Float(1), expected_eigenvalues[matrix_id].norm());
        REQUIRE((actual_psd - expected_psd[matrix_id]).norm() / psd_scale <= Float(1e-12));
        REQUIRE((actual_eigenvalues - expected_eigenvalues[matrix_id]).norm() / eigen_scale
                <= Float(1e-12));
        REQUIRE((actual_psd - actual_psd.transpose()).norm() / psd_scale <= Float(1e-12));
    }
}
}  // namespace

TEST_CASE("fixed-size shared EVD matches CPU Eigen", "[fixed_size_evd]")
{
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024));
    run_fixed_size_evd_test<6>();
    run_fixed_size_evd_test<9>();
    run_fixed_size_evd_test<12>();
}

template <int BlockSize>
void check_contact_type_block_layout()
{
    for(int pt_count = 0; pt_count <= 2 * BlockSize; ++pt_count)
    {
        for(int ee_count = 0; ee_count <= 2 * BlockSize; ++ee_count)
        {
            for(int pe_count = 0; pe_count <= 2 * BlockSize; ++pe_count)
            {
                for(int pp_count = 0; pp_count <= 2 * BlockSize; ++pp_count)
                {
                    const auto layout = make_contact_type_block_layout<BlockSize>(
                        pt_count, ee_count, pe_count, pp_count);
                    REQUIRE(layout.pt_end == pt_count);
                    REQUIRE(layout.ee_end - layout.ee_offset == ee_count);
                    REQUIRE(layout.pe_end - layout.pe_offset == pe_count);
                    REQUIRE(layout.padded_total - layout.pp_offset == pp_count);
                    REQUIRE(layout.ee_offset % BlockSize == 0);
                    REQUIRE(layout.pe_offset % BlockSize == 0);
                    REQUIRE(layout.pp_offset % BlockSize == 0);

                    for(int block_begin = 0; block_begin < layout.padded_total;
                        block_begin += BlockSize)
                    {
                        int block_type = -1;
                        for(int idx = block_begin;
                            idx < block_begin + BlockSize && idx < layout.padded_total;
                            ++idx)
                        {
                            int type = -1;
                            if(idx < layout.pt_end)
                                type = 0;
                            else if(idx >= layout.ee_offset && idx < layout.ee_end)
                                type = 1;
                            else if(idx >= layout.pe_offset && idx < layout.pe_end)
                                type = 2;
                            else if(idx >= layout.pp_offset)
                                type = 3;

                            if(type >= 0)
                            {
                                if(block_type < 0)
                                    block_type = type;
                                REQUIRE(type == block_type);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("contact types never share a padded block", "[contact_block_padding]")
{
    check_contact_type_block_layout<8>();
    check_contact_type_block_layout<12>();
}
