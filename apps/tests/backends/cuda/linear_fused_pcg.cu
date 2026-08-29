#include <app/app.h>

#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/fused_pcg_kernels.h>
#include <linear_system/global_preconditioner.h>
#include <linear_system/local_preconditioner.h>
#include <linear_system/spmv.h>
#include <backends/cuda/sim_engine.h>
#include <uipc/backend/engine_create_info.h>
#include <muda/buffer/device_var.h>
#include <muda/buffer/device_buffer.h>
#include <muda/check/check_cuda_errors.h>
#include <muda/ext/linear_system/device_bcoo_matrix.h>
#include <muda/ext/linear_system/device_dense_vector.h>
#include <cub/block/block_reduce.cuh>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace uipc;
using namespace uipc::backend::cuda;

namespace
{
using DeviceVector = muda::DeviceDenseVector<Float>;
using Matrix3      = Eigen::Matrix<Float, 3, 3>;

template <typename Tag, typename Tag::type Member>
struct PrivateMemberAccess
{
    friend constexpr typename Tag::type get_private_member(Tag)
    {
        return Member;
    }
};

struct LocalPreconditionerSubsystemMember
{
    using type = DiagLinearSubsystem* LocalPreconditioner::*;
    friend constexpr type get_private_member(LocalPreconditionerSubsystemMember);
};

struct DiagLinearSubsystemIndexMember
{
    using type = SizeT    DiagLinearSubsystem::*;
    friend constexpr type get_private_member(DiagLinearSubsystemIndexMember);
};

template struct PrivateMemberAccess<LocalPreconditionerSubsystemMember, &LocalPreconditioner::m_subsystem>;
template struct PrivateMemberAccess<DiagLinearSubsystemIndexMember, &DiagLinearSubsystem::m_index>;

class PolicyDiagSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

  private:
    void do_init(InitInfo&) override {}
    void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo&) override
    {
    }
    void  do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo&) override {}
    void  do_report_extent(GlobalLinearSystem::DiagExtentInfo&) override {}
    void  do_assemble(GlobalLinearSystem::DiagInfo&) override {}
    void  do_accuracy_check(GlobalLinearSystem::AccuracyInfo&) override {}
    void  do_retrieve_solution(GlobalLinearSystem::SolutionInfo&) override {}
    Float do_diag_norm(GlobalLinearSystem::DiagNormInfo&) override
    {
        return 0.0;
    }
    Float do_mass_norm(GlobalLinearSystem::DiagNormInfo&) override
    {
        return 0.0;
    }
    U64 get_uid() const noexcept override { return 0; }
};

class PolicyLocalPreconditioner final : public LocalPreconditioner
{
  public:
    PolicyLocalPreconditioner(SimEngine& engine, bool supports_fused)
        : LocalPreconditioner{engine}
        , m_supports_fused{supports_fused}
    {
    }

  private:
    void do_build(BuildInfo&) override {}
    void do_init(InitInfo&) override {}
    void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo&) override
    {
    }
    void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo&) override {}
    bool do_supports_fused_pcg() const override { return m_supports_fused; }

    bool m_supports_fused = false;
};

class PolicyGlobalPreconditioner final : public GlobalPreconditioner
{
  public:
    using GlobalPreconditioner::GlobalPreconditioner;

  private:
    void do_build(BuildInfo&) override {}
    void do_assemble(GlobalLinearSystem::GlobalPreconditionerAssemblyInfo&) override
    {
    }
    void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo&) override {}
};

template <typename T>
T copy_var(const muda::DeviceVar<T>& value)
{
    return static_cast<T>(value);
}

std::vector<Float> copy_vector(const DeviceVector& value)
{
    std::vector<Float> result;
    value.copy_to(result);
    return result;
}

std::vector<Float> copy_vector_allocation(const DeviceVector& value, SizeT count)
{
    REQUIRE(count <= value.capacity());
    std::vector<Float> result(count);
    checkCudaErrors(cudaMemcpy(
        result.data(), value.cview().origin_data(), count * sizeof(Float), cudaMemcpyDeviceToHost));
    return result;
}

void poison_vector_tail(DeviceVector& value, SizeT logical_size, SizeT allocation_size)
{
    REQUIRE(logical_size <= value.size());
    REQUIRE(allocation_size <= value.capacity());
    REQUIRE(logical_size <= allocation_size);
    const std::vector<Float> poison(allocation_size - logical_size,
                                    std::numeric_limits<Float>::quiet_NaN());
    checkCudaErrors(cudaMemcpy(value.view().origin_data() + logical_size,
                               poison.data(),
                               poison.size() * sizeof(Float),
                               cudaMemcpyHostToDevice));
}

void require_poisoned_tail(const DeviceVector& value, SizeT logical_size, SizeT allocation_size)
{
    const auto allocation = copy_vector_allocation(value, allocation_size);
    for(SizeT i = logical_size; i < allocation_size; ++i)
        REQUIRE(std::isnan(allocation[i]));
}

void require_near(const std::vector<Float>& actual,
                  const std::vector<Float>& expected,
                  Float                     absolute_tolerance,
                  Float                     relative_tolerance)
{
    REQUIRE(actual.size() == expected.size());
    for(SizeT i = 0; i < actual.size(); ++i)
    {
        const Float scale = std::max(std::abs(actual[i]), std::abs(expected[i]));
        REQUIRE(std::abs(actual[i] - expected[i])
                <= absolute_tolerance + relative_tolerance * scale);
    }
}

struct GraphOwner
{
    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t exec  = nullptr;

    GraphOwner()                             = default;
    GraphOwner(const GraphOwner&)            = delete;
    GraphOwner& operator=(const GraphOwner&) = delete;
    GraphOwner(GraphOwner&& other) noexcept
        : graph{std::exchange(other.graph, nullptr)}
        , exec{std::exchange(other.exec, nullptr)}
    {
    }
    GraphOwner& operator=(GraphOwner&&) = delete;

    ~GraphOwner()
    {
        if(exec)
            cudaGraphExecDestroy(exec);
        if(graph)
            cudaGraphDestroy(graph);
    }
};

__global__ void force_terminal_iteration_kernel(IndexT*             status,
                                                FusedPcgCheckState* check_state,
                                                const FusedPcgDeviceParams* params,
                                                IndexT current_iteration,
                                                IndexT terminal_iteration)
{
    if(blockIdx.x == 0 && threadIdx.x == 0 && current_iteration <= params->active_iterations
       && current_iteration == terminal_iteration
       && *status == static_cast<IndexT>(FusedPcgStatus::Running))
    {
        *status             = static_cast<IndexT>(FusedPcgStatus::Converged);
        check_state->status = *status;
        check_state->iteration_in_chunk = current_iteration;
    }
}

__global__ void legacy_update_xr_kernel(Float*        x,
                                        const Float*  p,
                                        Float*        r,
                                        const Float*  Ap,
                                        SizeT         size,
                                        const Float*  alpha,
                                        const IndexT* status)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < size && *status == static_cast<IndexT>(FusedPcgStatus::Running))
    {
        x[i] += *alpha * p[i];
        r[i] -= *alpha * Ap[i];
    }
}

// Test-only decomposition of the v0.0.25 fused_update_xr arithmetic. The
// production legacy kernel computes the same quotient independently in every
// vector thread; separating it here lets the test reuse the extracted legacy
// update/apply/dot launchers without changing the division semantics.
__global__ void legacy_alpha_oracle_kernel(const Float* rz_old,
                                           const Float* pAp,
                                           Float*       alpha,
                                           IndexT*      status)
{
    if(blockIdx.x != 0 || threadIdx.x != 0
       || *status != static_cast<IndexT>(FusedPcgStatus::Running))
        return;

    *alpha = *rz_old / *pAp;
}

__global__ void legacy_dot_kernel(
    const Float* lhs, const Float* rhs, SizeT size, Float* result, const IndexT* status)
{
    constexpr int BlockSize = 256;
    using BlockReduce       = cub::BlockReduce<Float, BlockSize>;
    __shared__ typename BlockReduce::TempStorage storage;

    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Float value =
        i < size && *status == static_cast<IndexT>(FusedPcgStatus::Running) ?
            lhs[i] * rhs[i] :
            0.0;
    const Float sum = BlockReduce(storage).Sum(value);
    if(threadIdx.x == 0)
        atomicAdd(result, sum);
}

struct FullGraphFixture
{
    static constexpr int ScalarCount = 36;
    static constexpr int BlockCount  = ScalarCount / 3;

    muda::DeviceBCOOMatrix<Float, 3> A;
    Spmv                             spmv;
    DeviceVector                     x{ScalarCount};
    DeviceVector                     r{ScalarCount};
    DeviceVector                     z{ScalarCount};
    DeviceVector                     p{ScalarCount};
    std::array<DeviceVector, 2> Ap{DeviceVector{ScalarCount}, DeviceVector{ScalarCount}};
    std::array<muda::DeviceVar<Float>, 2> pAp;
    std::array<muda::DeviceVar<Float>, 2> rz_old;
    std::array<muda::DeviceVar<Float>, 2> rz_new;
    muda::DeviceVar<Float>                beta;
    muda::DeviceVar<IndexT>               status;
    muda::DeviceVar<FusedPcgDeviceParams> params;
    muda::DeviceVar<FusedPcgCheckState>   check_state;
    cudaStream_t                          stream = nullptr;
    std::vector<Float>                    diagonal;
    std::vector<Float>                    dense_matrix;
    std::vector<Float>                    rhs;
    int                                   triplet_count = BlockCount;

    struct Snapshot
    {
        std::vector<Float>                x;
        std::vector<Float>                r;
        std::vector<Float>                z;
        std::vector<Float>                p;
        std::array<std::vector<Float>, 2> Ap;
        std::array<Float, 2>              pAp;
        std::array<Float, 2>              rz_old;
        std::array<Float, 2>              rz_new;
        Float                             beta   = 0.0;
        IndexT                            status = 0;
        FusedPcgCheckState                check_state;
    };

    explicit FullGraphFixture(bool coupled = false)
    {
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        diagonal.resize(ScalarCount);
        dense_matrix.assign(ScalarCount * ScalarCount, 0.0);
        rhs.resize(ScalarCount);
        for(int i = 0; i < ScalarCount; ++i)
        {
            diagonal[i] = Float{1.0} + Float{0.025} * static_cast<Float>(i);
            rhs[i]      = Float{0.2} + Float{0.01} * static_cast<Float>(i % 11);
        }

        std::vector<int>     rows;
        std::vector<int>     cols;
        std::vector<Matrix3> values;
        rows.reserve(coupled ? 2 * BlockCount - 1 : BlockCount);
        cols.reserve(coupled ? 2 * BlockCount - 1 : BlockCount);
        values.reserve(coupled ? 2 * BlockCount - 1 : BlockCount);
        for(int block = 0; block < BlockCount; ++block)
        {
            Matrix3 diagonal_block = Matrix3::Zero();
            for(int component = 0; component < 3; ++component)
                diagonal_block(component, component) =
                    coupled ? Float{4.0} + Float{0.025} * (3 * block + component) :
                              diagonal[3 * block + component];
            if(coupled)
            {
                diagonal_block(0, 1) = diagonal_block(1, 0) = 0.025;
                diagonal_block(1, 2) = diagonal_block(2, 1) = -0.015;
            }
            rows.push_back(block);
            cols.push_back(block);
            values.push_back(diagonal_block);

            for(int row_component = 0; row_component < 3; ++row_component)
                for(int col_component = 0; col_component < 3; ++col_component)
                    dense_matrix[(3 * block + row_component) * ScalarCount + 3 * block + col_component] =
                        diagonal_block(row_component, col_component);

            if(coupled && block + 1 < BlockCount)
            {
                Matrix3 cross = Matrix3::Zero();
                cross(0, 0)   = -0.15;
                cross(1, 1)   = -0.10;
                cross(2, 2)   = -0.05;
                cross(0, 2)   = 0.02;
                rows.push_back(block);
                cols.push_back(block + 1);
                values.push_back(cross);
                for(int row_component = 0; row_component < 3; ++row_component)
                    for(int col_component = 0; col_component < 3; ++col_component)
                    {
                        const Float entry = cross(row_component, col_component);
                        dense_matrix[(3 * block + row_component) * ScalarCount + 3 * (block + 1) + col_component] =
                            entry;
                        dense_matrix[(3 * (block + 1) + col_component) * ScalarCount + 3 * block + row_component] =
                            entry;
                    }
            }
        }

        triplet_count = static_cast<int>(rows.size());
        A.resize(BlockCount, BlockCount, triplet_count);
        A.row_indices().copy_from(rows.data());
        A.col_indices().copy_from(cols.data());
        A.values().copy_from(values.data());
        reset(-1.0);
    }

    ~FullGraphFixture()
    {
        if(stream)
            cudaStreamDestroy(stream);
    }

    void reset(Float tolerance)
    {
        const int          scalar_count = static_cast<int>(x.size());
        std::vector<Float> zeros(scalar_count, 0.0);
        x = Eigen::Map<const Eigen::VectorXd>(zeros.data(), zeros.size());
        r = Eigen::Map<const Eigen::VectorXd>(rhs.data(), scalar_count);
        z = Eigen::Map<const Eigen::VectorXd>(rhs.data(), scalar_count);
        p = Eigen::Map<const Eigen::VectorXd>(rhs.data(), scalar_count);
        for(auto& value : Ap)
            value = Eigen::Map<const Eigen::VectorXd>(zeros.data(), zeros.size());
        for(auto& value : pAp)
            value = 0.0;
        for(auto& value : rz_new)
            value = 0.0;
        beta = -103.0;

        const Float initial_rz = std::inner_product(
            rhs.begin(), rhs.begin() + scalar_count, rhs.begin(), Float{0.0});
        rz_old[0] = initial_rz;
        rz_old[1] = 0.0;
        status    = static_cast<IndexT>(FusedPcgStatus::Running);

        FusedPcgDeviceParams host_params;
        host_params.tolerance         = tolerance;
        host_params.triplet_count     = triplet_count;
        host_params.active_iterations = 10;
        params                        = host_params;

        FusedPcgCheckState host_check;
        host_check.rz     = initial_rz;
        host_check.status = static_cast<IndexT>(FusedPcgStatus::Running);
        host_check.iteration_in_chunk = 0;
        check_state                   = host_check;
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void resize_logical_dof(int scalar_count)
    {
        REQUIRE(scalar_count > 0);
        REQUIRE(scalar_count <= ScalarCount);
        REQUIRE(scalar_count % 3 == 0);
        x.resize(scalar_count);
        r.resize(scalar_count);
        z.resize(scalar_count);
        p.resize(scalar_count);
        for(auto& value : Ap)
            value.resize(scalar_count);
        triplet_count = scalar_count / 3;
    }

    void launch_iteration(int current_slot, int iteration_in_chunk, int forced_terminal_iteration = 0)
    {
        const int next_slot = current_slot ^ 1;
        spmv.rbk_sym_spmv_dot_pipelined(A.cview(),
                                        p.cview(),
                                        Ap[current_slot].view(),
                                        pAp[current_slot].view(),
                                        Ap[next_slot].view(),
                                        pAp[next_slot].view(),
                                        status.view(),
                                        params.view(),
                                        iteration_in_chunk,
                                        triplet_count,
                                        stream);
        launch_fused_pcg_identity_update_apply_dot(x.view(),
                                                   p.cview(),
                                                   r.view(),
                                                   Ap[current_slot].cview(),
                                                   z.view(),
                                                   rz_old[current_slot].view(),
                                                   pAp[current_slot].view(),
                                                   rz_new[current_slot].view(),
                                                   status.view(),
                                                   params.view(),
                                                   iteration_in_chunk,
                                                   stream);
        if(forced_terminal_iteration > 0)
        {
            launch_fused_pcg_update_convergence(rz_old[current_slot].view(),
                                                rz_new[current_slot].view(),
                                                beta.view(),
                                                status.view(),
                                                check_state.view(),
                                                params.view(),
                                                iteration_in_chunk,
                                                stream);
            force_terminal_iteration_kernel<<<1, 1, 0, stream>>>(
                status.data(), check_state.data(), params.data(), iteration_in_chunk, forced_terminal_iteration);
            launch_fused_pcg_update_p_prepare_next(p.view(),
                                                   z.cview(),
                                                   beta.view(),
                                                   rz_new[current_slot].view(),
                                                   rz_old[next_slot].view(),
                                                   rz_new[next_slot].view(),
                                                   status.view(),
                                                   params.view(),
                                                   iteration_in_chunk,
                                                   stream);
        }
        else
        {
            launch_fused_pcg_update_convergence_p_prepare_next(
                p.view(),
                z.cview(),
                rz_old[current_slot].view(),
                rz_new[current_slot].view(),
                beta.view(),
                rz_old[next_slot].view(),
                rz_new[next_slot].view(),
                status.view(),
                check_state.view(),
                params.view(),
                iteration_in_chunk,
                stream);
        }
    }

    GraphOwner capture(int iterations, int starting_slot)
    {
        GraphOwner owner;
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for(int iteration = 1; iteration <= iterations; ++iteration)
            launch_iteration((starting_slot + iteration - 1) & 1, iteration);
        checkCudaErrors(cudaStreamEndCapture(stream, &owner.graph));
        checkCudaErrors(cudaGraphInstantiate(&owner.exec, owner.graph, nullptr, nullptr, 0));
        return owner;
    }

    GraphOwner capture_with_forced_terminal(int iterations, int terminal_iteration)
    {
        GraphOwner owner;
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for(int iteration = 1; iteration <= iterations; ++iteration)
            launch_iteration((iteration - 1) & 1, iteration, terminal_iteration);
        checkCudaErrors(cudaStreamEndCapture(stream, &owner.graph));
        checkCudaErrors(cudaGraphInstantiate(&owner.exec, owner.graph, nullptr, nullptr, 0));
        return owner;
    }

    FusedPcgGraphSignature graph_signature() const
    {
        FusedPcgGraphSignature signature;
        signature.matrix_rows              = A.row_indices().data();
        signature.matrix_cols              = A.col_indices().data();
        signature.matrix_values            = A.values().data();
        signature.x                        = x.cview().data();
        signature.r                        = r.cview().data();
        signature.z                        = z.cview().data();
        signature.p                        = p.cview().data();
        signature.Ap_0                     = Ap[0].cview().data();
        signature.Ap_1                     = Ap[1].cview().data();
        signature.pAp_0                    = pAp[0].data();
        signature.pAp_1                    = pAp[1].data();
        signature.rz_old_0                 = rz_old[0].data();
        signature.rz_old_1                 = rz_old[1].data();
        signature.rz_new_0                 = rz_new[0].data();
        signature.rz_new_1                 = rz_new[1].data();
        signature.device_params            = params.data();
        signature.status                   = status.data();
        signature.check_state              = check_state.data();
        signature.beta                     = beta.data();
        signature.scalar_dof_count         = x.size();
        signature.triplet_bucket           = A.triplet_capacity();
        signature.preconditioner_signature = 1;
        signature.check_interval           = 10;
        return signature;
    }

    void force_bound_storage_reallocation()
    {
        A.reserve_triplets(A.triplet_capacity() + 64);

        const SizeT new_vector_capacity = x.capacity() + 64;
        x.reserve(new_vector_capacity);
        r.reserve(new_vector_capacity);
        z.reserve(new_vector_capacity);
        p.reserve(new_vector_capacity);
        for(auto& value : Ap)
            value.reserve(new_vector_capacity);

        checkCudaErrors(cudaDeviceSynchronize());
    }

    Snapshot snapshot() const
    {
        Snapshot value;
        value.x = copy_vector(x);
        value.r = copy_vector(r);
        value.z = copy_vector(z);
        value.p = copy_vector(p);
        for(int slot = 0; slot < 2; ++slot)
        {
            value.Ap[slot]     = copy_vector(Ap[slot]);
            value.pAp[slot]    = copy_var(pAp[slot]);
            value.rz_old[slot] = copy_var(rz_old[slot]);
            value.rz_new[slot] = copy_var(rz_new[slot]);
        }
        value.beta        = copy_var(beta);
        value.status      = copy_var(status);
        value.check_state = copy_var(check_state);
        return value;
    }

    std::vector<Float> cpu_reference(int iterations) const
    {
        const int          scalar_count = static_cast<int>(x.size());
        std::vector<Float> x_ref(scalar_count, 0.0);
        std::vector<Float> r_ref(rhs.begin(), rhs.begin() + scalar_count);
        std::vector<Float> z_ref = r_ref;
        std::vector<Float> p_ref = z_ref;
        Float              rz =
            std::inner_product(r_ref.begin(), r_ref.end(), z_ref.begin(), Float{0.0});

        for(int iteration = 0; iteration < iterations; ++iteration)
        {
            std::vector<Float> Ap_ref(scalar_count);
            for(int i = 0; i < scalar_count; ++i)
                for(int j = 0; j < scalar_count; ++j)
                    Ap_ref[i] += dense_matrix[i * ScalarCount + j] * p_ref[j];
            const Float pAp_value =
                std::inner_product(p_ref.begin(), p_ref.end(), Ap_ref.begin(), Float{0.0});
            const Float alpha_value = rz / pAp_value;
            for(int i = 0; i < scalar_count; ++i)
            {
                x_ref[i] += alpha_value * p_ref[i];
                r_ref[i] -= alpha_value * Ap_ref[i];
                z_ref[i] = r_ref[i];
            }
            const Float rz_next =
                std::inner_product(r_ref.begin(), r_ref.end(), z_ref.begin(), Float{0.0});
            const Float beta_value = rz_next / rz;
            for(int i = 0; i < scalar_count; ++i)
                p_ref[i] = z_ref[i] + beta_value * p_ref[i];
            rz = rz_next;
        }
        return x_ref;
    }
};

struct ActualDiagonalPreconditionerFixture
{
    static constexpr int AbdBodyCount   = 5;
    static constexpr int FemVertexCount = 96;
    static constexpr int AbdScalarCount = AbdBodyCount * 12;
    static constexpr int FemScalarCount = FemVertexCount * 3;
    static constexpr int ScalarCount    = AbdScalarCount + FemScalarCount;
    static constexpr int BlockRows      = ScalarCount / 3;

    muda::DeviceBCOOMatrix<Float, 3> A;
    Spmv                             spmv;
    muda::DeviceBuffer<Matrix12x12>  abd_diag_inv;
    muda::DeviceBuffer<Matrix3x3>    fem_diag_inv;
    DeviceVector                     x{ScalarCount};
    DeviceVector                     r{ScalarCount};
    DeviceVector                     z{ScalarCount};
    DeviceVector                     p{ScalarCount};
    std::array<DeviceVector, 2> Ap{DeviceVector{ScalarCount}, DeviceVector{ScalarCount}};
    std::array<muda::DeviceVar<Float>, 2> pAp;
    std::array<muda::DeviceVar<Float>, 2> rz_old;
    std::array<muda::DeviceVar<Float>, 2> rz_new;
    muda::DeviceVar<Float>                alpha;
    muda::DeviceVar<Float>                beta;
    muda::DeviceVar<IndexT>               status;
    muda::DeviceVar<FusedPcgDeviceParams> params;
    muda::DeviceVar<FusedPcgCheckState>   check_state;
    cudaStream_t                          stream = nullptr;
    std::vector<Float>                    rhs;
    std::vector<Float>                    initial_z;
    int                                   triplet_count = 0;

    ActualDiagonalPreconditionerFixture()
    {
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::vector<int>     rows;
        std::vector<int>     cols;
        std::vector<Matrix3> values;
        rows.reserve(2 * BlockRows - 1);
        cols.reserve(2 * BlockRows - 1);
        values.reserve(2 * BlockRows - 1);
        for(int block = 0; block < BlockRows; ++block)
        {
            Matrix3 diagonal = Matrix3::Identity() * 5.0;
            diagonal(0, 1) = diagonal(1, 0) = 0.03;
            diagonal(1, 2) = diagonal(2, 1) = -0.02;
            rows.push_back(block);
            cols.push_back(block);
            values.push_back(diagonal);
            if(block + 1 < BlockRows)
            {
                Matrix3 cross = Matrix3::Identity() * -0.2;
                cross(0, 2)   = 0.01;
                rows.push_back(block);
                cols.push_back(block + 1);
                values.push_back(cross);
            }
        }
        triplet_count = static_cast<int>(rows.size());
        A.resize(BlockRows, BlockRows, triplet_count);
        A.row_indices().copy_from(rows.data());
        A.col_indices().copy_from(cols.data());
        A.values().copy_from(values.data());

        std::vector<Matrix12x12> host_abd(AbdBodyCount, Matrix12x12::Zero());
        std::vector<Matrix3x3>   host_fem(FemVertexCount, Matrix3x3::Zero());
        // Deterministic dense, symmetric, strictly diagonally dominant blocks
        // are SPD and exercise every cross-component shuffle/load path.
        for(int body = 0; body < AbdBodyCount; ++body)
        {
            for(int row = 0; row < 12; ++row)
            {
                host_abd[body](row, row) = 0.18 + 0.002 * row + 0.0001 * (body % 7);
                for(int col = 0; col < row; ++col)
                {
                    const Float magnitude =
                        0.0001 * (1 + ((body + 3 * row + 5 * col) % 5));
                    const Float coupling =
                        ((body + row + col) & 1) == 0 ? magnitude : -magnitude;
                    host_abd[body](row, col) = coupling;
                    host_abd[body](col, row) = coupling;
                }
            }
        }
        for(int vertex = 0; vertex < FemVertexCount; ++vertex)
        {
            host_fem[vertex](0, 0) = 0.20 + 0.00001 * (vertex % 13);
            host_fem[vertex](1, 1) = 0.22 + 0.00001 * (vertex % 11);
            host_fem[vertex](2, 2) = 0.24 + 0.00001 * (vertex % 17);
            host_fem[vertex](0, 1) = host_fem[vertex](1, 0) =
                ((vertex & 1) == 0 ? 0.001 : -0.001);
            host_fem[vertex](0, 2) = host_fem[vertex](2, 0) =
                ((vertex & 2) == 0 ? -0.0008 : 0.0008);
            host_fem[vertex](1, 2) = host_fem[vertex](2, 1) =
                ((vertex & 4) == 0 ? 0.0006 : -0.0006);
        }
        abd_diag_inv.resize(AbdBodyCount);
        fem_diag_inv.resize(FemVertexCount);
        abd_diag_inv.view().copy_from(host_abd.data());
        fem_diag_inv.view().copy_from(host_fem.data());

        rhs.resize(ScalarCount);
        initial_z.resize(ScalarCount);
        for(int i = 0; i < ScalarCount; ++i)
            rhs[i] = 0.05 + 0.0002 * (i % 101);
        for(int body = 0; body < AbdBodyCount; ++body)
        {
            const Eigen::Map<const Vector12> host_r(rhs.data() + body * 12);
            const Vector12                   host_z = host_abd[body] * host_r;
            std::copy(host_z.data(), host_z.data() + 12, initial_z.data() + body * 12);
        }
        for(int vertex = 0; vertex < FemVertexCount; ++vertex)
        {
            const Eigen::Map<const Vector3> host_r(rhs.data() + AbdScalarCount + vertex * 3);
            const Vector3 host_z = host_fem[vertex] * host_r;
            std::copy(host_z.data(),
                      host_z.data() + 3,
                      initial_z.data() + AbdScalarCount + vertex * 3);
        }
        reset();
    }

    ~ActualDiagonalPreconditionerFixture()
    {
        if(stream)
            cudaStreamDestroy(stream);
    }

    void reset()
    {
        Eigen::VectorXd zeros = Eigen::VectorXd::Zero(ScalarCount);
        x                     = zeros;
        r = Eigen::Map<const Eigen::VectorXd>(rhs.data(), rhs.size());
        z = Eigen::Map<const Eigen::VectorXd>(initial_z.data(), initial_z.size());
        p = Eigen::Map<const Eigen::VectorXd>(initial_z.data(), initial_z.size());
        for(auto& value : Ap)
            value = zeros;
        for(auto& value : pAp)
            value = 0.0;
        for(auto& value : rz_new)
            value = 0.0;

        const Float initial_rz =
            std::inner_product(rhs.begin(), rhs.end(), initial_z.begin(), Float{0.0});
        rz_old[0] = initial_rz;
        rz_old[1] = 0.0;
        status    = static_cast<IndexT>(FusedPcgStatus::Running);

        FusedPcgDeviceParams host_params;
        host_params.tolerance         = -1.0;
        host_params.triplet_count     = triplet_count;
        host_params.active_iterations = 10;
        params                        = host_params;

        FusedPcgCheckState host_check;
        host_check.rz     = initial_rz;
        host_check.status = static_cast<IndexT>(FusedPcgStatus::Running);
        host_check.iteration_in_chunk = 0;
        check_state                   = host_check;
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void launch_fused_iteration(int current_slot, int iteration)
    {
        const int next_slot = current_slot ^ 1;
        spmv.rbk_sym_spmv_dot_pipelined(A.cview(),
                                        p.cview(),
                                        Ap[current_slot].view(),
                                        pAp[current_slot].view(),
                                        Ap[next_slot].view(),
                                        pAp[next_slot].view(),
                                        status.view(),
                                        params.view(),
                                        iteration,
                                        triplet_count,
                                        stream);
        launch_fused_pcg_abd_update_apply_dot(abd_diag_inv.view(),
                                              x.view().subview(0, AbdScalarCount),
                                              p.cview().subview(0, AbdScalarCount),
                                              r.view().subview(0, AbdScalarCount),
                                              Ap[current_slot].cview().subview(0, AbdScalarCount),
                                              z.view().subview(0, AbdScalarCount),
                                              rz_old[current_slot].view(),
                                              pAp[current_slot].view(),
                                              rz_new[current_slot].view(),
                                              status.view(),
                                              params.view(),
                                              iteration,
                                              stream);
        launch_fused_pcg_fem_update_apply_dot(
            fem_diag_inv.view(),
            x.view().subview(AbdScalarCount, FemScalarCount),
            p.cview().subview(AbdScalarCount, FemScalarCount),
            r.view().subview(AbdScalarCount, FemScalarCount),
            Ap[current_slot].cview().subview(AbdScalarCount, FemScalarCount),
            z.view().subview(AbdScalarCount, FemScalarCount),
            rz_old[current_slot].view(),
            pAp[current_slot].view(),
            rz_new[current_slot].view(),
            status.view(),
            params.view(),
            iteration,
            stream);
        launch_fused_pcg_update_convergence_p_prepare_next(
            p.view(),
            z.cview(),
            rz_old[current_slot].view(),
            rz_new[current_slot].view(),
            beta.view(),
            rz_old[next_slot].view(),
            rz_new[next_slot].view(),
            status.view(),
            check_state.view(),
            params.view(),
            iteration,
            stream);
    }

    GraphOwner capture(int iterations)
    {
        GraphOwner owner;
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for(int iteration = 1; iteration <= iterations; ++iteration)
            launch_fused_iteration((iteration - 1) & 1, iteration);
        checkCudaErrors(cudaStreamEndCapture(stream, &owner.graph));
        checkCudaErrors(cudaGraphInstantiate(&owner.exec, owner.graph, nullptr, nullptr, 0));
        return owner;
    }

    void run_legacy(int iterations)
    {
        constexpr int VectorBlockSize = 64;
        const int vector_grid = (ScalarCount + VectorBlockSize - 1) / VectorBlockSize;
        const int dot_grid = (ScalarCount + 255) / 256;
        for(int iteration = 1; iteration <= iterations; ++iteration)
        {
            const int current_slot = (iteration - 1) & 1;
            const int next_slot    = current_slot ^ 1;
            spmv.rbk_sym_spmv_dot(1.0,
                                  A.cview(),
                                  p.cview(),
                                  0.0,
                                  Ap[current_slot].view(),
                                  pAp[current_slot].view());
            legacy_alpha_oracle_kernel<<<1, 1>>>(rz_old[current_slot].data(),
                                                 pAp[current_slot].data(),
                                                 alpha.data(),
                                                 status.data());
            legacy_update_xr_kernel<<<vector_grid, VectorBlockSize>>>(
                x.view().data(),
                p.cview().data(),
                r.view().data(),
                Ap[current_slot].cview().data(),
                ScalarCount,
                alpha.data(),
                status.data());
            launch_abd_diag_preconditioner_apply(abd_diag_inv.view(),
                                                 r.cview().subview(0, AbdScalarCount),
                                                 z.view().subview(0, AbdScalarCount),
                                                 status.view());
            launch_fem_diag_preconditioner_apply(
                fem_diag_inv.view(),
                r.cview().subview(AbdScalarCount, FemScalarCount),
                z.view().subview(AbdScalarCount, FemScalarCount),
                status.view());
            legacy_dot_kernel<<<dot_grid, 256>>>(r.cview().data(),
                                                 z.cview().data(),
                                                 ScalarCount,
                                                 rz_new[current_slot].data(),
                                                 status.data());
            launch_fused_pcg_update_convergence(rz_old[current_slot].view(),
                                                rz_new[current_slot].view(),
                                                beta.view(),
                                                status.view(),
                                                check_state.view(),
                                                params.view(),
                                                iteration,
                                                nullptr);
            Ap[next_slot].buffer_view().fill(0.0);
            checkCudaErrors(cudaMemsetAsync(pAp[next_slot].data(), 0, sizeof(Float)));
            launch_fused_pcg_update_p_prepare_next(p.view(),
                                                   z.cview(),
                                                   beta.view(),
                                                   rz_new[current_slot].view(),
                                                   rz_old[next_slot].view(),
                                                   rz_new[next_slot].view(),
                                                   status.view(),
                                                   params.view(),
                                                   iteration,
                                                   nullptr);
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }

    FullGraphFixture::Snapshot snapshot() const
    {
        FullGraphFixture::Snapshot value;
        value.x = copy_vector(x);
        value.r = copy_vector(r);
        value.z = copy_vector(z);
        value.p = copy_vector(p);
        for(int slot = 0; slot < 2; ++slot)
        {
            value.Ap[slot]     = copy_vector(Ap[slot]);
            value.pAp[slot]    = copy_var(pAp[slot]);
            value.rz_old[slot] = copy_var(rz_old[slot]);
            value.rz_new[slot] = copy_var(rz_new[slot]);
        }
        value.beta        = copy_var(beta);
        value.status      = copy_var(status);
        value.check_state = copy_var(check_state);
        return value;
    }
};

void require_snapshot_near(const FullGraphFixture::Snapshot& actual,
                           const FullGraphFixture::Snapshot& expected,
                           bool compare_chunk_local_iteration = true)
{
    require_near(actual.x, expected.x, 1e-12, 1e-12);
    require_near(actual.r, expected.r, 1e-12, 1e-12);
    require_near(actual.z, expected.z, 1e-12, 1e-12);
    require_near(actual.p, expected.p, 1e-12, 1e-12);
    for(int slot = 0; slot < 2; ++slot)
    {
        require_near(actual.Ap[slot], expected.Ap[slot], 1e-12, 1e-12);
        REQUIRE(actual.pAp[slot] == Catch::Approx(expected.pAp[slot]).margin(1e-11));
        REQUIRE(actual.rz_old[slot] == Catch::Approx(expected.rz_old[slot]).margin(1e-11));
        REQUIRE(actual.rz_new[slot] == Catch::Approx(expected.rz_new[slot]).margin(1e-11));
    }
    REQUIRE(actual.beta == Catch::Approx(expected.beta).margin(1e-12));
    REQUIRE(actual.status == expected.status);
    REQUIRE(actual.check_state.status == expected.check_state.status);
    if(compare_chunk_local_iteration)
        REQUIRE(actual.check_state.iteration_in_chunk == expected.check_state.iteration_in_chunk);
    REQUIRE(actual.check_state.rz == Catch::Approx(expected.check_state.rz).margin(1e-11));
}
}  // namespace

TEST_CASE("fused_pcg_scalar_status_and_ping_pong", "[cuda][fused_pcg]")
{
    muda::DeviceVar<Float> rz_old{4.0};
    muda::DeviceVar<Float> beta{0.0};
    muda::DeviceVar<Float> rz_new{1.0};
    muda::DeviceVar<Float> rz_old_next{-7.0};
    muda::DeviceVar<Float> rz_new_next{-9.0};
    muda::DeviceVar<IndexT> status{static_cast<IndexT>(FusedPcgStatus::Running)};
    muda::DeviceVar<FusedPcgDeviceParams> params;
    muda::DeviceVar<FusedPcgCheckState>   check_state;

    FusedPcgDeviceParams host_params;
    host_params.tolerance         = 0.5;
    host_params.active_iterations = 10;
    params                        = host_params;

    launch_fused_pcg_update_convergence(rz_old.view(),
                                        rz_new.view(),
                                        beta.view(),
                                        status.view(),
                                        check_state.view(),
                                        params.view(),
                                        1,
                                        nullptr);
    REQUIRE(copy_var(beta) == Catch::Approx(0.25));
    const auto running_check = copy_var(check_state);
    REQUIRE(running_check.iteration_in_chunk == 1);
    REQUIRE(running_check.status == static_cast<IndexT>(FusedPcgStatus::Running));

    DeviceVector    p{3};
    DeviceVector    z{3};
    Eigen::VectorXd host_p(3);
    Eigen::VectorXd host_z(3);
    host_p << 1.0, 2.0, 3.0;
    host_z << 4.0, 5.0, 6.0;
    p = host_p;
    z = host_z;
    launch_fused_pcg_update_p_prepare_next(p.view(),
                                           z.cview(),
                                           beta.view(),
                                           rz_new.view(),
                                           rz_old_next.view(),
                                           rz_new_next.view(),
                                           status.view(),
                                           params.view(),
                                           1,
                                           nullptr);
    require_near(copy_vector(p), {4.25, 5.5, 6.75}, 1e-14, 1e-14);
    REQUIRE(copy_var(rz_old_next) == Catch::Approx(1.0));
    REQUIRE(copy_var(rz_new_next) == Catch::Approx(0.0));

    SECTION("convergence records the first terminal iteration")
    {
        status = static_cast<IndexT>(FusedPcgStatus::Running);
        rz_new = 0.25;
        launch_fused_pcg_update_convergence(rz_old.view(),
                                            rz_new.view(),
                                            beta.view(),
                                            status.view(),
                                            check_state.view(),
                                            params.view(),
                                            3,
                                            nullptr);
        const auto terminal = copy_var(check_state);
        REQUIRE(terminal.status == static_cast<IndexT>(FusedPcgStatus::Converged));
        REQUIRE(terminal.iteration_in_chunk == 3);
        REQUIRE(terminal.rz == Catch::Approx(0.25));
    }
}

TEST_CASE("fused_pcg_f03_convergence_prepare_fusion_matches_two_node_oracle",
          "[cuda][fused_pcg][graph][f03][focused]")
{
    struct Result
    {
        std::vector<Float> p;
        Float              beta        = 0.0;
        Float              rz_old_next = 0.0;
        Float              rz_new_next = 0.0;
        IndexT             status       = 0;
        FusedPcgCheckState check_state;
        size_t             graph_nodes = 0;
    };

    const auto run = [](bool fused, SizeT vector_size, Float tolerance, IndexT active_iterations)
    {
        DeviceVector p{vector_size};
        DeviceVector z{vector_size};
        if(vector_size != 0)
        {
            std::vector<Float> host_p(vector_size);
            std::vector<Float> host_z(vector_size);
            for(SizeT i = 0; i < vector_size; ++i)
            {
                host_p[i] = Float{0.125} + Float(i % 17) * Float{0.03125};
                host_z[i] = Float{-0.25} + Float(i % 23) * Float{0.015625};
            }
            p = Eigen::Map<const Eigen::VectorXd>(host_p.data(), host_p.size());
            z = Eigen::Map<const Eigen::VectorXd>(host_z.data(), host_z.size());
        }

        muda::DeviceVar<Float> rz_old{4.0};
        muda::DeviceVar<Float> rz_new{1.0};
        muda::DeviceVar<Float> beta{-103.0};
        muda::DeviceVar<Float> rz_old_next{-7.0};
        muda::DeviceVar<Float> rz_new_next{-9.0};
        muda::DeviceVar<IndexT> status{static_cast<IndexT>(FusedPcgStatus::Running)};
        muda::DeviceVar<FusedPcgDeviceParams> params;
        muda::DeviceVar<FusedPcgCheckState>   check_state;

        FusedPcgDeviceParams host_params;
        host_params.tolerance         = tolerance;
        host_params.active_iterations = active_iterations;
        params                        = host_params;
        FusedPcgCheckState host_check;
        host_check.rz                 = -11.0;
        host_check.status             = static_cast<IndexT>(FusedPcgStatus::Running);
        host_check.iteration_in_chunk = -13;
        check_state                   = host_check;

        cudaStream_t stream = nullptr;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        GraphOwner graph;
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        if(fused)
        {
            launch_fused_pcg_update_convergence_p_prepare_next(p.view(),
                                                               z.cview(),
                                                               rz_old.view(),
                                                               rz_new.view(),
                                                               beta.view(),
                                                               rz_old_next.view(),
                                                               rz_new_next.view(),
                                                               status.view(),
                                                               check_state.view(),
                                                               params.view(),
                                                               1,
                                                               stream);
        }
        else
        {
            launch_fused_pcg_update_convergence(rz_old.view(),
                                                rz_new.view(),
                                                beta.view(),
                                                status.view(),
                                                check_state.view(),
                                                params.view(),
                                                1,
                                                stream);
            launch_fused_pcg_update_p_prepare_next(p.view(),
                                                   z.cview(),
                                                   beta.view(),
                                                   rz_new.view(),
                                                   rz_old_next.view(),
                                                   rz_new_next.view(),
                                                   status.view(),
                                                   params.view(),
                                                   1,
                                                   stream);
        }
        checkCudaErrors(cudaStreamEndCapture(stream, &graph.graph));
        size_t graph_node_count = 0;
        checkCudaErrors(cudaGraphGetNodes(graph.graph, nullptr, &graph_node_count));
        checkCudaErrors(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
        checkCudaErrors(cudaGraphLaunch(graph.exec, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));

        Result result;
        result.p             = copy_vector(p);
        result.beta          = copy_var(beta);
        result.rz_old_next    = copy_var(rz_old_next);
        result.rz_new_next    = copy_var(rz_new_next);
        result.status         = copy_var(status);
        result.check_state    = copy_var(check_state);
        result.graph_nodes    = graph_node_count;
        checkCudaErrors(cudaStreamDestroy(stream));
        return result;
    };

    const auto require_exact = [](const Result& fused, const Result& legacy)
    {
        REQUIRE(fused.p == legacy.p);
        REQUIRE(fused.beta == legacy.beta);
        REQUIRE(fused.rz_old_next == legacy.rz_old_next);
        REQUIRE(fused.rz_new_next == legacy.rz_new_next);
        REQUIRE(fused.status == legacy.status);
        REQUIRE(fused.check_state.rz == legacy.check_state.rz);
        REQUIRE(fused.check_state.status == legacy.check_state.status);
        REQUIRE(fused.check_state.iteration_in_chunk
                == legacy.check_state.iteration_in_chunk);
    };

    for(const SizeT vector_size :
        {SizeT{0}, SizeT{1}, SizeT{63}, SizeT{64}, SizeT{65}, SizeT{420}})
    {
        DYNAMIC_SECTION("running n=" << vector_size)
        {
            const auto legacy = run(false, vector_size, 0.5, 10);
            const auto fused  = run(true, vector_size, 0.5, 10);
            require_exact(fused, legacy);
            REQUIRE(fused.graph_nodes == 1);
            REQUIRE(legacy.graph_nodes == (vector_size == 0 ? 1 : 2));
        }
        DYNAMIC_SECTION("converged n=" << vector_size)
        {
            const auto legacy = run(false, vector_size, 1.0, 10);
            const auto fused  = run(true, vector_size, 1.0, 10);
            require_exact(fused, legacy);
        }
        DYNAMIC_SECTION("inactive n=" << vector_size)
        {
            const auto legacy = run(false, vector_size, 0.5, 0);
            const auto fused  = run(true, vector_size, 0.5, 0);
            require_exact(fused, legacy);
        }
    }
}

TEST_CASE("fused_pcg_graph_5_and_10_match_reference", "[cuda][fused_pcg][graph]")
{
    FullGraphFixture fixture;

    SECTION("one_iteration")
    {
        fixture.reset(-1.0);
        auto graph = fixture.capture(1, 0);
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(1), 1e-12, 1e-12);
    }

    SECTION("graph_10")
    {
        fixture.reset(-1.0);
        auto graph = fixture.capture(10, 0);
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(10), 1e-11, 1e-11);
    }

    SECTION("graph_5_even_then_odd")
    {
        fixture.reset(-1.0);
        auto even = fixture.capture(5, 0);
        auto odd  = fixture.capture(5, 1);
        checkCudaErrors(cudaGraphLaunch(even.exec, fixture.stream));
        checkCudaErrors(cudaGraphLaunch(odd.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(10), 1e-11, 1e-11);
    }

    SECTION("configurable_even_interval_reuses_one_parity")
    {
        fixture.reset(-1.0);
        auto graph = fixture.capture(4, 0);
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(8), 1e-11, 1e-11);
    }

    SECTION("configurable_odd_interval_uses_two_parities")
    {
        fixture.reset(-1.0);
        auto even = fixture.capture(3, 0);
        auto odd  = fixture.capture(3, 1);
        checkCudaErrors(cudaGraphLaunch(even.exec, fixture.stream));
        checkCudaErrors(cudaGraphLaunch(odd.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(6), 1e-11, 1e-11);
    }

    SECTION("device_active_iteration_count_handles_a_partial_final_chunk")
    {
        fixture.reset(-1.0);
        auto host_params              = copy_var(fixture.params);
        host_params.active_iterations = 3;
        fixture.params                = host_params;
        auto graph                    = fixture.capture(10, 0);
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(3), 1e-11, 1e-11);
    }

    SECTION("early_convergence_freezes_state_and_reports_iteration_one")
    {
        fixture.diagonal.assign(FullGraphFixture::ScalarCount, 1.0);
        std::vector<Matrix3> values(FullGraphFixture::BlockCount, Matrix3::Identity());
        fixture.A.values().copy_from(values.data());
        fixture.reset(1e-20);
        auto graph = fixture.capture(10, 0);
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.rhs, 1e-13, 1e-13);
        const auto terminal = copy_var(fixture.check_state);
        REQUIRE(terminal.status == static_cast<IndexT>(FusedPcgStatus::Converged));
        REQUIRE(terminal.iteration_in_chunk == 1);
    }
}

TEST_CASE("fused_pcg_graph_5_and_10_match_on_coupled_sparse_matrix", "[cuda][fused_pcg][graph][coupled]")
{
    FullGraphFixture fixture{true};
    const auto       reference = fixture.cpu_reference(10);

    auto graph_10     = fixture.capture(10, 0);
    auto graph_5_even = fixture.capture(5, 0);
    auto graph_5_odd  = fixture.capture(5, 1);

    fixture.reset(-1.0);
    for(int iteration = 1; iteration <= 10; ++iteration)
        fixture.launch_iteration((iteration - 1) & 1, iteration);
    checkCudaErrors(cudaStreamSynchronize(fixture.stream));
    const auto x_sequential = copy_vector(fixture.x);
    const auto sequential   = fixture.snapshot();

    fixture.reset(-1.0);
    checkCudaErrors(cudaGraphLaunch(graph_10.exec, fixture.stream));
    checkCudaErrors(cudaStreamSynchronize(fixture.stream));
    const auto x_graph_10     = copy_vector(fixture.x);
    const auto graph_10_state = fixture.snapshot();

    fixture.reset(-1.0);
    checkCudaErrors(cudaGraphLaunch(graph_5_even.exec, fixture.stream));
    checkCudaErrors(cudaGraphLaunch(graph_5_odd.exec, fixture.stream));
    checkCudaErrors(cudaStreamSynchronize(fixture.stream));
    const auto x_graph_5     = copy_vector(fixture.x);
    const auto graph_5_state = fixture.snapshot();

    require_near(x_graph_10, reference, 1e-11, 1e-11);
    require_near(x_graph_5, reference, 1e-11, 1e-11);
    require_near(x_graph_10, x_sequential, 1e-12, 1e-12);
    require_near(x_graph_5, x_graph_10, 1e-12, 1e-12);
    require_snapshot_near(graph_10_state, sequential);
    // A five-iteration Graph reports an iteration local to its current host-check
    // chunk.  The second Graph therefore reports 5 while the ten-iteration Graph
    // reports 10.  The host adds the completed first chunk (5), so both expose
    // the same effective terminal iteration, 10.  All numerical state must still
    // match exactly within the fixed atomic-reduction tolerance above.
    require_snapshot_near(graph_5_state, graph_10_state, false);
    REQUIRE(graph_5_state.check_state.iteration_in_chunk == 5);
    REQUIRE(graph_10_state.check_state.iteration_in_chunk == 10);
    REQUIRE(5 + graph_5_state.check_state.iteration_in_chunk
            == graph_10_state.check_state.iteration_in_chunk);
}

TEST_CASE("actual_abd_fem_fused_preconditioners_match_legacy_for_forced_iterations",
          "[cuda][fused_pcg][graph][preconditioner][legacy]")
{
    for(const int iterations : {1, 5, 10})
    {
        DYNAMIC_SECTION("iterations=" << iterations)
        {
            ActualDiagonalPreconditionerFixture fixture;

            fixture.reset();
            fixture.run_legacy(iterations);
            const auto legacy = fixture.snapshot();

            auto graph = fixture.capture(iterations);
            fixture.reset();
            checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
            checkCudaErrors(cudaStreamSynchronize(fixture.stream));
            const auto fused = fixture.snapshot();

            // The old reduction and the fused ABD/FEM reductions accumulate in
            // a different order. All vectors and all scalar state are compared;
            // the tolerance only accounts for double-precision atomic ordering.
            require_near(fused.x, legacy.x, 1e-10, 1e-10);
            require_near(fused.r, legacy.r, 1e-10, 1e-10);
            require_near(fused.z, legacy.z, 1e-10, 1e-10);
            require_near(fused.p, legacy.p, 1e-10, 1e-10);
            for(int slot = 0; slot < 2; ++slot)
            {
                require_near(fused.Ap[slot], legacy.Ap[slot], 1e-10, 1e-10);
                REQUIRE(fused.pAp[slot] == Catch::Approx(legacy.pAp[slot]).margin(1e-9));
                REQUIRE(fused.rz_old[slot]
                        == Catch::Approx(legacy.rz_old[slot]).margin(1e-9));
                REQUIRE(fused.rz_new[slot]
                        == Catch::Approx(legacy.rz_new[slot]).margin(1e-9));
            }
            REQUIRE(fused.beta == Catch::Approx(legacy.beta).margin(1e-10));
            REQUIRE(fused.status == legacy.status);
            REQUIRE(fused.check_state.status == legacy.check_state.status);
            REQUIRE(fused.check_state.iteration_in_chunk == legacy.check_state.iteration_in_chunk);
            REQUIRE(fused.check_state.rz
                    == Catch::Approx(legacy.check_state.rz).margin(1e-9));
            REQUIRE(fused.status == static_cast<IndexT>(FusedPcgStatus::Running));
            REQUIRE(fused.check_state.iteration_in_chunk == iterations);
        }
    }
}

TEST_CASE("full_abd_fused_update_apply_dot_matches_dense_reference",
          "[cuda][fused_pcg][graph][preconditioner][full_abd_graph]")
{
    constexpr int ScalarCount = 24;

    std::vector<Float> full_inverse(ScalarCount * ScalarCount, 0.0);
    for(int row = 0; row < ScalarCount; ++row)
    {
        full_inverse[row + row * ScalarCount] = 1.5 + 0.01 * row;
        for(int col = 0; col < row; ++col)
        {
            const Float coupling = ((row + col) & 1) == 0 ? 0.002 : -0.002;
            full_inverse[row + col * ScalarCount] = coupling;
            full_inverse[col + row * ScalarCount] = coupling;
        }
    }

    std::vector<Float> host_x(ScalarCount);
    std::vector<Float> host_p(ScalarCount);
    std::vector<Float> host_r(ScalarCount);
    std::vector<Float> host_Ap(ScalarCount);
    for(int i = 0; i < ScalarCount; ++i)
    {
        host_x[i]  = 0.01 * (i + 1);
        host_p[i]  = 0.02 * (1 + i % 7);
        host_r[i]  = 0.15 + 0.003 * i;
        host_Ap[i] = 0.04 + 0.002 * (i % 5);
    }

    constexpr Float    RzOld = 4.0;
    constexpr Float    PAp   = 8.0;
    constexpr Float    Alpha = RzOld / PAp;
    std::vector<Float> expected_x(ScalarCount);
    std::vector<Float> expected_r(ScalarCount);
    std::vector<Float> expected_z(ScalarCount, 0.0);
    for(int i = 0; i < ScalarCount; ++i)
    {
        expected_x[i] = host_x[i] + Alpha * host_p[i];
        expected_r[i] = host_r[i] - Alpha * host_Ap[i];
    }
    for(int row = 0; row < ScalarCount; ++row)
        for(int col = 0; col < ScalarCount; ++col)
            expected_z[row] += full_inverse[row + col * ScalarCount] * expected_r[col];
    const Float expected_rz = std::inner_product(
        expected_r.begin(), expected_r.end(), expected_z.begin(), Float{0.0});

    muda::DeviceBuffer<Float> full_inverse_device{ScalarCount * ScalarCount};
    full_inverse_device.view().copy_from(full_inverse.data());
    DeviceVector           x{ScalarCount};
    DeviceVector           p{ScalarCount};
    DeviceVector           r{ScalarCount};
    DeviceVector           Ap{ScalarCount};
    DeviceVector           z{ScalarCount};
    muda::DeviceVar<Float> rz_old{RzOld};
    muda::DeviceVar<Float> pAp{PAp};
    muda::DeviceVar<Float> rz_new{0.0};
    muda::DeviceVar<IndexT> status{static_cast<IndexT>(FusedPcgStatus::Running)};
    muda::DeviceVar<FusedPcgDeviceParams> params;

    x  = Eigen::Map<const Eigen::VectorXd>(host_x.data(), host_x.size());
    p  = Eigen::Map<const Eigen::VectorXd>(host_p.data(), host_p.size());
    r  = Eigen::Map<const Eigen::VectorXd>(host_r.data(), host_r.size());
    Ap = Eigen::Map<const Eigen::VectorXd>(host_Ap.data(), host_Ap.size());
    z.fill(0.0);
    FusedPcgDeviceParams host_params;
    host_params.tolerance         = -1.0;
    host_params.active_iterations = 1;
    params                        = host_params;
    checkCudaErrors(cudaDeviceSynchronize());

    cudaStream_t stream = nullptr;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    GraphOwner graph;
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch_fused_pcg_full_abd_update_apply_dot(full_inverse_device.view(),
                                               x.view(),
                                               p.cview(),
                                               r.view(),
                                               Ap.cview(),
                                               z.view(),
                                               rz_old.view(),
                                               pAp.view(),
                                               rz_new.view(),
                                               status.view(),
                                               params.view(),
                                               1,
                                               stream);
    checkCudaErrors(cudaStreamEndCapture(stream, &graph.graph));
    checkCudaErrors(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
    checkCudaErrors(cudaGraphLaunch(graph.exec, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    require_near(copy_vector(x), expected_x, 1e-12, 1e-12);
    require_near(copy_vector(r), expected_r, 1e-12, 1e-12);
    require_near(copy_vector(z), expected_z, 1e-11, 1e-11);
    REQUIRE(copy_var(rz_new) == Catch::Approx(expected_rz).margin(1e-10));
    REQUIRE(copy_var(status) == static_cast<IndexT>(FusedPcgStatus::Running));

    checkCudaErrors(cudaStreamDestroy(stream));
}

TEST_CASE("five_kernel_alpha_matches_legacy_division_semantics",
          "[cuda][fused_pcg][preconditioner][legacy]")
{
    ActualDiagonalPreconditionerFixture fixture;

    Eigen::VectorXd host_Ap(fixture.ScalarCount);
    for(int i = 0; i < fixture.ScalarCount; ++i)
        host_Ap[i] = Float{0.2} + Float{0.0005} * i;

    const auto run_stage = [&](bool fused, Float pAp_value)
    {
        fixture.reset();
        fixture.Ap[0]     = host_Ap;
        fixture.pAp[0]    = pAp_value;
        fixture.rz_new[0] = 0.0;

        if(fused)
        {
            launch_fused_pcg_abd_update_apply_dot(
                fixture.abd_diag_inv.view(),
                fixture.x.view().subview(0, fixture.AbdScalarCount),
                fixture.p.cview().subview(0, fixture.AbdScalarCount),
                fixture.r.view().subview(0, fixture.AbdScalarCount),
                fixture.Ap[0].cview().subview(0, fixture.AbdScalarCount),
                fixture.z.view().subview(0, fixture.AbdScalarCount),
                fixture.rz_old[0].view(),
                fixture.pAp[0].view(),
                fixture.rz_new[0].view(),
                fixture.status.view(),
                fixture.params.view(),
                1,
                nullptr);
            launch_fused_pcg_fem_update_apply_dot(
                fixture.fem_diag_inv.view(),
                fixture.x.view().subview(fixture.AbdScalarCount, fixture.FemScalarCount),
                fixture.p.cview().subview(fixture.AbdScalarCount, fixture.FemScalarCount),
                fixture.r.view().subview(fixture.AbdScalarCount, fixture.FemScalarCount),
                fixture.Ap[0].cview().subview(fixture.AbdScalarCount, fixture.FemScalarCount),
                fixture.z.view().subview(fixture.AbdScalarCount, fixture.FemScalarCount),
                fixture.rz_old[0].view(),
                fixture.pAp[0].view(),
                fixture.rz_new[0].view(),
                fixture.status.view(),
                fixture.params.view(),
                1,
                nullptr);
        }
        else
        {
            constexpr int VectorBlockSize = 64;
            const int vector_grid = (fixture.ScalarCount + VectorBlockSize - 1) / VectorBlockSize;
            const int dot_grid = (fixture.ScalarCount + 255) / 256;
            legacy_alpha_oracle_kernel<<<1, 1>>>(fixture.rz_old[0].data(),
                                                 fixture.pAp[0].data(),
                                                 fixture.alpha.data(),
                                                 fixture.status.data());
            legacy_update_xr_kernel<<<vector_grid, VectorBlockSize>>>(
                fixture.x.view().data(),
                fixture.p.cview().data(),
                fixture.r.view().data(),
                fixture.Ap[0].cview().data(),
                fixture.ScalarCount,
                fixture.alpha.data(),
                fixture.status.data());
            launch_abd_diag_preconditioner_apply(
                fixture.abd_diag_inv.view(),
                fixture.r.cview().subview(0, fixture.AbdScalarCount),
                fixture.z.view().subview(0, fixture.AbdScalarCount),
                fixture.status.view());
            launch_fem_diag_preconditioner_apply(
                fixture.fem_diag_inv.view(),
                fixture.r.cview().subview(fixture.AbdScalarCount, fixture.FemScalarCount),
                fixture.z.view().subview(fixture.AbdScalarCount, fixture.FemScalarCount),
                fixture.status.view());
            legacy_dot_kernel<<<dot_grid, 256>>>(fixture.r.cview().data(),
                                                 fixture.z.cview().data(),
                                                 fixture.ScalarCount,
                                                 fixture.rz_new[0].data(),
                                                 fixture.status.data());
        }
        checkCudaErrors(cudaDeviceSynchronize());
        return fixture.snapshot();
    };

    SECTION("negative pAp remains a finite negative-alpha update")
    {
        const auto legacy = run_stage(false, -1.0);
        const auto fused  = run_stage(true, -1.0);
        require_near(fused.x, legacy.x, 1e-12, 1e-12);
        require_near(fused.r, legacy.r, 1e-12, 1e-12);
        require_near(fused.z, legacy.z, 1e-12, 1e-12);
        REQUIRE(fused.rz_new[0] == Catch::Approx(legacy.rz_new[0]).margin(1e-9));
        REQUIRE(fused.status == legacy.status);
        REQUIRE(fused.status == static_cast<IndexT>(FusedPcgStatus::Running));
    }

    SECTION("zero pAp propagates the same non-finite classifications")
    {
        const auto legacy = run_stage(false, 0.0);
        const auto fused  = run_stage(true, 0.0);
        const auto require_same_classification = [](const auto& lhs, const auto& rhs)
        {
            REQUIRE(lhs.size() == rhs.size());
            for(size_t i = 0; i < lhs.size(); ++i)
            {
                REQUIRE(std::isnan(lhs[i]) == std::isnan(rhs[i]));
                REQUIRE(std::isinf(lhs[i]) == std::isinf(rhs[i]));
                if(std::isinf(lhs[i]))
                    REQUIRE(std::signbit(lhs[i]) == std::signbit(rhs[i]));
            }
        };
        require_same_classification(fused.x, legacy.x);
        require_same_classification(fused.r, legacy.r);
        require_same_classification(fused.z, legacy.z);
        REQUIRE(std::isinf(fused.rz_new[0]) == std::isinf(legacy.rz_new[0]));
        REQUIRE(std::isnan(fused.rz_new[0]) == std::isnan(legacy.rz_new[0]));
        REQUIRE(fused.status == legacy.status);
        REQUIRE(fused.status == static_cast<IndexT>(FusedPcgStatus::Running));
    }

    SECTION("NaN pAp propagates the same non-finite classifications")
    {
        const Float nan    = std::numeric_limits<Float>::quiet_NaN();
        const auto  legacy = run_stage(false, nan);
        const auto  fused  = run_stage(true, nan);
        const auto require_same_classification = [](const auto& lhs, const auto& rhs)
        {
            REQUIRE(lhs.size() == rhs.size());
            for(size_t i = 0; i < lhs.size(); ++i)
            {
                REQUIRE(std::isnan(lhs[i]) == std::isnan(rhs[i]));
                REQUIRE(std::isinf(lhs[i]) == std::isinf(rhs[i]));
            }
        };
        require_same_classification(fused.x, legacy.x);
        require_same_classification(fused.r, legacy.r);
        require_same_classification(fused.z, legacy.z);
        REQUIRE(std::isnan(fused.rz_new[0]) == std::isnan(legacy.rz_new[0]));
        REQUIRE(fused.status == legacy.status);
        REQUIRE(fused.status == static_cast<IndexT>(FusedPcgStatus::Running));
    }
}

TEST_CASE("five_kernel_nonfinite_values_reach_the_legacy_host_check_boundary",
          "[cuda][fused_pcg][graph][legacy][nonfinite]")
{
    const auto run_chunk = [](int graph_iterations, int active_iterations)
    {
        FullGraphFixture fixture;
        std::vector<Matrix3> values(FullGraphFixture::BlockCount, Matrix3::Identity());
        values[0](0, 0) = std::numeric_limits<Float>::quiet_NaN();
        fixture.A.values().copy_from(values.data());
        fixture.reset(-1.0);

        auto host_params              = copy_var(fixture.params);
        host_params.active_iterations = active_iterations;
        fixture.params                = host_params;

        auto graph = fixture.capture(graph_iterations, 0);
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));

        const auto state = fixture.snapshot();
        REQUIRE(state.status == static_cast<IndexT>(FusedPcgStatus::Running));
        REQUIRE(state.check_state.status == static_cast<IndexT>(FusedPcgStatus::Running));
        REQUIRE(state.check_state.iteration_in_chunk == active_iterations);
        REQUIRE_FALSE(std::isfinite(state.check_state.rz));
        REQUIRE_FALSE(std::isfinite(state.pAp[(active_iterations - 1) & 1]));
    };

    SECTION("Graph5 reports the non-finite residual at iteration five")
    {
        run_chunk(5, 5);
    }

    SECTION("Graph10 reports the non-finite residual at iteration ten")
    {
        run_chunk(10, 10);
    }

    SECTION("a partial Graph10 reports at its third active iteration")
    {
        run_chunk(10, 3);
    }
}

TEST_CASE("fused_pcg_graph_reads_dynamic_tolerance_without_recapture",
          "[cuda][fused_pcg][graph][dynamic][tolerance]")
{
    FullGraphFixture fixture{true};
    auto             graph              = fixture.capture(10, 0);
    const auto       graph_exec_address = graph.exec;

    fixture.reset(-1.0);
    checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
    checkCudaErrors(cudaStreamSynchronize(fixture.stream));
    const auto tight = fixture.snapshot();
    REQUIRE(tight.status == static_cast<IndexT>(FusedPcgStatus::Running));
    REQUIRE(tight.check_state.iteration_in_chunk == 10);

    fixture.reset(std::numeric_limits<Float>::max());
    checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
    checkCudaErrors(cudaStreamSynchronize(fixture.stream));
    const auto loose = fixture.snapshot();
    REQUIRE(loose.status == static_cast<IndexT>(FusedPcgStatus::Converged));
    REQUIRE(loose.check_state.status == static_cast<IndexT>(FusedPcgStatus::Converged));
    REQUIRE(loose.check_state.iteration_in_chunk == 1);
    REQUIRE(graph.exec == graph_exec_address);
}

TEST_CASE("fused_pcg_complete_graph_reuses_one_exec_across_reset_chunks",
          "[cuda][fused_pcg][graph][repeat]")
{
    constexpr int    TrialCount = 32;
    FullGraphFixture fixture{true};
    FullGraphFixture reference_fixture{true};
    reference_fixture.reset(-1.0);
    for(int iteration = 1; iteration <= 10; ++iteration)
        reference_fixture.launch_iteration((iteration - 1) & 1, iteration);
    checkCudaErrors(cudaStreamSynchronize(reference_fixture.stream));
    const auto reference = reference_fixture.snapshot();

    auto       graph             = fixture.capture(10, 0);
    const auto exec              = graph.exec;
    const auto signature         = fixture.graph_signature();
    const int  graph_build_count = 1;

    for(int trial = 0; trial < TrialCount; ++trial)
    {
        INFO("trial=" << trial);
        fixture.reset(-1.0);
        REQUIRE(fixture.graph_signature() == signature);
        REQUIRE(graph.exec == exec);
        checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_snapshot_near(fixture.snapshot(), reference);
        const auto check = copy_var(fixture.check_state);
        REQUIRE(copy_var(fixture.status) == static_cast<IndexT>(FusedPcgStatus::Running));
        REQUIRE(check.status == static_cast<IndexT>(FusedPcgStatus::Running));
        REQUIRE(check.iteration_in_chunk == 10);
    }
    REQUIRE(graph_build_count == 1);
}

TEST_CASE("fused_pcg_two_graph_execs_remain_isolated_across_interleaved_runs",
          "[cuda][fused_pcg][graph][multi_instance]")
{
    constexpr int    TrialCount = 32;
    FullGraphFixture running_fixture{true};
    FullGraphFixture converged_fixture{false};
    FullGraphFixture running_reference_fixture{true};
    FullGraphFixture converged_reference_fixture{false};

    running_reference_fixture.reset(-1.0);
    converged_reference_fixture.reset(std::numeric_limits<Float>::max());
    for(int iteration = 1; iteration <= 10; ++iteration)
    {
        running_reference_fixture.launch_iteration((iteration - 1) & 1, iteration);
        converged_reference_fixture.launch_iteration((iteration - 1) & 1, iteration);
    }
    checkCudaErrors(cudaStreamSynchronize(running_reference_fixture.stream));
    checkCudaErrors(cudaStreamSynchronize(converged_reference_fixture.stream));
    const auto running_reference   = running_reference_fixture.snapshot();
    const auto converged_reference = converged_reference_fixture.snapshot();

    auto       running_graph        = running_fixture.capture(10, 0);
    auto       converged_graph      = converged_fixture.capture(10, 0);
    const auto running_exec         = running_graph.exec;
    const auto converged_exec       = converged_graph.exec;
    const auto running_signature    = running_fixture.graph_signature();
    const auto converged_signature  = converged_fixture.graph_signature();
    const int  running_generation   = 1;
    const int  converged_generation = 1;

    REQUIRE(running_exec != converged_exec);
    REQUIRE(running_fixture.params.data() != converged_fixture.params.data());
    REQUIRE(running_fixture.status.data() != converged_fixture.status.data());

    for(int trial = 0; trial < TrialCount; ++trial)
    {
        INFO("trial=" << trial);
        running_fixture.reset(-1.0);
        converged_fixture.reset(std::numeric_limits<Float>::max());

        if((trial & 1) == 0)
        {
            checkCudaErrors(cudaGraphLaunch(running_graph.exec, running_fixture.stream));
            checkCudaErrors(
                cudaGraphLaunch(converged_graph.exec, converged_fixture.stream));
        }
        else
        {
            checkCudaErrors(
                cudaGraphLaunch(converged_graph.exec, converged_fixture.stream));
            checkCudaErrors(cudaGraphLaunch(running_graph.exec, running_fixture.stream));
        }
        checkCudaErrors(cudaStreamSynchronize(running_fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(converged_fixture.stream));

        REQUIRE(running_graph.exec == running_exec);
        REQUIRE(converged_graph.exec == converged_exec);
        REQUIRE(running_fixture.graph_signature() == running_signature);
        REQUIRE(converged_fixture.graph_signature() == converged_signature);
        require_snapshot_near(running_fixture.snapshot(), running_reference);
        require_snapshot_near(converged_fixture.snapshot(), converged_reference);
        const auto running_check   = copy_var(running_fixture.check_state);
        const auto converged_check = copy_var(converged_fixture.check_state);
        REQUIRE(running_check.status == static_cast<IndexT>(FusedPcgStatus::Running));
        REQUIRE(running_check.iteration_in_chunk == 10);
        REQUIRE(converged_check.status == static_cast<IndexT>(FusedPcgStatus::Converged));
        REQUIRE(converged_check.iteration_in_chunk == 1);
    }
    REQUIRE(running_generation == 1);
    REQUIRE(converged_generation == 1);
}

TEST_CASE("fused_pcg_logical_dof_change_within_capacity_rebuilds_and_guards_tail",
          "[cuda][fused_pcg][graph][dynamic][dof]")
{
    constexpr int    SmallDof = 30;
    constexpr int    LargeDof = FullGraphFixture::ScalarCount;
    FullGraphFixture fixture{false};
    REQUIRE(fixture.x.capacity() >= LargeDof);

    std::unique_ptr<GraphOwner> graph;
    FusedPcgGraphSignature      cached_signature;
    bool                        signature_valid = false;
    int                         rebuild_count   = 0;

    auto ensure_graph = [&]
    {
        const auto current = fixture.graph_signature();
        if(!signature_valid || current != cached_signature)
        {
            graph.reset();
            graph = std::make_unique<GraphOwner>(fixture.capture(10, 0));
            cached_signature = current;
            signature_valid  = true;
            ++rebuild_count;
        }
    };

    auto run_small_and_check_tail = [&]
    {
        fixture.reset(-1.0);
        ensure_graph();
        for(auto* vector : {&fixture.x, &fixture.r, &fixture.z, &fixture.p})
            poison_vector_tail(*vector, SmallDof, LargeDof);
        for(auto& vector : fixture.Ap)
            poison_vector_tail(vector, SmallDof, LargeDof);
        checkCudaErrors(cudaGraphLaunch(graph->exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(10), 1e-11, 1e-11);
        for(const auto* vector : {&fixture.x, &fixture.r, &fixture.z, &fixture.p})
            require_poisoned_tail(*vector, SmallDof, LargeDof);
        for(const auto& vector : fixture.Ap)
            require_poisoned_tail(vector, SmallDof, LargeDof);
    };

    const auto allocation_pointer = fixture.x.cview().origin_data();
    fixture.resize_logical_dof(SmallDof);
    run_small_and_check_tail();
    const auto small_signature = fixture.graph_signature();
    REQUIRE(rebuild_count == 1);
    REQUIRE(fixture.x.cview().origin_data() == allocation_pointer);

    fixture.resize_logical_dof(LargeDof);
    fixture.reset(-1.0);
    ensure_graph();
    REQUIRE(rebuild_count == 2);
    REQUIRE(fixture.graph_signature() != small_signature);
    REQUIRE(fixture.x.cview().origin_data() == allocation_pointer);
    checkCudaErrors(cudaGraphLaunch(graph->exec, fixture.stream));
    checkCudaErrors(cudaStreamSynchronize(fixture.stream));
    require_near(copy_vector(fixture.x), fixture.cpu_reference(10), 1e-11, 1e-11);

    fixture.resize_logical_dof(SmallDof);
    run_small_and_check_tail();
    REQUIRE(rebuild_count == 3);
    REQUIRE(fixture.x.cview().origin_data() == allocation_pointer);
}

TEST_CASE("fused_pcg_preconditioner_policy_rejects_global_duplicate_and_unsupported_local",
          "[cuda][fused_pcg][fallback][policy]")
{
    EngineCreateInfo create_info;
    create_info.workspace               = "";
    create_info.config["gpu"]["device"] = 0;
    SimEngine engine{&create_info};

    SECTION("global_preconditioner_forces_legacy_fallback")
    {
        GlobalLinearSystem::Impl   impl;
        PolicyGlobalPreconditioner global{engine};
        impl.global_preconditioner = global;
        REQUIRE_FALSE(impl.supports_fused_pcg());
    }

    SECTION("two_local_preconditioners_for_one_subsystem_force_legacy_fallback")
    {
        GlobalLinearSystem::Impl  impl;
        PolicyDiagSubsystem       subsystem{engine};
        PolicyLocalPreconditioner first{engine, true};
        PolicyLocalPreconditioner second{engine, true};

        subsystem.*get_private_member(DiagLinearSubsystemIndexMember{}) = 0;
        first.*get_private_member(LocalPreconditionerSubsystemMember{}) = &subsystem;
        second.*get_private_member(LocalPreconditionerSubsystemMember{}) = &subsystem;
        impl.diag_subsystems.register_sim_system(subsystem);
        impl.local_preconditioners.register_sim_system(first);
        impl.local_preconditioners.register_sim_system(second);
        REQUIRE_FALSE(impl.supports_fused_pcg());
    }

    SECTION("unsupported_local_preconditioner_forces_legacy_fallback")
    {
        GlobalLinearSystem::Impl  impl;
        PolicyDiagSubsystem       subsystem{engine};
        PolicyLocalPreconditioner local{engine, false};

        subsystem.*get_private_member(DiagLinearSubsystemIndexMember{}) = 0;
        local.*get_private_member(LocalPreconditionerSubsystemMember{}) = &subsystem;
        impl.diag_subsystems.register_sim_system(subsystem);
        impl.local_preconditioners.register_sim_system(local);
        REQUIRE_FALSE(impl.supports_fused_pcg());
    }

    SECTION("one_supported_local_preconditioner_per_subsystem_allows_graph")
    {
        GlobalLinearSystem::Impl  impl;
        PolicyDiagSubsystem       subsystem{engine};
        PolicyLocalPreconditioner local{engine, true};

        subsystem.*get_private_member(DiagLinearSubsystemIndexMember{}) = 0;
        local.*get_private_member(LocalPreconditionerSubsystemMember{}) = &subsystem;
        impl.diag_subsystems.register_sim_system(subsystem);
        impl.local_preconditioners.register_sim_system(local);
        REQUIRE(impl.supports_fused_pcg());
    }
}

TEST_CASE("fused_pcg_graph_freezes_after_each_possible_terminal_iteration",
          "[cuda][fused_pcg][graph][terminal]")
{
    FullGraphFixture fixture{true};
    for(int terminal_iteration = 1; terminal_iteration <= 10; ++terminal_iteration)
    {
        DYNAMIC_SECTION("terminal_iteration=" << terminal_iteration)
        {
            fixture.reset(-1.0);
            for(int iteration = 1; iteration <= 10; ++iteration)
                fixture.launch_iteration((iteration - 1) & 1, iteration, terminal_iteration);
            checkCudaErrors(cudaStreamSynchronize(fixture.stream));
            const auto sequential = fixture.snapshot();

            fixture.reset(-1.0);
            auto graph = fixture.capture_with_forced_terminal(10, terminal_iteration);
            checkCudaErrors(cudaGraphLaunch(graph.exec, fixture.stream));
            checkCudaErrors(cudaStreamSynchronize(fixture.stream));
            require_snapshot_near(fixture.snapshot(), sequential);
            const auto terminal = copy_var(fixture.check_state);
            REQUIRE(terminal.status == static_cast<IndexT>(FusedPcgStatus::Converged));
            REQUIRE(terminal.iteration_in_chunk == terminal_iteration);
        }
    }
}

TEST_CASE("fused_pcg_graph_signature_tracks_bound_addresses_and_capacity",
          "[cuda][fused_pcg][graph][dynamic]")
{
    FullGraphFixture fixture{true};

    std::unique_ptr<GraphOwner> graph;
    FusedPcgGraphSignature      cached_signature;
    bool                        signature_valid = false;
    int                         rebuild_count   = 0;

    auto ensure_graph = [&]
    {
        const auto current_signature = fixture.graph_signature();
        if(!signature_valid || current_signature != cached_signature)
        {
            // Do not retain an executable Graph whose kernel arguments contain
            // addresses invalidated by a real reserve/reallocation operation.
            graph.reset();
            graph = std::make_unique<GraphOwner>(fixture.capture(10, 0));
            cached_signature = current_signature;
            signature_valid  = true;
            ++rebuild_count;
        }
    };

    auto run_and_check = [&]
    {
        fixture.reset(-1.0);
        ensure_graph();
        checkCudaErrors(cudaGraphLaunch(graph->exec, fixture.stream));
        checkCudaErrors(cudaStreamSynchronize(fixture.stream));
        require_near(copy_vector(fixture.x), fixture.cpu_reference(10), 1e-11, 1e-11);
    };

    run_and_check();
    REQUIRE(rebuild_count == 1);

    // Resetting values does not change any captured address or capacity, so the
    // same executable Graph must be reused.
    run_and_check();
    REQUIRE(rebuild_count == 1);

    const auto before_growth = fixture.graph_signature();
    const auto old_row_ptr   = before_growth.matrix_rows;
    const auto old_col_ptr   = before_growth.matrix_cols;
    const auto old_value_ptr = before_growth.matrix_values;
    const auto old_x_ptr     = before_growth.x;
    const auto old_capacity  = before_growth.triplet_bucket;

    fixture.force_bound_storage_reallocation();
    const auto after_growth = fixture.graph_signature();

    REQUIRE(after_growth != before_growth);
    REQUIRE(after_growth.matrix_rows != old_row_ptr);
    REQUIRE(after_growth.matrix_cols != old_col_ptr);
    REQUIRE(after_growth.matrix_values != old_value_ptr);
    REQUIRE(after_growth.x != old_x_ptr);
    REQUIRE(after_growth.triplet_bucket > old_capacity);

    // The stale executable is destroyed and recaptured before launch. The new
    // Graph must remain numerically equivalent after real address changes.
    run_and_check();
    REQUIRE(rebuild_count == 2);

    run_and_check();
    REQUIRE(rebuild_count == 2);
}

TEST_CASE("fused_spmv_reuses_one_graph_for_dynamic_logical_counts", "[cuda][fused_pcg][graph][dynamic]")
{
    constexpr int            BlockRows = 96;
    constexpr int            MaxCount  = 513;
    const std::array<int, 5> counts{497, 501, 505, 509, 513};

    std::vector<int>     rows;
    std::vector<int>     cols;
    std::vector<Matrix3> values;
    rows.reserve(MaxCount);
    cols.reserve(MaxCount);
    values.reserve(MaxCount);
    for(int row = 0; row < BlockRows && static_cast<int>(rows.size()) < MaxCount; ++row)
    {
        for(int col = row; col < BlockRows && static_cast<int>(rows.size()) < MaxCount; ++col)
        {
            rows.push_back(row);
            cols.push_back(col);
            values.push_back((row == col ? Float{2.0} : Float{1e-4}) * Matrix3::Identity());
        }
    }
    REQUIRE(static_cast<int>(rows.size()) == MaxCount);

    muda::DeviceBCOOMatrix<Float, 3> matrix;
    matrix.resize(BlockRows, BlockRows, MaxCount);
    matrix.row_indices().copy_from(rows.data());
    matrix.col_indices().copy_from(cols.data());
    matrix.values().copy_from(values.data());

    const int       scalar_count = BlockRows * 3;
    Eigen::VectorXd host_p(scalar_count);
    for(int i = 0; i < scalar_count; ++i)
        host_p[i] = Float{0.1} + Float{1e-4} * static_cast<Float>(i % 97);

    DeviceVector p{scalar_count};
    DeviceVector old_Ap{scalar_count};
    DeviceVector graph_Ap{scalar_count};
    DeviceVector next_Ap{scalar_count};
    p = host_p;
    muda::DeviceVar<Float> old_pAp{0.0};
    muda::DeviceVar<Float> graph_pAp{0.0};
    muda::DeviceVar<Float> next_pAp{0.0};
    muda::DeviceVar<IndexT> status{static_cast<IndexT>(FusedPcgStatus::Running)};
    muda::DeviceVar<FusedPcgDeviceParams> params;

    Spmv         spmv;
    cudaStream_t stream = nullptr;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    FusedPcgDeviceParams host_params;
    host_params.active_iterations = 1;
    host_params.triplet_count     = MaxCount;
    params                        = host_params;
    checkCudaErrors(cudaDeviceSynchronize());

    GraphOwner graph;
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    spmv.rbk_sym_spmv_dot_pipelined(matrix.cview(),
                                    p.cview(),
                                    graph_Ap.view(),
                                    graph_pAp.view(),
                                    next_Ap.view(),
                                    next_pAp.view(),
                                    status.view(),
                                    params.view(),
                                    1,
                                    MaxCount,
                                    stream);
    checkCudaErrors(cudaStreamEndCapture(stream, &graph.graph));
    checkCudaErrors(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));

    constexpr int                      TrialCount = 32;
    std::mt19937                       random_engine{0x50434725u};
    std::uniform_int_distribution<int> count_index_distribution{
        0, static_cast<int>(counts.size()) - 1};
    std::array<int, counts.size()> seen{};
    const int                      graph_build_count = 1;

    for(int trial = 0; trial < TrialCount; ++trial)
    {
        // The first five trials cover every representative logical count;
        // all remaining trials use a fixed-seed random order.
        const int count_index = trial < static_cast<int>(counts.size()) ?
                                    trial :
                                    count_index_distribution(random_engine);
        const int count       = counts[count_index];
        ++seen[count_index];
        INFO("trial=" << trial << ", triplet_count=" << count);
        host_params.triplet_count = count;
        params                    = host_params;
        status = static_cast<IndexT>(FusedPcgStatus::Running);
        graph_Ap.fill(0.0);
        graph_pAp = 0.0;
        next_Ap.fill(7.0);
        next_pAp = 9.0;

        old_Ap.fill(0.0);
        old_pAp = 0.0;
        spmv.rbk_sym_spmv_dot(1.0,
                              matrix.cview().subview(0, count),
                              p.cview(),
                              0.0,
                              old_Ap.view(),
                              old_pAp.view());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGraphLaunch(graph.exec, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));

        require_near(copy_vector(graph_Ap), copy_vector(old_Ap), 1e-11, 1e-12);
        REQUIRE(copy_var(graph_pAp) == Catch::Approx(copy_var(old_pAp)).margin(1e-10));
        require_near(copy_vector(next_Ap), std::vector<Float>(scalar_count, 0.0), 0.0, 0.0);
        REQUIRE(copy_var(next_pAp) == Catch::Approx(0.0));
    }

    REQUIRE(graph_build_count == 1);
    for(const int observation_count : seen)
        REQUIRE(observation_count > 0);

    checkCudaErrors(cudaStreamDestroy(stream));
}
