#include <app/app.h>

#include <linear_system/linear_fused_pcg_graph_resources.h>
#include <backends/cuda/sim_engine.h>
#include <uipc/backend/engine_create_info.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/uipc.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace uipc::backend::cuda
{
class LinearFusedPcgTestAccess final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    ~LinearFusedPcgTestAccess() override
    {
        if(s_current == this)
            s_current = nullptr;
    }

    static LinearFusedPcgTestAccess& current()
    {
        if(!s_current || !s_current->m_solver)
            throw std::runtime_error("LinearFusedPCG test access is unavailable");
        return *s_current;
    }

    bool graph_resources_created() const noexcept
    {
        return m_solver->graph_resources != nullptr;
    }

    bool graph_exec_created() const noexcept
    {
        return graph_resources_created()
               && m_solver->graph_resources->graph_start_slot_0 != nullptr;
    }

    bool odd_graph_exec_created() const noexcept
    {
        return graph_resources_created()
               && m_solver->graph_resources->graph_start_slot_1 != nullptr;
    }

    SizeT graph_generation() const noexcept
    {
        return graph_resources_created() ? m_solver->graph_resources->generation : 0;
    }

    SizeT check_interval() const noexcept { return m_solver->check_interval; }
    SizeT terminal_iteration() const noexcept
    {
        return m_solver->last_terminal_iteration;
    }
    SizeT effective_iteration() const noexcept
    {
        return m_solver->last_effective_iteration;
    }
    SizeT reported_iteration() const noexcept
    {
        return m_solver->last_reported_iteration;
    }

    void set_check_interval(SizeT value) { m_solver->check_interval = value; }

    void set_max_iter_ratio(Float value) { m_solver->max_iter_ratio = value; }

    void force_solver_vector_reallocation()
    {
        const SizeT new_capacity =
            std::max<SizeT>(m_solver->r.capacity() + 64, m_solver->r.size() + 64);
        m_solver->r.reserve(new_capacity);
    }

  private:
    void do_build() override
    {
        m_solver  = &require<LinearFusedPCG>();
        s_current = this;
    }

    static inline LinearFusedPcgTestAccess* s_current = nullptr;
    LinearFusedPCG*                         m_solver  = nullptr;
};

REGISTER_SIM_SYSTEM(LinearFusedPcgTestAccess);
}  // namespace uipc::backend::cuda

namespace
{
using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;
using uipc::backend::cuda::LinearFusedPcgTestAccess;
using CudaSimEngine = uipc::backend::cuda::SimEngine;

struct ProductionRunResult
{
    std::vector<Vector3> positions;
    bool                 graph_resources_created = false;
    bool                 graph_exec_created      = false;
    bool                 odd_graph_exec_created  = false;
    SizeT                graph_generation        = 0;
    SizeT                check_interval          = 0;
    SizeT                terminal_iteration      = 0;
    SizeT                effective_iteration     = 0;
    SizeT                reported_iteration      = 0;
};

Json production_scene_config(IndexT graph_enable, IndexT check_interval, Float tolerance = 1e-4)
{
    auto config                               = test::Scene::default_config();
    config["gravity"]                         = Vector3{0, -9.8, 0};
    config["contact"]["enable"]               = false;
    config["linear_system"]["tol_rate"]       = tolerance;
    config["linear_system"]["check_interval"] = check_interval;
    config["linear_system"]["fused_pcg"]["graph_enable"] = graph_enable;
    config["linear_system"]["fused_pcg"]["fused_preconditioner_enable"] = 1;
    return config;
}

S<SimplicialComplexSlot> add_cloth(Scene& scene, int grid_size, bool use_mas, bool fix_all_vertices = false)
{
    NeoHookeanShell shell;
    auto            object = scene.objects().create("cloth");

    constexpr Float  cloth_size = 0.4;
    const Float      spacing    = cloth_size / grid_size;
    vector<Vector3>  Vs;
    vector<Vector3i> Fs;
    for(int i = 0; i <= grid_size; ++i)
        for(int j = 0; j <= grid_size; ++j)
            Vs.push_back(Vector3{i * spacing, 0.5, j * spacing});

    for(int i = 0; i < grid_size; ++i)
    {
        for(int j = 0; j < grid_size; ++j)
        {
            const int v00 = i * (grid_size + 1) + j;
            const int v10 = (i + 1) * (grid_size + 1) + j;
            const int v01 = i * (grid_size + 1) + j + 1;
            const int v11 = (i + 1) * (grid_size + 1) + j + 1;
            Fs.push_back(Vector3i{v00, v10, v11});
            Fs.push_back(Vector3i{v00, v11, v01});
        }
    }

    auto mesh = trimesh(Vs, Fs);
    label_surface(mesh);
    if(use_mas)
        mesh_partition(mesh, 4);

    shell.apply_to(mesh, ElasticModuli2D::youngs_poisson(1.0_MPa, 0.45));
    auto is_fixed      = mesh.vertices().find<IndexT>(builtin::is_fixed);
    auto is_fixed_view = view(*is_fixed);
    if(fix_all_vertices)
        std::fill(is_fixed_view.begin(), is_fixed_view.end(), 1);
    else
    {
        is_fixed_view[0]         = 1;
        is_fixed_view[grid_size] = 1;
    }
    return object->geometries().create(mesh).geometry;
}

ProductionRunResult run_production_case(IndexT graph_enable,
                                        IndexT check_interval,
                                        bool   use_mas                = false,
                                        bool   remove_new_config_keys = false,
                                        Float  tolerance              = 1e-4,
                                        int    advances               = 1)
{
    const auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
    EngineCreateInfo create_info;
    create_info.workspace               = output_path;
    create_info.config["gpu"]["device"] = 0;
    auto   backend_engine = uipc::make_shared<CudaSimEngine>(&create_info);
    Engine engine{"cuda", backend_engine, output_path};
    World  world{engine};

    Scene scene{production_scene_config(graph_enable, check_interval, tolerance)};
    if(remove_new_config_keys)
    {
        scene.config().destroy("linear_system/check_interval");
        scene.config().destroy("linear_system/fused_pcg/graph_enable");
        scene.config().destroy("linear_system/fused_pcg/fused_preconditioner_enable");
    }
    auto cloth = add_cloth(scene, use_mas ? 4 : 2, use_mas);

    world.init(scene);
    REQUIRE(world.is_valid());
    for(int i = 0; i < advances; ++i)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }

    auto&               access = LinearFusedPcgTestAccess::current();
    ProductionRunResult result;
    const auto          positions = view(cloth->geometry().positions());
    result.positions.assign(positions.begin(), positions.end());
    result.graph_resources_created = access.graph_resources_created();
    result.graph_exec_created      = access.graph_exec_created();
    result.odd_graph_exec_created  = access.odd_graph_exec_created();
    result.graph_generation        = access.graph_generation();
    result.check_interval          = access.check_interval();
    result.terminal_iteration      = access.terminal_iteration();
    result.effective_iteration     = access.effective_iteration();
    result.reported_iteration      = access.reported_iteration();
    return result;
}

void require_positions_near(const std::vector<Vector3>& actual,
                            const std::vector<Vector3>& expected)
{
    REQUIRE(actual.size() == expected.size());
    for(SizeT i = 0; i < actual.size(); ++i)
    {
        for(int axis = 0; axis < 3; ++axis)
            REQUIRE(actual[i][axis]
                    == Catch::Approx(expected[i][axis]).margin(1e-10).epsilon(1e-8));
    }
}
}  // namespace

TEST_CASE("fused_pcg_production_default_uses_legacy_path", "[fused_pcg][production][defaults]")
{
    const auto defaults = run_production_case(0, 5);
    REQUIRE(defaults.check_interval == 5);
    REQUIRE_FALSE(defaults.graph_resources_created);
    REQUIRE_FALSE(defaults.graph_exec_created);
    REQUIRE(defaults.graph_generation == 0);

    const auto missing_keys = run_production_case(1, 10, false, true);
    REQUIRE(missing_keys.check_interval == 5);
    REQUIRE_FALSE(missing_keys.graph_resources_created);
    REQUIRE_FALSE(missing_keys.graph_exec_created);
    REQUIRE(missing_keys.graph_generation == 0);
    require_positions_near(missing_keys.positions, defaults.positions);
}

TEST_CASE("fused_pcg_production_graph5_and_graph10_match_legacy", "[fused_pcg][production][graph]")
{
    const auto legacy  = run_production_case(0, 5);
    const auto graph5  = run_production_case(1, 5);
    const auto graph10 = run_production_case(1, 10);

    REQUIRE(graph5.graph_exec_created);
    REQUIRE(graph5.odd_graph_exec_created);
    REQUIRE(graph5.graph_generation >= 1);
    REQUIRE(graph10.graph_exec_created);
    REQUIRE_FALSE(graph10.odd_graph_exec_created);
    REQUIRE(graph10.graph_generation >= 1);
    require_positions_near(graph5.positions, legacy.positions);
    require_positions_near(graph10.positions, legacy.positions);
}

TEST_CASE("fused_pcg_production_graph_cache_reuses_and_rebuilds",
          "[fused_pcg][production][graph][signature]")
{
    const auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
    EngineCreateInfo create_info;
    create_info.workspace               = output_path;
    create_info.config["gpu"]["device"] = 0;
    auto   backend_engine = uipc::make_shared<CudaSimEngine>(&create_info);
    Engine engine{"cuda", backend_engine, output_path};
    World  world{engine};
    Scene  scene{production_scene_config(1, 10)};
    add_cloth(scene, 2, false);
    world.init(scene);
    REQUIRE(world.is_valid());

    world.advance();
    REQUIRE(world.is_valid());
    auto&       access                   = LinearFusedPcgTestAccess::current();
    const SizeT generation_after_capture = access.graph_generation();
    REQUIRE(generation_after_capture >= 1);

    world.advance();
    REQUIRE(world.is_valid());
    REQUIRE(access.graph_generation() == generation_after_capture);

    access.force_solver_vector_reallocation();
    world.advance();
    REQUIRE(world.is_valid());
    const SizeT generation_after_pointer_change = access.graph_generation();
    REQUIRE(generation_after_pointer_change == generation_after_capture + 1);

    access.set_check_interval(5);
    world.advance();
    REQUIRE(world.is_valid());
    REQUIRE(access.graph_generation() == generation_after_pointer_change + 1);
    REQUIRE(access.odd_graph_exec_created());
}

TEST_CASE("fused_pcg_unsupported_preconditioner_uses_production_fallback",
          "[fused_pcg][production][fallback][mas]")
{
    const auto result = run_production_case(1, 10, true);
    REQUIRE_FALSE(result.graph_resources_created);
    REQUIRE_FALSE(result.graph_exec_created);
    REQUIRE(result.graph_generation == 0);
    REQUIRE_FALSE(result.positions.empty());
}

TEST_CASE("fused_pcg_production_zero_residual_and_partial_chunk_are_safe",
          "[fused_pcg][production][graph][edge]")
{
    SECTION("zero residual")
    {
        auto config       = production_scene_config(1, 10);
        config["gravity"] = Vector3{0, 0, 0};

        const auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
        EngineCreateInfo create_info;
        create_info.workspace               = output_path;
        create_info.config["gpu"]["device"] = 0;
        auto   backend_engine = uipc::make_shared<CudaSimEngine>(&create_info);
        Engine engine{"cuda", backend_engine, output_path};
        World  world{engine};
        Scene  scene{config};
        add_cloth(scene, 2, false, true);
        world.init(scene);
        world.advance();
        REQUIRE(world.is_valid());
        auto& access = LinearFusedPcgTestAccess::current();
        REQUIRE_FALSE(access.graph_exec_created());
        REQUIRE(access.reported_iteration() == 0);
    }

    SECTION("partial final chunk")
    {
        const auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
        EngineCreateInfo create_info;
        create_info.workspace               = output_path;
        create_info.config["gpu"]["device"] = 0;
        auto   backend_engine = uipc::make_shared<CudaSimEngine>(&create_info);
        Engine engine{"cuda", backend_engine, output_path};
        World  world{engine};
        Scene  scene{production_scene_config(1, 10, -1.0)};
        add_cloth(scene, 2, false);
        world.init(scene);

        auto& access = LinearFusedPcgTestAccess::current();
        access.set_max_iter_ratio(0.2);
        world.advance();
        REQUIRE(world.is_valid());
        REQUIRE(access.graph_exec_created());
        REQUIRE(access.reported_iteration() == 5);
        REQUIRE(access.effective_iteration() == 4);
        REQUIRE(access.terminal_iteration() == 0);
    }
}

TEST_CASE("fused_pcg_production_legacy_check_interval_compatibility",
          "[fused_pcg][production][compatibility]")
{
    SECTION("legacy zero interval preserves compatibility")
    {
        const auto result = run_production_case(0, 0);
        REQUIRE_FALSE(result.graph_resources_created);
        REQUIRE_FALSE(result.positions.empty());
    }

    SECTION("legacy large interval preserves compatibility")
    {
        const auto result = run_production_case(0, 1025);
        REQUIRE_FALSE(result.graph_resources_created);
        REQUIRE_FALSE(result.positions.empty());
    }
}
