#include <sim_engine.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <dytopo_effect_system/dytopo_distribution.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/cub/device/device_select.h>
#include <cub/iterator/counting_input_iterator.cuh>
#include <uipc/common/enumerate.h>
#include <kernel_cout.h>
#include <uipc/common/unit.h>
#include <uipc/common/zip.h>
#include <energy_component_flags.h>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::GlobalDyTopoEffectManager>
{
  public:
    static U<cuda::GlobalDyTopoEffectManager> create(cuda::SimEngine& engine)
    {
        auto dytopo_effect_enable_attr =
            engine.world().scene().config().find<IndexT>("contact/enable");
        bool dytopo_effect_enable = dytopo_effect_enable_attr->view()[0] != 0;

        auto& types = engine.world().scene().constitution_tabular().types();
        bool  has_inter_primitive_constitution =
            types.find(std::string{builtin::InterPrimitive}) != types.end();

        if(dytopo_effect_enable || has_inter_primitive_constitution)
            return make_unique<cuda::GlobalDyTopoEffectManager>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalDyTopoEffectManager);

muda::CBCOOVectorView<Float, 3> GlobalDyTopoEffectManager::gradients() const noexcept
{
    return m_impl.sorted_dytopo_effect_gradient.view();
}

muda::CBCOOMatrixView<Float, 3> GlobalDyTopoEffectManager::hessians() const noexcept
{
    return m_impl.sorted_dytopo_effect_hessian.view();
}

void GlobalDyTopoEffectManager::do_build()
{
    const auto& config = world().scene().config();

    m_impl.global_vertex_manager = require<GlobalVertexManager>();
}

void GlobalDyTopoEffectManager::Impl::init(WorldVisitor& world)
{
    // 3) reporters
    auto dytopo_effect_reporter_view = dytopo_effect_reporters.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->m_index = i;

    reporter_energy_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_gradient_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_hessian_offsets_counts.resize(dytopo_effect_reporter_view.size());

    // 4) receivers
    auto dytopo_effect_receiver_view = dytopo_effect_receivers.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->m_index = i;

    const auto receiver_count = dytopo_effect_receiver_view.size();
    receiver_classify_infos.resize(receiver_count);
    host_distribution_queries.resize(receiver_count);
    host_distribution_results.resize(receiver_count);
    distribution_queries.resize(receiver_count);
    distribution_results.resize(receiver_count);
    classified_dytopo_effect_hessians.resize(receiver_count);
    classified_dytopo_effect_gradients.resize(receiver_count);
}

void GlobalDyTopoEffectManager::Impl::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    _assemble(info);
    _convert_matrix();
    _distribute(info);
}

void GlobalDyTopoEffectManager::Impl::_assemble(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Assemble Dytopo Effect"};

    auto vertex_count = global_vertex_manager->positions().size();

    auto reporter_gradient_counts = reporter_gradient_offsets_counts.counts();
    auto reporter_hessian_counts  = reporter_hessian_offsets_counts.counts();
    bool gradient_only            = info.m_gradient_only;

    logger::info("DyTopo Effect Assembly: GradientOnly={}, ComponentFlags={}",
                 info.m_gradient_only,
                 enum_flags_name(info.m_component_flags));

    {
        Timer timer{"Report Extent"};
        for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
        {
            reporter_gradient_counts[i] = 0;
            reporter_hessian_counts[i]  = 0;

            if(!has_flags(info.m_component_flags, reporter->component_flags()))
                continue;

            GradientHessianExtentInfo extent_info;
            extent_info.m_gradient_only = gradient_only;
            reporter->report_gradient_hessian_extent(extent_info);

            reporter_gradient_counts[i] = extent_info.m_gradient_count;
            reporter_hessian_counts[i] = gradient_only ? 0 : extent_info.m_hessian_count;
            logger::info("<{}> DyTopo Grad3 count: {}, DyTopo Hess3x3 count: {}",
                         reporter->name(),
                         extent_info.m_gradient_count,
                         extent_info.m_hessian_count);
        }
    }

    {
        Timer timer{"Scan and Allocate"};
        // scan
        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        auto total_gradient_count = reporter_gradient_offsets_counts.total_count();
        auto total_hessian_count  = reporter_hessian_offsets_counts.total_count();

        // allocate
        loose_resize_entries(collected_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(sorted_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(collected_dytopo_effect_hessian, total_hessian_count);
        loose_resize_entries(sorted_dytopo_effect_hessian, total_hessian_count);
        collected_dytopo_effect_gradient.reshape(vertex_count);
        collected_dytopo_effect_hessian.reshape(vertex_count, vertex_count);
    }

    // collect
    for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
    {
        if(!has_flags(info.m_component_flags, reporter->component_flags()))
            continue;

        auto [g_offset, g_count] = reporter_gradient_offsets_counts[i];
        auto [h_offset, h_count] = reporter_hessian_offsets_counts[i];

        GradientHessianInfo info;
        info.m_gradient_only = gradient_only;

        info.m_gradients =
            collected_dytopo_effect_gradient.view().subview(g_offset, g_count);
        info.m_hessians = collected_dytopo_effect_hessian.view().subview(h_offset, h_count);

        reporter->assemble(info);
    }
}

void GlobalDyTopoEffectManager::Impl::_convert_matrix()
{
    Timer timer{"Convert Dytopo Matrix"};

    matrix_converter.convert(collected_dytopo_effect_hessian, sorted_dytopo_effect_hessian);
    matrix_converter.convert(collected_dytopo_effect_gradient, sorted_dytopo_effect_gradient);
}

void GlobalDyTopoEffectManager::Impl::_distribute(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Distribute Dytopo Effect"};

    using namespace muda;

    const auto vertex_count = global_vertex_manager->positions().size();
    const auto receivers    = dytopo_effect_receivers.view();

    const auto sorted_gradient =
        std::as_const(sorted_dytopo_effect_gradient).view();
    const auto sorted_hessian =
        std::as_const(sorted_dytopo_effect_hessian).view();
    const IndexT gradient_count = sorted_gradient.doublet_count();
    const IndexT hessian_count  = sorted_hessian.triplet_count();

    UIPC_ASSERT(sorted_gradient.total_extent() == vertex_count,
                "Sorted DyTopo gradient extent mismatch, expected {}, got {}",
                vertex_count,
                sorted_gradient.total_extent());
    UIPC_ASSERT(sorted_hessian.total_rows() == vertex_count
                    && sorted_hessian.total_cols() == vertex_count,
                "Sorted DyTopo Hessian extent mismatch, expected {}x{}, got {}x{}",
                vertex_count,
                vertex_count,
                sorted_hessian.total_rows(),
                sorted_hessian.total_cols());

    const SizeT receiver_count_size = receivers.size();
    IndexT      hessian_virtual_count;
    const bool  virtual_count_is_valid =
        dytopo_distribution::checked_hessian_virtual_count(
            receiver_count_size, hessian_count, hessian_virtual_count);
    UIPC_ASSERT(virtual_count_is_valid,
                "DyTopo Hessian selection size overflow: receiver_count={}, "
                "hessian_count={}",
                receiver_count_size,
                hessian_count);
    if(!virtual_count_is_valid)
        return;

    const IndexT receiver_count = static_cast<IndexT>(receiver_count_size);

    // Stage 1: collect every receiver's query before launching shared GPU work.
    // report() must only describe classification ranges and must not depend on a
    // previous receiver's receive() side effects.
    bool has_diag_receiver    = false;
    bool has_hessian_receiver = false;
    for(auto&& [i, receiver] : enumerate(receivers))
    {
        // Clear all per-call metadata before report() so no range can leak
        // across full, gradient-only, or empty-input calls.
        receiver_classify_infos[i] = DyTopoClassifyInfo{};
        host_distribution_results[i] =
            dytopo_distribution::empty_distribution_result();

        auto& classify_info = receiver_classify_infos[i];
        receiver->report(classify_info);

        host_distribution_queries[i] =
            dytopo_distribution::make_distribution_query(
                classify_info.gradient_i_range(),
                classify_info.hessian_i_range(),
                classify_info.hessian_j_range());

        has_diag_receiver |= classify_info.is_diag();
        has_hessian_receiver |= !classify_info.is_empty();
    }

    const bool has_gradient_work = has_diag_receiver && gradient_count > 0;
    const bool has_hessian_work = !info.m_gradient_only
                                  && has_hessian_receiver
                                  && hessian_count > 0
                                  && receiver_count > 0;
    const bool has_metadata_work =
        receiver_count > 0 && (has_gradient_work || has_hessian_work);
    IndexT selected_hessian_count = 0;

    // DeviceSelect's output allocation is conservatively R*N. Reuse capacity
    // across calls, but reset the logical size on every path (including N=0 and
    // gradient_only) so stale selections can never be consumed.
    loose_resize(selected_hessian_virtual_indices,
                 has_hessian_work ? hessian_virtual_count : 0);

    if(has_metadata_work)
    {
        // Queries remain alive until the single D2H wait below. The H2D copy,
        // DeviceSelect, result-query kernel, and D2H copy all use the default
        // stream and therefore execute in order.
        BufferLaunch().copy(distribution_queries.view(),
                            host_distribution_queries.data());

        if(has_hessian_work)
        {
            cub::CountingInputIterator<IndexT> virtual_indices{0};

            // The virtual input order is receiver-major: q = receiver*N + k.
            // CUB DeviceSelect is stable, so the compact output is grouped by
            // receiver and preserves each receiver's original Hessian order.
            DeviceSelect().If(
                virtual_indices,
                selected_hessian_virtual_indices.data(),
                selected_hessian_total_count.data(),
                hessian_virtual_count,
                dytopo_distribution::HessianRangePredicate{
                    sorted_hessian.row_indices().data(),
                    sorted_hessian.col_indices().data(),
                    distribution_queries.data(),
                    hessian_count});
        }

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(receiver_count,
                   [gradient_indices = sorted_gradient.indices().cviewer().name(
                        "sorted_dytopo_gradient_indices"),
                    selected_hessian_virtual_indices =
                        selected_hessian_virtual_indices.cviewer().name(
                            "selected_hessian_virtual_indices"),
                    queries = distribution_queries.cviewer().name(
                        "dytopo_distribution_queries"),
                    results = distribution_results.viewer().name(
                        "dytopo_distribution_results"),
                    selected_hessian_total_count =
                        selected_hessian_total_count.cviewer().name(
                            "selected_hessian_total_count"),
                    gradient_count,
                    hessian_count,
                    has_gradient_work,
                    has_hessian_work] __device__(IndexT receiver_index) mutable
                   {
                       auto result =
                           dytopo_distribution::empty_distribution_result();
                       const auto query = queries(receiver_index);

                       if(has_gradient_work)
                       {
                           result.gradient_entry_range =
                               dytopo_distribution::query_sorted_gradient_range(
                                   gradient_indices,
                                   gradient_count,
                                   query.gradient_range);
                       }

                       if(has_hessian_work)
                       {
                           result.hessian_selection_range =
                               dytopo_distribution::query_selected_hessian_range(
                                   selected_hessian_virtual_indices,
                                   *selected_hessian_total_count,
                                   receiver_index,
                                   hessian_count);
                       }

                       results(receiver_index) = result;
                   });

        // This is the only explicit host synchronization in distribution
        // metadata: one D2H transfer returns every receiver's two compact ranges.
        distribution_results.view().copy_to(host_distribution_results.data());

        if(has_hessian_work)
        {
            // Receiver-major selection means the final receiver's exclusive
            // end is exactly CUB's total selected count.
            selected_hessian_count =
                host_distribution_results.back().hessian_selection_range.y();
            UIPC_ASSERT(selected_hessian_count >= 0
                            && selected_hessian_count
                                   <= selected_hessian_virtual_indices.size(),
                        "Invalid DyTopo Hessian selected count {} for capacity {}",
                        selected_hessian_count,
                        selected_hessian_virtual_indices.size());
        }
    }

    // Stage 2: materialize independent receiver-owned buffers. This preserves
    // the lifetime contract used by receivers after this function returns.
    for(auto&& [i, receiver] : enumerate(receivers))
    {
        const auto& classify_info = receiver_classify_infos[i];

        // Value initialization guarantees empty gradient/Hessian views on
        // every call, including off-diagonal and gradient-only receivers.
        ClassifiedDyTopoEffectInfo classified_info{};
        auto& classified_gradients = classified_dytopo_effect_gradients[i];
        auto& classified_hessians = classified_dytopo_effect_hessians[i];
        classified_gradients.reshape(vertex_count);
        classified_hessians.reshape(vertex_count, vertex_count);

        // 1) report gradient
        if(classify_info.is_diag())
        {
            const auto range = host_distribution_results[i].gradient_entry_range;
            const auto count = range.y() - range.x();

            UIPC_ASSERT(range.x() >= 0 && range.x() <= range.y()
                            && range.y() <= gradient_count,
                        "Invalid sorted DyTopo gradient subview [{}, {}) for count {}",
                        range.x(),
                        range.y(),
                        gradient_count);

            loose_resize_entries(classified_gradients, count);

            if(count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(count,
                           [sorted_gradient = sorted_gradient.cviewer().name(
                                "sorted_dytopo_effect_gradient"),
                            classified_gradient = classified_gradients.viewer().name(
                                "classified_gradient"),
                            begin = range.x()] __device__(IndexT I) mutable
                           {
                               auto&& [index, value] = sorted_gradient(begin + I);
                               classified_gradient(I).write(index, value);
                           });
            }

            classified_info.m_gradients = classified_gradients.view();
        }

        // 2) report hessian
        if(!info.m_gradient_only && !classify_info.is_empty())
        {
            const auto range =
                host_distribution_results[i].hessian_selection_range;
            const auto count = range.y() - range.x();

            UIPC_ASSERT(
                range.x() >= 0 && range.x() <= range.y()
                    && range.y() <= selected_hessian_count,
                "Invalid compact DyTopo Hessian range [{}, {}) for selection "
                "count {}",
                range.x(),
                range.y(),
                selected_hessian_count);

            loose_resize_entries(classified_hessians, count);

            if(count > 0)
            {
                const IndexT receiver_index = static_cast<IndexT>(i);
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(count,
                           [selected_hessian_virtual_indices =
                                selected_hessian_virtual_indices.cviewer().name(
                                    "selected_hessian_virtual_indices"),
                            sorted_hessian = sorted_hessian.cviewer().name(
                                "sorted_dytopo_effect_hessian"),
                            classified_hessian = classified_hessians.viewer().name(
                                "classified_hessian"),
                            begin = range.x(),
                            receiver_index,
                            hessian_count] __device__(IndexT I) mutable
                           {
                               const IndexT virtual_index =
                                   selected_hessian_virtual_indices(begin + I);
                               const IndexT source_index =
                                   virtual_index - receiver_index * hessian_count;

                               MUDA_KERNEL_ASSERT(
                                   source_index >= 0
                                       && source_index < hessian_count,
                                   "DyTopo selected Hessian source index out of "
                                   "range: source=%d, count=%d",
                                   source_index,
                                   hessian_count);

                               auto&& [row, col, value] =
                                   sorted_hessian(source_index);
                               classified_hessian(I).write(row, col, value);
                           });
            }

            classified_info.m_hessians = classified_hessians.view();
        }

        receiver->receive(classified_info);
    }
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceTripletMatrix<Float, 3>& m, SizeT size)
{
    if(size > m.triplet_capacity())
    {
        m.reserve_triplets(size * reserve_ratio);
    }
    m.resize_triplets(size);
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceDoubletVector<Float, 3>& v, SizeT size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets(size * reserve_ratio);
    }
    v.resize_doublets(size);
}
}  // namespace uipc::backend::cuda


namespace uipc::backend::cuda
{
void GlobalDyTopoEffectManager::init()
{
    m_impl.init(world());
}

void GlobalDyTopoEffectManager::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::compute_dytopo_effect()
{
    ComputeDyTopoEffectInfo info;
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::add_reporter(DyTopoEffectReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    auto flag = reporter->component_flags();
    UIPC_ASSERT(is_valid_flag(flag),
                "reporter component_flags() is not valid single flag, it's {}",
                enum_flags_name(flag));
    m_impl.dytopo_effect_reporters.register_sim_system(*reporter);

    // classify into contact / non-contact
    if(reporter->component_flags() == EnergyComponentFlags::Contact)
    {
        m_impl.contact_reporters.register_sim_system(*reporter);
    }
    else
    {
        m_impl.non_contact_reporters.register_sim_system(*reporter);
    }
}

void GlobalDyTopoEffectManager::add_receiver(DyTopoEffectReceiver* receiver)
{
    check_state(SimEngineState::BuildSystems, "add_receiver()");
    UIPC_ASSERT(receiver != nullptr, "receiver is nullptr");
    m_impl.dytopo_effect_receivers.register_sim_system(*receiver);
}
}  // namespace uipc::backend::cuda
