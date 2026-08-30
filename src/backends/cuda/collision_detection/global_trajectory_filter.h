#pragma once
#include <sim_system.h>
#include <muda/buffer/device_buffer.h>

namespace uipc::backend::cuda
{
class TrajectoryFilter;
class GlobalContactManager;
class ContactExporterManager;

class GlobalTrajectoryFilter final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class FilterTOIInfo
    {
      public:
        Float                alpha() const noexcept { return m_alpha; }
        muda::VarView<Float> toi() const noexcept { return m_toi; }

      private:
        friend class GlobalTrajectoryFilter;
        Float                m_alpha = 0.0;
        muda::VarView<Float> m_toi;
    };

    class DetectInfo
    {
      public:
        Float alpha() const noexcept { return m_alpha; }
        bool reuse_bvh_topology() const noexcept { return m_reuse_bvh_topology; }

      private:
        friend class GlobalTrajectoryFilter;
        Float m_alpha              = 0.0;
        bool  m_reuse_bvh_topology = false;
    };

    class FilterActiveInfo
    {
      public:
        FilterActiveInfo(Impl* impl, bool batch_active_counts) noexcept
            : m_impl(impl)
            , m_batch_active_counts(batch_active_counts)
        {
        }

        bool batch_active_counts() const noexcept { return m_batch_active_counts; }


      private:
        friend class GlobalTrajectoryFilter;
        Impl* m_impl;
        bool  m_batch_active_counts = false;
    };

    class LabelActiveVerticesInfo
    {
      public:
        LabelActiveVerticesInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }
        muda::BufferView<IndexT> vert_is_active() const noexcept;

      private:
        friend class GlobalTrajectoryFilter;
        Impl* m_impl;
    };

    class RecordFrictionCandidatesInfo
    {
      public:
    };

    class Impl
    {
      public:
        void  init();
        Float filter_toi(Float alpha);

        SimSystemSlotCollection<TrajectoryFilter> filters;
        SimSystemSlot<GlobalContactManager>       global_contact_manager;
        bool                                      friction_enabled = false;
        bool should_discard_friction_candidates                    = false;


        muda::DeviceBuffer<Float> tois;
        vector<Float>             h_tois;
    };

    template <std::derived_from<SimSystem> T>
    SimSystemSlot<T> find()
    {
        return m_impl.filters.find<T>();
    }

    void add_filter(TrajectoryFilter* filter);
    void require_discard_friction();

  private:
    virtual void do_build() override final;
    virtual void do_apply_recover(RecoverInfo& info) override final;

    friend class SimEngine;
    friend class ContactExporterManager;
    void detect(Float alpha,
                bool reuse_bvh_topology = false);  // called by SimEngine and ContactExporterManager
    void filter_active(bool batch_active_counts = false);  // called by SimEngine and ContactExporterManager

    Float filter_toi(Float alpha);       // only called by SimEngine
    void  record_friction_candidates();  // only called by SimEngine
    friend class GlobalContactManager;
    void label_active_vertices();      // only called by GlobalContactManager
    void clear_friction_candidates();  // called by GlobalContactManager

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
