#pragma once
#include <sim_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>

namespace uipc::backend::cuda
{
class DyTopoEffectReceiver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

  protected:
    virtual void do_init(InitInfo&);

    /**
     * Describe this receiver's fixed classification ranges for the current
     * distribution pass. All receivers are reported before any receiver is
     * given its classified data, so this callback must be query-only and must
     * not depend on do_receive() side effects from another receiver.
     */
    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) = 0;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) = 0;
    virtual void do_build(BuildInfo& info) = 0;

  private:
    friend class GlobalDyTopoEffectManager;
    virtual void do_build() final override;
    void         init();  // only be called by GlobalDyTopoEffectManager
    void         report(GlobalDyTopoEffectManager::ClassifyInfo& info);
    void  receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);
    SizeT m_index = ~0ull;
};
}  // namespace uipc::backend::cuda
