#pragma once
#include <sim_system.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::cuda
{
class DiagLinearSubsystem;

class LocalPreconditioner : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class InitInfo
    {
      public:
    };

    class BuildInfo
    {
      public:
        void connect(DiagLinearSubsystem* system);

      private:
        friend class LocalPreconditioner;
        DiagLinearSubsystem* m_subsystem = nullptr;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info)   = 0;
    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) = 0;
    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) = 0;
    virtual bool  do_supports_fused_pcg() const { return false; }
    virtual SizeT do_fused_pcg_signature() const { return 0; }
    virtual void do_fused_pcg_apply(GlobalLinearSystem::FusedPcgIterationInfo& info);

  private:
    friend class GlobalLinearSystem;

    virtual void do_build() final override;
    virtual void init();

    void  assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info);
    void  apply(GlobalLinearSystem::ApplyPreconditionerInfo& info);
    bool  supports_fused_pcg() const { return do_supports_fused_pcg(); }
    SizeT fused_pcg_signature() const { return do_fused_pcg_signature(); }
    void  apply_fused_pcg(GlobalLinearSystem::FusedPcgIterationInfo& info)
    {
        do_fused_pcg_apply(info);
    }
    DiagLinearSubsystem* m_subsystem = nullptr;
};
}  // namespace uipc::backend::cuda
