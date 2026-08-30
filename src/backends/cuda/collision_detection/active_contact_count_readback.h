#pragma once

#include <type_define.h>
#include <muda/buffer/device_buffer.h>

#include <array>

namespace uipc::backend::cuda
{
enum class ActiveContactCountSlot : SizeT
{
    PP = 0,
    PE,
    PT,
    EE,
    Count
};

class ActiveContactCountReadback
{
  public:
    void prepare()
    {
        m_device_counts.resize(static_cast<SizeT>(ActiveContactCountSlot::Count));
        m_device_counts.view().fill(0);
    }

    IndexT* output(ActiveContactCountSlot slot) noexcept
    {
        return m_device_counts.data() + static_cast<SizeT>(slot);
    }

    const std::array<IndexT, static_cast<SizeT>(ActiveContactCountSlot::Count)>& read()
    {
        m_device_counts.view().copy_to(m_host_counts.data());
        return m_host_counts;
    }

  private:
    muda::DeviceBuffer<IndexT> m_device_counts;
    std::array<IndexT, static_cast<SizeT>(ActiveContactCountSlot::Count)> m_host_counts{};
};
}  // namespace uipc::backend::cuda
