#include <app/app.h>
#include <collision_detection/active_contact_count_readback.h>
#include <muda/buffer/device_buffer.h>
#include <muda/cub/device/device_select.h>
#include <type_define.h>

#include <array>
#include <vector>

using namespace muda;
using namespace uipc;
using namespace uipc::backend::cuda;

namespace
{
constexpr SizeT SlotCount = static_cast<SizeT>(ActiveContactCountSlot::Count);
using Inputs              = std::array<std::vector<IndexT>, SlotCount>;
using Outputs             = std::array<std::vector<IndexT>, SlotCount>;

ActiveContactCountSlot slot(SizeT i)
{
    return static_cast<ActiveContactCountSlot>(i);
}

std::vector<IndexT> legacy_select(const std::vector<IndexT>& input)
{
    if(input.empty())
        return {};

    DeviceBuffer<IndexT> device_input(input.size());
    DeviceBuffer<IndexT> device_output(input.size());
    DeviceBuffer<IndexT> device_count(1);

    device_input.view().copy_from(input.data());

    DeviceSelect().If(device_input.data(),
                      device_output.data(),
                      device_count.data(),
                      input.size(),
                      [] CUB_RUNTIME_FUNCTION(IndexT value) { return value >= 0; });

    IndexT count = -1;
    device_count.view().copy_to(&count);

    std::vector<IndexT> output(count);
    if(count > 0)
        device_output.view(0, count).copy_to(output.data());
    return output;
}

class BatchedCountFixture
{
  public:
    void compare(const Inputs& inputs)
    {
        Outputs legacy;
        for(SizeT i = 0; i < SlotCount; ++i)
            legacy[i] = legacy_select(inputs[i]);

        std::array<DeviceBuffer<IndexT>, SlotCount> device_inputs;
        std::array<DeviceBuffer<IndexT>, SlotCount> device_outputs;

        m_readback.prepare();
        for(SizeT i = 0; i < SlotCount; ++i)
        {
            device_inputs[i].resize(inputs[i].size());
            device_outputs[i].resize(inputs[i].size());
            if(!inputs[i].empty())
                device_inputs[i].view().copy_from(inputs[i].data());

            DeviceSelect().If(device_inputs[i].data(),
                              device_outputs[i].data(),
                              m_readback.output(slot(i)),
                              inputs[i].size(),
                              [] CUB_RUNTIME_FUNCTION(IndexT value) { return value >= 0; });
        }

        const auto& counts = m_readback.read();
        Outputs     batched;
        for(SizeT i = 0; i < SlotCount; ++i)
        {
            REQUIRE(counts[i] == static_cast<IndexT>(legacy[i].size()));
            batched[i].resize(counts[i]);
            if(counts[i] > 0)
                device_outputs[i].view(0, counts[i]).copy_to(batched[i].data());
            REQUIRE(batched[i] == legacy[i]);
        }
    }

  private:
    ActiveContactCountReadback m_readback;
};

std::vector<IndexT> mixed_input(SizeT size)
{
    std::vector<IndexT> values(size);
    for(SizeT i = 0; i < size; ++i)
        values[i] = i % 3 == 0 ? -1 : static_cast<IndexT>(i);
    return values;
}
}  // namespace

TEST_CASE("Line Search active contact counts preserve four-readback results",
          "[cuda][line_search][active_count][LS10]")
{
    BatchedCountFixture fixture;

    SECTION("all four classes are zero length")
    {
        fixture.compare(Inputs{});
    }

    SECTION("one class is nonzero with a non-integral CTA-sized input")
    {
        Inputs inputs;
        inputs[static_cast<SizeT>(ActiveContactCountSlot::PT)] = mixed_input(257);
        fixture.compare(inputs);
    }

    SECTION("empty middle classes preserve PP PE PT EE slots")
    {
        Inputs inputs;
        inputs[static_cast<SizeT>(ActiveContactCountSlot::PP)] = mixed_input(7);
        inputs[static_cast<SizeT>(ActiveContactCountSlot::PT)] = mixed_input(129);
        fixture.compare(inputs);
    }

    SECTION("all classes preserve stable selection order")
    {
        Inputs inputs;
        inputs[static_cast<SizeT>(ActiveContactCountSlot::PP)] = mixed_input(33);
        inputs[static_cast<SizeT>(ActiveContactCountSlot::PE)] = mixed_input(65);
        inputs[static_cast<SizeT>(ActiveContactCountSlot::PT)] = mixed_input(129);
        inputs[static_cast<SizeT>(ActiveContactCountSlot::EE)] = mixed_input(257);
        fixture.compare(inputs);
    }

    SECTION("nonzero zero nonzero reuse does not retain stale counts")
    {
        Inputs first;
        first[0] = mixed_input(17);
        first[1] = mixed_input(19);
        first[2] = mixed_input(23);
        first[3] = mixed_input(29);
        fixture.compare(first);
        fixture.compare(Inputs{});

        Inputs third;
        third[0] = mixed_input(5);
        third[1] = mixed_input(11);
        third[2] = mixed_input(13);
        third[3] = mixed_input(31);
        fixture.compare(third);
    }
}
