#pragma once

namespace uipc::backend::cuda
{
template <typename Index>
struct ContactTypeBlockLayout
{
    Index pt_end;
    Index ee_offset;
    Index ee_end;
    Index pe_offset;
    Index pe_end;
    Index pp_offset;
    Index padded_total;
};

// Lay out PT, EE, PE and PP contacts so that no CUDA thread block crosses a
// contact-type boundary. Padding indices are intentionally left unmapped and
// return before evaluating a contact formula.
template <int BlockSize, typename Index>
constexpr ContactTypeBlockLayout<Index> make_contact_type_block_layout(
    Index pt_count,
    Index ee_count,
    Index pe_count,
    Index pp_count)
{
    static_assert(BlockSize > 0);
    const auto round_up = [](Index count)
    {
        return ((count + Index(BlockSize) - Index(1)) / Index(BlockSize))
               * Index(BlockSize);
    };

    ContactTypeBlockLayout<Index> layout{};
    layout.pt_end       = pt_count;
    layout.ee_offset    = round_up(pt_count);
    layout.ee_end       = layout.ee_offset + ee_count;
    layout.pe_offset    = layout.ee_offset + round_up(ee_count);
    layout.pe_end       = layout.pe_offset + pe_count;
    layout.pp_offset    = layout.pe_offset + round_up(pe_count);
    layout.padded_total = layout.pp_offset + pp_count;
    return layout;
}
}  // namespace uipc::backend::cuda
