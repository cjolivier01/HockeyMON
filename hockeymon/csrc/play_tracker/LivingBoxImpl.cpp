#include "hockeymon/csrc/play_tracker/LivingBoxImpl.h"
#include "hockeymon/csrc/play_tracker/LogCapture.h"

#include <cassert>
#include <iostream>
#include "BoxUtils.h"

namespace hm {
namespace play_tracker {

namespace {

// Helper to define a visitor based on lambda expressions
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
} // namespace

LivingBox::LivingBox(
    std::string label,
    BBox bbox,
    const AllLivingBoxConfig& config)
    : BoundingBox(
          bbox.make_scaled(config.scale_dest_width, config.scale_dest_height)),
      ResizingBox(config),
      TranslatingBox(config),
      label_(label),
      config_(config) {
  // Sanity check
  if (config.arena_box.has_value()) {
    const BBox& arena_box = *config.arena_box;
    assert(config.max_width <= arena_box.width());
    assert(config.max_height <= arena_box.height());
    max_eff_width_ = config.max_width != 0
        ? std::min(arena_box.width(), config.max_width)
        : arena_box.width();
    max_eff_height_ = config.max_height != 0
        ? std::min(arena_box.height(), config.max_height)
        : arena_box.height();
  }
}

WHDims LivingBox::get_size_scale() const {
  return WHDims{
      .width = config_.scale_dest_width,
      .height = config_.scale_dest_height,
  };
}

std::optional<BBox> LivingBox::get_arena_box() const {
  return TranslatingBox::get_config().arena_box;
}

BBox LivingBox::next_position() {
  // These diffs take into account size or translation stickiness
  const PointDiff translation_change = get_proposed_next_position_change();
  const SizeDiff size_change = get_proposed_next_size_change();

  BBox new_box = bounding_box();
  new_box.left += translation_change.dx - size_change.dw;
  new_box.top += translation_change.dy - size_change.dh;
  new_box.right += translation_change.dx + size_change.dw;
  new_box.bottom += translation_change.dy + size_change.dh;

  new_box = new_box.normalize(/*reversible=*/false);
  if (new_box.empty()) {
    hm_log_error("Empty box for next position!");
  }

  // Constrain size
  const FloatValue new_ww = new_box.width(), new_hh = new_box.height();
  const WHDims min_allowed_width_height = get_min_allowed_width_height();
  const FloatValue ww = std::max(new_ww, min_allowed_width_height.width);
  const FloatValue hh = std::max(new_hh, min_allowed_width_height.height);
  state_.was_size_constrained = ww != new_ww || hh != new_hh;

  new_box = BBox(new_box.center(), WHDims{.width = ww, .height = hh});
  // Assign new bounding box
  set_bbox(new_box);

  update_nonstop_delay();
  TranslatingBox::update_stop_delays();
  ResizingBox::update_stop_delays();
  stop_translation_if_out_of_arena();
  clamp_to_arena();

  // Maybe adjust aspect ratio
  if (config_.fixed_aspect_ratio.has_value()) {
    // std::cout << name() << ": " << bounding_box() << std::endl;
    const TranslatingBoxConfig& tconfig = TranslatingBox::get_config();
    set_bbox(set_box_aspect_ratio(
        bounding_box(), *config_.fixed_aspect_ratio, tconfig.arena_box));
    // std::cout << name() << ": " << bounding_box() << std::endl;
  }

  clamp_size_scaled();
  // std::cout << name() << ": " << bounding_box() << std::endl;

  on_new_position();
  // std::cout << name() << ": " << bounding_box() << std::endl;

  // Check that we maintained our aspect ratio
  if (config_.fixed_aspect_ratio.has_value()) {
    assert(isClose(bounding_box().aspect_ratio(), *config_.fixed_aspect_ratio));
  }
  // std::cout << name() << ": " << bounding_box() << std::endl;
  return bounding_box();
}

void LivingBox::clamp_to_arena() {
  BBox bbox = bounding_box();
  auto arena = TranslatingBox::get_config().arena_box;
  // Constrain size
  const ResizingConfig& rconfig = ResizingBox::get_config();
  bbox = hm::BBox(
      bbox.center(),
      hm::WHDims{
          .width = std::min(bbox.width(), max_eff_width_),
          .height = std::min(bbox.height(), max_eff_height_)});
  if (arena.has_value()) {
    // is shifting necessary or was already done?
    // shift_box_to_edge(bbox, *arena);
    // Constrain by arena outer limits
    bbox = clamp_box(bbox, *arena);
  }
  set_bbox(bbox);
}

// -IBasicLivingBox
void LivingBox::set_bbox(const BBox& bbox) {
  BoundingBox::set_bbox(bbox);
}

BBox LivingBox::bounding_box() const {
  return BoundingBox::bounding_box();
}

void LivingBox::set_destination_ex(
    const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest) {
  BBox dest_box;
  std::visit(
      overloaded{
          [&dest_box](std::shared_ptr<IBasicLivingBox> living_box) {
            dest_box = living_box->bounding_box();
          },
          [&dest_box](BBox bbox) { dest_box = bbox; },
      },
      dest);
  set_destination(dest_box);
}

BBox LivingBox::forward(
    const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest) {
  set_destination_ex(dest);
  BBox new_pos = next_position();
  assert(new_pos.left >= 0);
  assert(new_pos.top >= 0);
  ++forward_counter_;
  return new_pos;
}

void LivingBox::set_destination(const BBox& dest_box) {
  BBox destination_box = dest_box;
  WHDims scale_box = get_size_scale();
  if (scale_box.width != 1 || scale_box.height != 1) {
    destination_box =
        destination_box.make_scaled(scale_box.width, scale_box.height);
    auto& arena_box = TranslatingBox::get_config().arena_box;
    if (config_.clamp_scaled_input_box && arena_box.has_value()) {
      destination_box = clamp_box(destination_box, *arena_box);
    }
  }
  // std::cout << name() << ": set_destination: " << destination_box <<
  // std::endl;
  ResizingBox::set_destination(destination_box);
  TranslatingBox::set_destination(destination_box);
}
// IBasicLivingBox-

} // namespace play_tracker
} // namespace hm
