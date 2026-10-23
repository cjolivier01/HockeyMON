"""Draw authoritative player labels without a second evidence or expiry cache."""

from hmlib.aspen.player_semantics import SettledJerseys
from hmlib.utils.image import is_channels_last
from hmlib.vis.labels import draw_label_box


def draw_current_jerseys(image, tracking_ids, boxes_tlwh, labels: SettledJerseys):
    current = {label.tracking_id: label for label in labels}
    width = int(image.shape[1] if is_channels_last(image) else image.shape[-1])
    for tracking_id, (x, y, width_box, _) in zip(tracking_ids, boxes_tlwh, strict=True):
        label = current.get(int(tracking_id))
        if label is None:
            continue
        image = draw_label_box(
            image=image,
            text=label.number,
            center_x=int(x + width_box / 2),
            bottom_y=int(y),
            font_scale=max(0.6, width / 1800),
            fg=(255, 255, 255),
            bg=(0, 0, 0),
            bg_alpha=int(max(80, min(200, 80 + label.score * 60))),
        )
    return image
