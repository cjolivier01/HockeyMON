"""Preview rendering must preserve recording pixels and follow camera geometry."""

import cv2
import numpy as np
import torch

from hmlib.ui.preview_overlays import PreviewOverlayRenderer, compose_transform, copy_metadata


def should_draw_each_preview_layer_on_a_private_copy():
    renderer = PreviewOverlayRenderer()
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[10:110, 20:180] = 1
    metadata = {
        "matrix": np.eye(3),
        "boxes": torch.tensor([[40, 30, 60, 80]]),
        "ids": torch.tensor([4]),
        "play_box": torch.tensor([30, 20, 150, 100]),
        "fast_box": torch.tensor([40, 30, 130, 90]),
        "rink_mask": mask,
    }
    for layers in ((True, False, False), (False, True, False), (False, False, True)):
        result = renderer.draw(image, metadata, (200, 120), layers)
        assert result.any()
        assert not image.any()
        assert not np.shares_memory(result, image)
    assert renderer.draw(image, metadata, (200, 120), (False, False, False)) is image
    assert torch.equal(metadata["boxes"], torch.tensor([[40, 30, 60, 80]]))


def should_copy_geometry_without_copying_large_static_masks():
    mask = np.ones((1024, 2048), dtype=np.uint8)
    source = [{"matrix": np.eye(3), "rink_mask": mask, "source_frame_id": 17, "decoded_pts_ns": 19}]
    transformed = copy_metadata(source)
    translation = np.eye(3)
    translation[:2, 2] = [-20, -10]
    compose_transform(transformed, 0, translation)
    assert np.array_equal(source[0]["matrix"], np.eye(3))
    assert transformed[0]["rink_mask"] is mask
    assert transformed[0]["source_frame_id"] == 17 and transformed[0]["decoded_pts_ns"] == 19
    renderer = PreviewOverlayRenderer()
    contours = renderer._rink_polygons(mask)
    assert renderer._rink_polygons(mask) is contours
    assert sum(len(polygon) for polygon in contours) == 4
    assert contours[0][:, 0].max() > 2000


def should_place_rotated_preview_boxes_in_program_coordinates():
    source = [{"matrix": np.eye(3), "boxes": np.array([[20, 10, 40, 30]]), "ids": np.array([1])}]
    transformed = copy_metadata(source)
    rotation = np.eye(3)
    rotation[:2] = cv2.getRotationMatrix2D((50, 40), 90, 1)
    compose_transform(transformed, 0, rotation)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    result = PreviewOverlayRenderer().draw(image, transformed[0], (100, 80), (True, False, False))
    expected = cv2.transform(np.array([[[20.0, 10.0]]]), rotation[:2])[0, 0].astype(int)
    x, y = expected
    assert result[y - 1 : y + 2, x - 1 : x + 2].any()
    assert not image.any()


def should_match_rotation_metadata_to_actual_torchvision_pixels():
    from hmlib.transforms.perspective_rotation import HmPerspectiveRotation

    image = torch.zeros((1, 80, 160, 3))
    image[0, 25:28, 30:33] = 1
    metadata = [{"matrix": np.eye(3)}]
    transform = HmPerspectiveRotation(fixed_edge_rotation_angle=25, pre_clip=True)
    inputs = {
        "img": image,
        "camera_box": torch.tensor([[10.0, 5.0, 90.0, 65.0]]),
        "shared_preview_metadata": metadata,
    }
    output = transform(inputs)["img"][0]
    projected = metadata[0]["matrix"] @ [31, 26, 1]
    ys, xs = torch.where(output[:, :, 0] > 0.4)
    assert xs.numel() > 0
    assert abs(float(xs.float().mean()) - projected[0]) < 1
    assert abs(float(ys.float().mean()) - projected[1]) < 1
