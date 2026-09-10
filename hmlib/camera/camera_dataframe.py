"""DataFrame wrapper for storing per-frame camera bounding boxes.

Used by camera training and analysis tools to persist camera TLWH boxes.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hmlib.datasets.dataframe import HmDataFrameBase
from hmlib.tracking_utils.tracking_dataframe import convert_tlbr_to_tlwh


class CameraPolicyDataFrame(HmDataFrameBase):
    """Durable, headerless Frame/PolicyJSON companion for camera training data."""

    def __init__(self, output_file: str):
        super().__init__(fields=["Frame", "PolicyJSON"], output_file=output_file)

    def add_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        rows = [
            {
                "Frame": event["frame"],
                "PolicyJSON": json.dumps(
                    {key: value for key, value in event.items() if key != "frame"},
                    allow_nan=False,
                    sort_keys=True,
                ),
            }
            for event in events
        ]
        self._dataframe_list.append(pd.DataFrame(rows))
        # Policy changes are infrequent. Persist before the associated camera
        # rows can be flushed, and propagate uncertain-write failures normally.
        self.write_data()


class CameraTrackingDataFrame(HmDataFrameBase):
    """DataFrame-based storage for camera TLWH boxes."""

    def __init__(self, *args, input_batch_size: int, **kwargs):
        fields: List[str] = [
            "Frame",
            "BBox_X",
            "BBox_Y",
            "BBox_W",
            "BBox_H",
        ]
        super().__init__(*args, fields=fields, input_batch_size=input_batch_size, **kwargs)

    def add_frame_records(
        self,
        frame_id: int,
        tlbr: Optional[np.ndarray] = None,
        tlwh: Optional[np.ndarray] = None,
    ):
        """Append tracking records for a given frame.

        @param frame_id: Frame index.
        @param tlbr: Optional Nx4 TLBR array.
        @param tlwh: Optional Nx4 TLWH array; if None, derived from ``tlbr``.
        """
        if tlwh is None:
            assert tlbr is not None
            tlwh = convert_tlbr_to_tlwh(tlbr)
        tlwh = self._make_array(tlwh)
        frame_id = int(frame_id)
        new_record = pd.DataFrame(
            {
                "Frame": frame_id,
                "BBox_X": tlwh[:, 0],
                "BBox_Y": tlwh[:, 1],
                "BBox_W": tlwh[:, 2],
                "BBox_H": tlwh[:, 3],
            }
        )
        self._dataframe_list.append(new_record)
        self.counter += 1

        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0  # Reset the counter after writing

    def get_data_by_frame(self, frame_number):
        """Get all tracking data for a specific frame."""
        if not self.data.empty:
            return self.data[self.data["Frame"] == frame_number]
        else:
            print("No data loaded.")
            return None

    # Function to extract tracking info by frame
    def get_data_dict_by_frame(self, frame_id: int) -> Dict[str, Any]:
        """Return a dict of camera tracking data for a frame.

        The dict contains keys: ``frame_id`` and ``bboxes`` (TLWH array).
        """
        assert self.batch_size == 1
        frame_id = int(frame_id)
        assert frame_id  # First frame is 1
        # Filter the DataFrame for the specified frame
        frame_data = self.data[self.data["Frame"] == frame_id]
        # Extract columns as NumPy arrays
        bboxes = frame_data[["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy()
        return {"frame_id": frame_id, "bboxes": bboxes}
