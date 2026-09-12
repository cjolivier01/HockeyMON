"""Read a single recording through HM's tracking-reuse interface."""

import json

import numpy as np
import pandas as pd

from hmlib.telemetry.database import completed_runs, read_database
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame


class DatabaseTrackingDataFrame(TrackingDataFrame):
    def read_data(self):
        with read_database(self.input_file) as connection:
            runs = completed_runs(connection)
            if len(runs) != 1:
                raise ValueError("Tracking reuse requires one telemetry run")
            run_id = runs[0]["run_id"]
            frames = connection.execute(
                "SELECT source_frame,source_id,seek_epoch,reset_epoch,geometry_id FROM frames "
                "WHERE run_id=? ORDER BY sample_id",
                (run_id,),
            ).fetchall()
            self._frames = {row[0] for row in frames}
            if len(self._frames) != len(frames) or len({tuple(row[1:]) for row in frames}) != 1:
                raise ValueError("Tracking reuse requires unambiguous frames in one geometry/epoch")
            rows = connection.execute(
                "SELECT f.source_frame,t.* FROM tracks t JOIN frames f USING(run_id,sample_id) "
                "WHERE t.run_id=? ORDER BY t.sample_id,t.ordinal",
                (run_id,),
            ).fetchall()
        records = []
        for row in rows:
            tid = int(row["tracking_id"])
            if not 0 <= tid <= np.iinfo(np.int64).max:
                raise ValueError(
                    "Tracking ID cannot be represented by HM's signed tracking tensors"
                )
            attributes = json.loads(row["attributes_json"])
            action = attributes.get("action_results", {})
            records.append(
                (
                    row["source_frame"],
                    tid,
                    row["left"],
                    row["top"],
                    row["width"],
                    row["height"],
                    row["score"],
                    row["class_id"],
                    1.0,
                    json.dumps(attributes.get("jersey_results", {})),
                    action.get("label", ""),
                    action.get("score", 0.0),
                    action.get("label_index", -1),
                )
            )
        self.data = pd.DataFrame(records, columns=self.fields).astype(
            {
                "Frame": "int64",
                "ID": "int64",
                "Labels": "int64",
                "ActionIndex": "int64",
                **{
                    key: "float64"
                    for key in (
                        "BBox_X",
                        "BBox_Y",
                        "BBox_W",
                        "BBox_H",
                        "Scores",
                        "Visibility",
                        "ActionScore",
                    )
                },
            }
        )

    def get_data_dict_by_frame(self, frame_id):
        if int(frame_id) not in self._frames:
            raise ValueError(f"Source frame {frame_id} is absent from the telemetry recording")
        return super().get_data_dict_by_frame(frame_id)
