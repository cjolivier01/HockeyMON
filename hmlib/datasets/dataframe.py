"""DataFrame-backed datasets for tracking, camera, and action results.

Defines :class:`HmDataFrameBase` and helpers for reading/writing CSV-based
frame data plus a thin :class:`Dataset` wrapper for training.
"""

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hmlib.log import logger
from hmlib.utils.gpu import unwrap_tensor


class HmDataFrameBase:
    def __init__(
        self,
        fields: List[str],
        input_batch_size: Union[int, None] = None,
        input_file=None,
        output_file=None,
        write_interval: int = 250,
    ):
        self._fields = fields
        self.input_file = input_file
        self._input_batch_size: Union[int, None] = input_batch_size
        self.output_file = output_file
        self.write_interval = write_interval
        self.first_write = True
        self._write_error: Optional[Exception] = None
        self._dataframe_list: List[pd.DataFrame] = []
        self.counter = 0  # Counter to track number of records since the last write
        self.data: Optional[pd.DataFrame] = None
        self._ilocator = None
        if input_file:
            self.read_data()

    def read_data(self) -> None:
        """Read data from a CSV file."""
        if self.input_file:
            if not os.path.exists(self.input_file):
                logger.error(f"Could not open dataframe file: {self.input_file}")
                # In case self.data it was set already, None it out
                self.data = None
                return
            self.data = pd.read_csv(
                self.input_file,
                header=None,
                names=self._fields,
            )

    @property
    def batch_size(self) -> int:
        assert self._input_batch_size is not None
        return self._input_batch_size

    def __iter__(self) -> Iterable:
        return self.data.itertuples(index=True, name="TrackingDataFrame")

    def __len__(self) -> int:
        assert self.data is not None
        return len(self.data)

    @property
    def fields(self) -> List[str]:
        return self._fields

    @staticmethod
    def _make_array(t: Union[np.ndarray, torch.Tensor, List[Any], Tuple[Any, ...]]) -> np.ndarray:
        """Convert tensors (including sequences of tensors) to CPU numpy arrays."""
        t = unwrap_tensor(t)
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        if isinstance(t, (list, tuple)):
            # Handle sequences that may contain tensors
            if not t:
                return np.empty((0,), dtype=np.float32)
            converted = []
            for item in t:
                if isinstance(item, torch.Tensor):
                    converted.append(item.detach().cpu().numpy())
                else:
                    converted.append(item)
            return np.asarray(converted)
        return t

    @staticmethod
    def get_ndarray(obj, name: str, dflt: np.ndarray) -> np.ndarray:
        """Convert various types to numpy ndarray."""
        t = getattr(obj, name)
        if t is None:
            return dflt
        return HmDataFrameBase._make_array(t)

    def has_input_data(self):
        return self.input_file is not None

    def write_data(self, output_path=None, header=False):
        """Write buffered rows durably, without retrying an uncertain append."""
        if self._write_error is not None:
            raise RuntimeError(
                f"CSV output previously failed: {self.output_file}"
            ) from self._write_error
        if not output_path:
            output_path = self.output_file
        else:
            self.output_file = output_path

        if self.output_file:
            if self._dataframe_list or self.first_write:
                data = (
                    pd.concat(self._dataframe_list, ignore_index=True)
                    if self._dataframe_list
                    else pd.DataFrame(columns=self._fields)
                )
                mode = "a" if not self.first_write else "w"
                try:
                    with open(output_path, mode, encoding="utf-8", newline="") as output:
                        data.to_csv(output, header=header, index=False)
                        output.flush()
                        os.fsync(output.fileno())
                except Exception as error:
                    self._write_error = error
                    raise
                self.first_write = False
                self._dataframe_list = []
                logger.info("Data saved successfully to %s.", output_path)
            else:
                logger.info("No data available to save.")

    def flush(self):
        self.write_data()

    def close(self):
        self.flush()


# Convert dataclass to JSON
def dataclass_to_json(dataclass_instance):
    if dataclass_instance is None:
        return ""
    if not is_dataclass(dataclass_instance):
        raise ValueError("Provided input is not a dataclass instance")
    dataclass_dict = asdict(dataclass_instance)
    json_str = json.dumps(dataclass_dict)
    return json_str


# Convert JSON to dataclass
def json_to_dataclass(json_str, cls):
    if not hasattr(cls, "__dataclass_fields__"):
        raise ValueError("Provided class is not a dataclass")
    if not json_str:
        return None
    data = json.loads(json_str)
    return cls(**data)


class DataFrameDataset(Dataset):
    def __init__(
        self, dataframe: Union[pd.DataFrame, HmDataFrameBase], transform=None, seek_base: int = 0
    ):
        """
        Args:
            dataframe (pd.DataFrame): a pandas DataFrame containing the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        self._seek_base: int = seek_base

    @property
    def batch_size(self) -> int:
        assert isinstance(self.dataframe, HmDataFrameBase)
        return self.dataframe.batch_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Any:
        idx = idx + self._seek_base
        if isinstance(self.dataframe, HmDataFrameBase):
            result = self.dataframe[idx]
            if self.transform is not None:
                result = self.transform(result)
            return result
        sample = self.dataframe.iloc[idx]
        if self.transform:
            sample = self.transform(sample)
        # Assuming the last column is the target variable
        iloc = sample.iloc
        return dict(features=iloc[:-1].values, target=iloc[-1])

    def __iter__(self) -> Iterable:
        return DataFrameDatasetIterator(dataset=self)

    def set_seek_base(self, pos: int):
        self._seek_base = pos


class DataFrameDatasetIterator:
    def __init__(self, dataset: DataFrameDataset) -> None:
        self._dataset = dataset
        self.position: int = 0
        self.length = len(dataset)

    def __next__(self) -> Any:
        if self.position >= self.length:
            raise StopIteration()
        pos: int = self.position
        self.position += 1
        return self._dataset[pos]


# @TRANSFORMS.register_module()
# class DataFrameReader:
#     def __init__(self, dataset_key, key: str, data_type: Type) -> None
#         self._key = key
#         self._data_type = data_type

#     def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         return data


def find_latest_dataframe_file(
    game_dir: Optional[str], stem: str, extension: str = ".csv"
) -> Optional[str]:
    """Return the newest CSV path for a dataframe in ``game_dir``.

    Looks for ``{stem}.csv`` and ``{stem}-N.csv`` files, returning the one with
    the highest numeric suffix (treating the bare filename as suffix 0).
    """
    if not game_dir:
        return None
    base_path = Path(game_dir)
    if not base_path.is_dir():
        return None
    candidates: List[Tuple[int, Path]] = []
    main_file = base_path / f"{stem}{extension}"
    if main_file.exists():
        candidates.append((0, main_file))
    pattern = f"{stem}-*{extension}"
    for path in base_path.glob(pattern):
        suffix = path.stem[len(stem) + 1 :]
        if not suffix.isdigit():
            continue
        candidates.append((int(suffix), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return str(candidates[0][1])
    return str(candidates[0][1])
