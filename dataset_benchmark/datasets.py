from abc import ABC, abstractmethod
from pathlib import Path
import time
from functools import partial
from uuid import uuid4

# from tqdm.autonotebook import tqdm, trange
from tqdm import tqdm, trange
import numpy as np
from datasets import Dataset as HfDataset
import datasets as hf_datasets
from torch.utils.data import Dataset as TorchDataset
import h5py


class BenchmarkDataset(ABC):
    @abstractmethod
    def __init__(self, n_rows: int, n_cols: int):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def delete(self):
        pass


class NpzBenchmarkDataset(BenchmarkDataset, TorchDataset):
    def __init__(self, n_rows: int, n_values_pre_row: int):
        self.target_dir = Path(__file__).parent.parent / "data" / "npz_temp"
        self.target_dir.mkdir(parents=True, exist_ok=True)

        for i in trange(n_rows, desc="Creating npz dataset", leave=False):
            np.savez(
                self.target_dir / f"{i}.npz",
                comumn1=np.random.rand(n_values_pre_row),
                column2=np.random.rand(),
            )

        self.files = list(self.target_dir.glob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return dict(np.load(self.files[idx]))

    def delete(self):
        for f in tqdm(self.files, desc="Deleting npz dataset", leave=False):
            # try for 5 seconds to delete the file
            for _ in range(5):
                try:
                    f.unlink()
                    break
                except Exception:
                    time.sleep(1)

            # try a final time
            if f.exists():
                f.unlink()


class HfArrowBenchmarkDataset(BenchmarkDataset):
    @staticmethod
    def data_generator(n_rows: int, n_values_per_row: int):
        for i in trange(n_rows, desc="Creating hf dataset", leave=False):
            yield {
                "column1": np.random.rand(n_values_per_row, 1),
                "column2": np.random.rand(),
            }

    def __init__(self, n_rows: int, n_values_per_row: int):
        self.cache_dir = (
            Path(__file__).parent.parent / "data" / "hf_temp" / f"{uuid4()}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        hf_datasets.disable_progress_bars()
        self.hf_dataset = HfDataset.from_generator(
            partial(self.data_generator, n_rows, n_values_per_row),
            cache_dir=self.cache_dir,
            features=hf_datasets.Features(
                {
                    "column1": hf_datasets.Array2D(
                        dtype="float64", shape=(n_values_per_row, 1)
                    ),
                    "column2": hf_datasets.Value(dtype="float64"),
                }
            ),
        )
        hf_datasets.enable_progress_bars()

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        return self.hf_dataset[idx]

    def delete(self):
        for cache_file in tqdm(
            self.hf_dataset.cache_files, desc="Deleting hf dataset", leave=False
        ):
            cache_file_path = Path(cache_file["filename"])
            if cache_file_path.exists():
                cache_file_path.unlink()

        for file in self.cache_dir.glob("*"):
            try:
                file.unlink()
            except Exception:
                pass


class Hdf5BenchmarkDatset(BenchmarkDataset, TorchDataset):
    def __init__(self, n_rows: int, n_values_per_row: int):
        self.file_path = (
            Path(__file__).parent.parent / "data" / "hdf5_temp" / "data.hdf5"
        )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.file_path, "w") as f:
            f.create_dataset(
                "column1",
                (n_rows, n_values_per_row),
                dtype="f",
            )
            f.create_dataset(
                "column2", (n_rows,), dtype="f", data=np.random.rand(n_rows)
            )

        # add the data to column1 without loading the whole dataset into memory
        with h5py.File(self.file_path, "r+") as f:
            for i in trange(n_rows, desc="Adding data to hdf5 dataset", leave=False):
                f["column1"][i] = np.random.rand(n_values_per_row)

        self.file = h5py.File(self.file_path, "r")

    def __len__(self) -> int:
        with h5py.File(self.file_path, "r") as f:
            return f["column1"].shape[0]

    def __getitem__(self, idx: int):
        return {
            "column1": self.file["column1"][idx],
            "column2": self.file["column2"][idx],
        }

    def delete(self):
        self.file.close()
        self.file_path.unlink()
