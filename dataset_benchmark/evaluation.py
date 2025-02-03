from typing import Type, List, TypedDict, Dict
import socket
import time
from datetime import datetime

from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
import numpy as np

from dataset_benchmark.datasets import BenchmarkDataset

ReadBenchmarkDatapoint = TypedDict(
    "BenchmarkDatapoint",
    {
        "n_rows": int,
        "n_values_per_row": int,
        "shuffle": bool,
        "batch_size": int,
        "n_dataloader_workers": int,
        "dataloder_pin_memory": bool,
        "time": float,
        "date": int,
        "computer": str,
        "dataset_type": str,
    },
)


def evaluate_plain_loading(
    dataset_classes: List[Type[BenchmarkDataset]],
    n_rows: List[int],
    n_values_per_row: List[int],
    shuffle: List[bool],
    max_dataset_size: int = 100_000_000_000,
    n_repeats: int = 3,
    dataset_init_kwargs: List[Dict] = None,
) -> List[ReadBenchmarkDatapoint]:
    dataset_init_kwargs = dataset_init_kwargs or [{} for _ in dataset_classes]

    datapoints = []

    for i_dataclass, _dataset_class in enumerate(
        tqdm(dataset_classes, desc="dataset_class", leave=False)
    ):
        for _n_rows in tqdm(n_rows, desc="n_rows", leave=False):
            for _n_values_per_row in tqdm(
                n_values_per_row, desc="n_values_per_row", leave=False
            ):
                for _shuffle in tqdm(shuffle, desc="shuffle", leave=False):
                    if _n_rows * _n_values_per_row * 8 > max_dataset_size:
                        continue

                    dataset = _dataset_class(
                        _n_rows, _n_values_per_row, **dataset_init_kwargs[i_dataclass]
                    )

                    for _ in trange(n_repeats, desc="repeats", leave=False):
                        indices = list(range(len(dataset)))
                        if _shuffle:
                            np.random.shuffle(indices)

                        start = time.time()
                        for i in tqdm(indices, desc="reading", leave=False):
                            _ = dataset[i]
                        end = time.time()

                        datapoints.append(
                            {
                                "n_rows": _n_rows,
                                "n_values_per_row": _n_values_per_row,
                                "shuffle": _shuffle,
                                "batch_size": None,
                                "n_dataloader_workers": None,
                                "dataloder_pin_memory": None,
                                "time": end - start,
                                "date": datetime.now().timestamp(),
                                "computer": socket.gethostname(),
                                "dataset_type": _dataset_class.__name__,
                            }
                        )

                    dataset.delete()

    return datapoints


def evaluate_batched_loading(
    dataset_classes: List[Type[BenchmarkDataset]],
    n_rows: int,
    n_values_per_row: int,
    shuffle: List[bool],
    batch_sizes: List[int],
    n_dataloader_workers: List[int],
    dataloder_pin_memory: List[bool],
    max_batch_size: int = 4_000_000_000,
    max_dataset_size: int = 100_000_000_000,
    n_repeats: int = 3,
    dataset_init_kwargs: List[Dict] = None,
) -> List[ReadBenchmarkDatapoint]:
    dataset_init_kwargs = dataset_init_kwargs or [{} for _ in dataset_classes]

    if n_rows * n_values_per_row * 8 > max_dataset_size:
        raise ValueError("Dataset too large")

    datapoints = []

    for i_dataset_class, _dataset_class in enumerate(
        tqdm(dataset_classes, desc="dataset_class", leave=False)
    ):
        dataset = _dataset_class(
            n_rows, n_values_per_row, **dataset_init_kwargs[i_dataset_class]
        )

        for _batch_size in tqdm(batch_sizes, desc="batch_size", leave=False):
            for _n_dataloader_workers in tqdm(
                n_dataloader_workers, desc="n_dataloader_workers", leave=False
            ):
                for _dataloder_pin_memory in tqdm(
                    dataloder_pin_memory, desc="dataloder_pin_memory", leave=False
                ):
                    for _shuffle in tqdm(shuffle, desc="shuffle", leave=False):
                        if _batch_size * n_values_per_row * 8 > max_batch_size:
                            continue

                        for _ in trange(n_repeats, desc="repeats", leave=False):
                            dataloader = DataLoader(
                                dataset,
                                batch_size=_batch_size,
                                shuffle=_shuffle,
                                num_workers=_n_dataloader_workers,
                                pin_memory=_dataloder_pin_memory,
                            )

                            start = time.time()
                            for batch in tqdm(
                                dataloader, desc="reading batches", leave=False
                            ):
                                pass
                            end = time.time()

                            datapoints.append(
                                {
                                    "n_rows": n_rows,
                                    "n_values_per_row": n_values_per_row,
                                    "shuffle": _shuffle,
                                    "batch_size": _batch_size,
                                    "n_dataloader_workers": _n_dataloader_workers,
                                    "dataloder_pin_memory": _dataloder_pin_memory,
                                    "time": end - start,
                                    "date": datetime.now().timestamp(),
                                    "computer": socket.gethostname(),
                                    "dataset_type": _dataset_class.__name__,
                                }
                            )

        dataset.delete()

    return datapoints


# def save_results(datapoints: List[BenchmarkDatapoint], prefix: str = "results"):
#     df = pd.DataFrame(datapoints)

#     results_path = (
#         Path.cwd().parent
#         / "results"
#         / f"{prefix}_{df['dataset_type'].iloc[0]}_{df['computer'].iloc[0]}_{df['date'].iloc[0]}.csv"
#     )
#     if results_path.parent.exists():
#         df.to_csv(results_path, mode="a", header=False, index=False)
#     else:
#         results_path.parent.mkdir(parents=True, exist_ok=True)
#         df.to_csv(results_path, index=False)
