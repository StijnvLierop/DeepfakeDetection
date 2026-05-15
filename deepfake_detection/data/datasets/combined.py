import bisect
import random as _random
from typing import Iterable, List, Optional

from deepfake_detection.data.dataset import Dataset


def _is_map_style(dataset: Dataset) -> bool:
    """Returns True if the dataset supports __len__ and random access via __getitem__."""
    try:
        len(dataset)
        return True
    except TypeError:
        return False


class CombinedDataset(Dataset):
    """
    Combines multiple datasets into a single dataset.

    Sequential mode (default): datasets are concatenated in order. If all
    sub-datasets support map-style access, __len__ and __getitem__ are also
    available on the combined dataset.

    Interleaved mode (interleave=True or probabilities provided): samples are
    drawn round-robin or weighted-random across sub-datasets. Map-style access
    is not available in this mode. Iteration continues until all sub-datasets
    are exhausted.
    """

    def __init__(
        self,
        datasets: Iterable[Dataset],
        dataset_name: Optional[str] = None,
        interleave: bool = False,
        probabilities: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        :param datasets: Sub-datasets to combine.
        :param dataset_name: Name for this combined dataset.
        :param interleave: If True, yield samples round-robin across sub-datasets
                           instead of sequentially.
        :param probabilities: Per-dataset sampling weights for weighted interleaving.
                              Implies interleave=True. Must match len(datasets).
        :param seed: Random seed used for weighted interleaving.
        """
        super().__init__(dataset_name=dataset_name)
        self.datasets = list(datasets)
        self._interleave = interleave or (probabilities is not None)
        self._probabilities = probabilities
        self._seed = seed

        if self._probabilities is not None and len(self._probabilities) != len(self.datasets):
            raise ValueError(
                f"len(probabilities)={len(self._probabilities)} must match "
                f"len(datasets)={len(self.datasets)}."
            )

        # Map-style access requires sequential mode and all sub-datasets to be sized.
        self._map_style = not self._interleave and all(
            _is_map_style(d) for d in self.datasets
        )
        self._cumulative_sizes: List[int] = (
            self._build_cumulative_sizes() if self._map_style else []
        )

    def _build_cumulative_sizes(self) -> List[int]:
        sizes = []
        total = 0
        for d in self.datasets:
            total += len(d)
            sizes.append(total)
        return sizes

    def __len__(self) -> int:
        if not self._map_style:
            raise TypeError(
                "CombinedDataset is in interleaved mode or contains non-indexable "
                "datasets and has no defined length."
            )
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx):
        if not self._map_style:
            raise TypeError(
                "CombinedDataset is in interleaved mode or contains non-indexable "
                "datasets and does not support indexing."
            )
        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of range")
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        local_idx = idx if dataset_idx == 0 else idx - self._cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][local_idx]

    def __iter__(self):
        if not self._interleave:
            for dataset in self.datasets:
                yield from dataset
        elif self._probabilities is not None:
            yield from self._weighted_interleave()
        else:
            yield from self._round_robin()

    def _round_robin(self):
        iterators = [iter(d) for d in self.datasets]
        exhausted = [False] * len(iterators)
        while not all(exhausted):
            for i, it in enumerate(iterators):
                if not exhausted[i]:
                    try:
                        yield next(it)
                    except StopIteration:
                        exhausted[i] = True

    def _weighted_interleave(self):
        rng = _random.Random(self._seed)
        iterators = [iter(d) for d in self.datasets]
        active = list(range(len(self.datasets)))
        probs = list(self._probabilities)

        while active:
            active_weights = [probs[i] for i in active]
            chosen = rng.choices(active, weights=active_weights)[0]
            try:
                yield next(iterators[chosen])
            except StopIteration:
                active.remove(chosen)
