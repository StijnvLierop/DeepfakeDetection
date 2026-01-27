from typing import Iterable, Sized, Optional, List

import bisect

from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin


class CombinedDataset(MapStyleDatasetMixin, Dataset):
    """
    Helper class that can be used to combine multiple datasets into a single dataset.

    Can only be used with MapStyleDatasets.
    """

    def __init__(self, datasets: Iterable[MapStyleDatasetMixin], dataset_name: Optional[str] = None):
        """
        :param datasets: The datasets to combine.
        :param dataset_name: The name of the combined dataset.
        """
        super().__init__(dataset_name=dataset_name)

        # Convert to list to ensure stable ordering and indexing
        self.datasets = list(datasets)

        # Check if we can support Map-Style features (__len__ and __getitem__)
        self.is_map_style = all(isinstance(d, Sized) for d in self.datasets)

        # Calculate cumulative sizes based on dataset type
        if self.is_map_style:
            self.cumulative_sizes = self._calculate_cumulative_sizes()
        else:
            self.cumulative_sizes = []

    def _calculate_cumulative_sizes(self) -> List[int]:
        sizes = []
        current_total = 0
        for d in self.datasets:
            if not isinstance(d, Sized):
                raise ValueError(f"Dataset {d} must support __len__ to be used in Mapstyle.")
            current_total += len(d)
            sizes.append(current_total)
        return sizes

    def __len__(self) -> int:
        # Only return length of MapStyle datasets
        if not self.is_map_style:
            raise TypeError("CombinedDataset contains IterableDatasets and has no length.")
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        # Method only works for MapStyle datasets
        if not self.is_map_style:
            raise TypeError("Cannot index a CombinedDataset containing IterableDatasets.")

        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of range")

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][local_idx]