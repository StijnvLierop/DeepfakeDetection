import functools
import operator
import os
import pydoc
from typing import Optional, Mapping, Union

import datasets
from datasets import load_dataset, load_from_disk, IterableDataset
from datasets.features import Image

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import ImageInstance
from deepfake_detection.data.annotation import Annotation

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "in": lambda x, s: x in s,
}


def _build_hf_filter(filter_config: dict):
    """Builds a filter function that operates on raw HuggingFace row dicts."""
    if "func" in filter_config:
        func = pydoc.locate(filter_config["func"])
        if func is None:
            raise ValueError(f"Filter function {filter_config['func']} not found.")
        return functools.partial(func, **filter_config.get("params", {}))

    if "and" in filter_config:
        funcs = [_build_hf_filter(f) for f in filter_config["and"]]
        return lambda x: all(f(x) for f in funcs)

    if "or" in filter_config:
        funcs = [_build_hf_filter(f) for f in filter_config["or"]]
        return lambda x: any(f(x) for f in funcs)

    if "not" in filter_config:
        inner = _build_hf_filter(filter_config["not"])
        return lambda x: not inner(x)

    # Inline shorthand: {dotted.func.path: {params}} — equivalent to func/params style
    if len(filter_config) == 1:
        key, params = next(iter(filter_config.items()))
        func = pydoc.locate(key)
        if func is not None and callable(func):
            return functools.partial(func, **(params or {}))

    op = _OPS[filter_config["op"]]
    col = filter_config["label"]
    val = filter_config["value"]
    return lambda x: op(x.get(col), val)


class HuggingfaceDataset(Dataset):
    """
    Wraps a HuggingFace dataset as a deepfake_detection Dataset.

    HuggingFace-native operations (map, filter, shuffle, take) are specified via
    ``hf_ops`` and run in the listed order before wrapping, keeping streaming
    datasets efficient. Regular post-wrap operations (map, filter, sample) are
    applied by the configuration loader on the resulting Dataset instances.
    """

    def __init__(
        self,
        dataset: Union[str, datasets.Dataset],
        instance_col: str,
        label_cols: Optional[Mapping[str, str]] = None,
        mask_col: Optional[str] = None,
        dataset_name: Optional[str] = None,
        hf_ops: Optional[list] = None,
        in_memory: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        """
        :param dataset: HuggingFace dataset object, Hub identifier, or local path.
        :param instance_col: Column name containing the image data.
        :param label_cols: Mapping from HF column names to instance label keys.
        :param mask_col: Optional column name containing a grayscale binary forgery mask
                         (>128 = forged region). When set, the mask is attached to the
                         instance annotation and emitted as ``"masks"`` by TorchDataset.
        :param dataset_name: Name for this dataset.
        :param hf_ops: Ordered list of HF-level operations to apply before wrapping.
                       Each entry is a dict with one of these keys:
                         - map:    {func: "dotted.path", params: {...}}
                         - filter: <filter config (and/or/not/op/inline shorthand)>
                         - shuffle: {seed: 42, buffer_size: 1000}  (buffer_size streaming only)
                         - take:   <int>
                       Operations run in the order listed.
        :param in_memory: If True, materialize a streaming dataset into memory after all
                          hf_ops, enabling map-style access.
        :param num_proc: Number of processes for parallel map/filter (non-streaming only).
                         Defaults to None (single process). Set to os.cpu_count() for max
                         parallelism. Ignored for streaming datasets.
        :param **kwargs: Extra arguments forwarded to load_dataset / load_from_disk.
        """
        super().__init__(dataset_name=dataset_name)
        self.instance_col = instance_col
        self.label_cols = label_cols or {}
        self.mask_col = mask_col

        # If a dataset is already provided, use that
        if not isinstance(dataset, str):
            self._hf_dataset = dataset
        # If a directory with dataset_info.json exists, assume it's a saved dataset and load it from disk
        elif os.path.isdir(dataset) and os.path.exists(
            os.path.join(dataset, "dataset_info.json")
        ):
            self._hf_dataset = load_from_disk(dataset, **kwargs)
        # Otherwise, load the dataset from the Hub
        else:
            self._hf_dataset = load_dataset(dataset, **kwargs)

        # Get number of processes to use for map/filter operations
        streaming = isinstance(self._hf_dataset, IterableDataset)
        _num_proc = None if streaming else num_proc
        _proc_kwargs = {"num_proc": _num_proc} if _num_proc is not None else {}

        # Loop over the defined operations
        for op in hf_ops or []:
            if "map" in op:
                cfg = op["map"]
                func = pydoc.locate(cfg["func"])
                if func is None:
                    raise ValueError(f"Map function {cfg['func']} not found.")
                self._hf_dataset = self._hf_dataset.map(
                    functools.partial(func, **cfg.get("params", {})), **_proc_kwargs
                )
            elif "filter" in op:
                self._hf_dataset = self._hf_dataset.filter(
                    _build_hf_filter(op["filter"]), **_proc_kwargs
                )
            elif "shuffle" in op:
                cfg = op["shuffle"] or {}
                if isinstance(self._hf_dataset, IterableDataset):
                    self._hf_dataset = self._hf_dataset.shuffle(
                        seed=cfg.get("seed", 42),
                        buffer_size=cfg.get("buffer_size", 1000),
                    )
                else:
                    self._hf_dataset = self._hf_dataset.shuffle(seed=cfg.get("seed", 42))
            elif "take" in op:
                n = op["take"]
                if isinstance(self._hf_dataset, IterableDataset):
                    self._hf_dataset = self._hf_dataset.take(n)
                else:
                    self._hf_dataset = self._hf_dataset.select(
                        range(min(n, len(self._hf_dataset)))
                    )
            else:
                raise ValueError(
                    f"Unknown hf_ops entry (expected 'map', 'filter', 'shuffle', or 'take'): {op}"
                )

        # Materialize a streaming dataset into memory if requested
        if in_memory and isinstance(self._hf_dataset, IterableDataset):
            self._hf_dataset = datasets.Dataset.from_list(list(self._hf_dataset))

        self._streaming = isinstance(self._hf_dataset, IterableDataset)

        # Validate that all declared image columns are actually Image-typed.
        # features may be None for streaming datasets without a pre-defined schema.
        features = self._hf_dataset.features
        if features is not None:
            image_cols = [
                c for c in [instance_col, mask_col] if c is not None and c in features
            ]
            for col in image_cols:
                if not isinstance(features[col], Image):
                    raise ValueError(
                        f"Column {col!r} has type {features[col]!r} which is not supported. "
                        "Only Image columns are currently handled."
                    )
        self._instance_class = ImageInstance

    def __len__(self):
        if self._streaming:
            raise TypeError("Streaming HuggingFace datasets do not support len().")
        return len(self._hf_dataset)

    def __iter__(self):
        for sample in self._hf_dataset:
            yield self._make_instance(sample)

    def __getitem__(self, idx):
        if self._streaming:
            raise TypeError(
                "Streaming HuggingFace datasets do not support random access."
            )
        if isinstance(idx, slice):
            return [
                self._make_instance(self._hf_dataset[i])
                for i in range(*idx.indices(len(self)))
            ]
        return self._make_instance(self._hf_dataset[idx])

    def _make_instance(self, sample):
        labels = {
            label: sample[col]
            for col, label in self.label_cols.items()
            if col in sample
        }
        mask = None
        if self.mask_col is not None and self.mask_col in sample:
            mask = sample[self.mask_col].convert("L")
        annotation = (
            Annotation(labels=labels, mask=mask)
            if (labels or mask is not None)
            else None
        )
        return self._instance_class(
            data=sample[self.instance_col], annotation=annotation
        )
