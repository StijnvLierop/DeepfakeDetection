import hashlib

from deepfake_detection.data.instance import Instance


def hf_hash_filter(
    row: dict, column: str, range_min: float = 0.0, range_max: float = 1.0
) -> bool:
    """
    HuggingFace-level hash filter for reproducible splits without a split file.

    Hashes the value of `column` using MD5 (stable across processes and runs),
    maps it to a bucket in [0.0, 1.0), and keeps the row if the bucket falls
    within [range_min, range_max).

    Example YAML usage — keep 80 % of rows for training:
        hf_filter:
          func: deepfake_detection.data.utils.filter.hf_hash_filter
          params:
            column: file_name
            range_min: 0.0
            range_max: 0.8

    :param row: Raw HuggingFace row dict.
    :param column: Column whose value is hashed (e.g. a filename or unique ID).
    :param range_min: Lower bound of the kept hash range (inclusive), 0.0–1.0.
    :param range_max: Upper bound of the kept hash range (exclusive), 0.0–1.0.
    :return: True if the hash bucket falls within [range_min, range_max).
    """
    value = row.get(column)
    if value is None:
        return False
    digest = hashlib.md5(str(value).encode()).hexdigest()
    bucket = int(digest, 16) % 100 / 100
    return range_min <= bucket < range_max


def filter_on_hash_value(instance: Instance, range_min=0.0, range_max=0.1) -> bool:
    """
    Returns True if the image hash falls within [range_min, range_max].
    Values should be between 0.0 and 1.0.

    :param instance: Instance to assess.
    :param range_min: Lower bound of the hash range.
    :param range_max: Upper bound of the hash range.
    :return: True if the hash falls within the specified range.
    """
    # Use modulo 100 to get a stable bucket between 0 and 99
    # abs() ensures we handle negative hashes correctly
    hash_bucket = (abs(instance.__hash__()) % 100) / 100

    # Match the bucket with the configured range (0 - 1)
    return range_min <= hash_bucket < range_max
