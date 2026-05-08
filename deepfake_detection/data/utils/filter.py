from deepfake_detection.data.instance import Instance


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
