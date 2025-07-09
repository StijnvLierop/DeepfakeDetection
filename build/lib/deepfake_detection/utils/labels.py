

def encode_label(label)-> int:
    """
    Encodes a string label to a numeric label.
    For now, only supports outputting binary labels 0 for "fake" and 1 for "real".

    :param label: String label.
    """
    # Define label_dict
    label_dict = {'fake': 0, 'real': 1}

    # Check if label is binary
    if label not in label_dict.keys():
        raise ValueError(f'Label {label} not recognized. Please ensure label is one of {label_dict.keys()}.')

    return label_dict[label]


