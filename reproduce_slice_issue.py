import os
from typing import Optional, Mapping, Any, Union, List
import datasets
from datasets import Dataset as HFDataset
from datasets.features import Image
from deepfake_detection.data.datasets.huggingface import HuggingfaceDataset
from deepfake_detection.data.instance import ImageInstance
import PIL.Image

def test_huggingface_slice():
    # Create a small dummy HF dataset
    img = PIL.Image.new('RGB', (10, 10), color='red')
    data = {
        "image": [img, img, img],
        "label": [0, 1, 0]
    }
    hf_ds = HFDataset.from_dict(data).cast_column("image", Image())
    
    # Initialize our wrapper
    ds = HuggingfaceDataset(
        dataset=hf_ds,
        instance_col="image",
        label_cols={"label": "label_key"}
    )
    
    print("Testing single index access...")
    item = ds[0]
    print(f"Single item type: {type(item)}")
    assert isinstance(item, ImageInstance)
    
    print("Testing slice access...")
    try:
        items = ds[0:2]
        print(f"Slice items length: {len(items)}")
        assert len(items) == 2
        assert isinstance(items[0], ImageInstance)
        assert isinstance(items[1], ImageInstance)
        print("Slice access successful!")
    except Exception as e:
        print(f"Slice access failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_huggingface_slice()
