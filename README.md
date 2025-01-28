## Requirements

Python >= 3.10
Python-dev >= 3.10

## Installing Dependencies
All dependencies can then be installed by running the commands below. After starting your virtual environment:

```
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-no-deps.txt --no-deps 
pip install -r test-requirements.txt
```

Note that certain libraries are installed with the `--no-deps` flag. 
This is because the full installation of these libraries would introduce conflicting dependencies.
The requirements that are needed for these libraries are specified in `requirements.txt`.

## ImageBind Model

This repository implements the [ImageBind](https://github.com/facebookresearch/ImageBind) model developed by Meta. 
The ImageBind model is a model that has learned a joint embedding across six different modalities. 
This makes it possible to input any of these six modalities and produce a same-size embedding vector that can be used for multimodal tasks. 
Currently, the implementation in this repository supports image, audio and text embeddings using the ImageBind model. 
More info about the model (including a demo of its capabilities) can be found [here](https://imagebind.metademolab.com/). 
A [model card](https://github.com/facebookresearch/ImageBind/blob/main/model_card.md) is also available.

The model uses the ``multimodal/models/embedding/imagebind_resources/bpe/bpe_simple_vocab_16e6.txt.gz`` file to produce text embeddings.
This file was copied from the ImageBind repository.
The original can be found [here](https://github.com/facebookresearch/ImageBind/tree/main/bpe).

You can easily update this file by running the following command:
```
wget -P -O multimodal/models/embedding/imagebind_resources/bpe -S --header="accept-encoding: gzip" https://github.com/facebookresearch/ImageBind/blob/main/bpe/bpe_simple_vocab_16e6.txt.gz
```
