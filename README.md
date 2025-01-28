## Requirements

Python >= 3.10
Python-dev >= 3.10

## Installing Dependencies
All dependencies can then be installed by running the commands below. After starting your virtual environment:

```
pip install --upgrade pip
pip install -r requirements.txt
pip install -r test-requirements.txt
```

Note that certain libraries are installed with the `--no-deps` flag. 
This is because the full installation of these libraries would introduce conflicting dependencies.
The requirements that are needed for these libraries are specified in `requirements.txt`.