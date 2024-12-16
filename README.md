# RANO-classification <!-- omit from toc -->

This repository contains the scripts used to classify the RANO criteria. 
It includes scripts for preprocessing and organizing of the data from the LUMIERE dataset.

## Table of Contents <!-- omit from toc -->
- [How to use](#how-to-use)
  - [1. Organize data Script](#1-organize-data-script)
  - [2. Run Preprocessing and Organization](#2-run-preprocessing-and-organization)
  - [3. Run Benchmarking Script](#3-run-benchmarking-script)


## How to use

First, clone the repository using:
```bash
git clone https://github.com/anamatoso/RANO-classification.git
```

Copy the LUMIERE folder into this folder.

Then create a virtual environment, activate it, and install the requirements:
```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
Check whether the packages were indeed installed using `pip list`. If not use `python3 -m pip install -r requirements.txt`.

### 1. Organize data Script

In the `LUMIERE-ExpertRating-v202211.csv` file in `line 172`, `line 578` and in `line 613` delete the extra space after "Post-Op". Additionally, change the "Date" header to "Timepoint"

### 3. Modify MONAI package

Before running the any file you must add a transform class to the monai package.

Add the following transformation to the file `./venv/lib/python[VERSION]/site-packages/monai/transforms/utility/dictionary.py` (replace [VERSION] with the one you're using, in my case it was 3.8) in `line 926` and add its name (`SubtractItemsd`) to `line 159`. Additionally, add its name also to the file `./venv/lib/python3.8/site-packages/monai/transforms/__init__.py` in `line 622` so that the package is aware of it.

```python
class SubtractItemsd(MapTransform):
    """
    Subtract specified items from data dictionary elementwise.
    Expect all the items are numpy array or PyTorch Tensor or MetaTensor.
    Return the first input's meta information when items are MetaTensor.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]


    def __init__(self, keys: KeysCollection, name: str, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be subtracted.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name corresponding to the key to store the resulting data.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.name = name

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Raises:
            TypeError: When items in ``data`` differ in type.
            TypeError: When the item type is not in ``Union[numpy.ndarray, torch.Tensor, MetaTensor]``.

        """
        d = dict(data)
        output = []
        data_type = None
        for key in self.key_iterator(d):
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])

        if len(output) == 0:
            return d

        if data_type is np.ndarray:
            d[self.name] = np.subtract(output[0], output[1])
        elif issubclass(data_type, torch.Tensor):  
            d[self.name] = torch.sub(output[0], output[1])
        else:
            raise TypeError(
                f"Unsupported data type: {data_type}, available options are (numpy.ndarray, torch.Tensor, MetaTensor)."
            )
        return d
```

### 2. Run Scripts

Run the scripts: note that each might take a while.
```bash
python ./01_preprocessing.py
python ./02_organize_data.py
```

Then you can run an experiment. An example of an experiment to run would be:

```bash
python RANO_benchmarking.py --model_name monai_densenet264 --n_epochs 100 --decrease_LR --stop_decrease --mods_keep T1,T2,FLAIR
```
