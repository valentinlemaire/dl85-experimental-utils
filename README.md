# DL8.5 experimental utils

This package was constructed to enable easy experimentation with the pydl8.5 package, especially the version that allows for Quantile Regression, that package is available [here](https://github.com/valentinlemaire/pydl8.5.git).

## Contents

In this package you will find the following:
  1. The `DatasetBinarizer` class that will allow you to quickly binarize any dataset with both discrete and continous values by easily specifying how many bins are desired for each feature.
  2. The `DatasetCreator` class will allow you to quickly create an artificial dataset of binary features and gaussian mixtures as target column to see how a Quantile Regressor performs.
  3. The `DistributionRegressor` wrapper class that goes on top of the DL85QuantileRegressor class and enables it to create smooth pdf estimations rather than just quantile values.
  4. The `crps`, `mean_quantile_error`, `mise`and `log_likelihood` metric functions that were customized to easily adapt to smooth pdf and cdf predictions

## Installation
  
To install this package you must first clone this repository

``` 
git clone https://github.com/valentinlemaire/dl85-eperimental-utils
```

Then go inside the created folder and install the requirements with 

```
pip install -r requirements.txt
```

If the installation of pydl8.5 fails you must go to [this repository](https://github.com/valentinlemaire/pydl8.5) and follow the instructions to install it.

Finally, run this command to install the package

```
python setup.py install
