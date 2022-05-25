import pandas as pd
import numpy as np


class DatasetBinarizer:
    def __init__(self, mode="quantile", bins_per_feature=10):
        """
        Class that binarizes a dataset.

        Args:
            mode (str, optional):
                Controls whether to binarize the features with equal number of samples in each bin ("quantile" mode) or equal range for each bin ("range" mode). Defaults to "quantile".

            bins_per_feature (Union[int, dict[str, int]], optional):
                Controls how many bins are created per feature, if an int, then that number is used for all features, if a dict, then it must be in the format : col: bins and the keys must match the columns of the dataset. Defaults to 10. This argument is ignored for columns that are already categorical.
        """
        self.mode = mode
        self.bins_per_feature = bins_per_feature
        self.continuous_feature_ranges = {}
        self.categorical_feature_values = {}
        self.is_continuous = []

    def fit(self, X):
        """
        Fits the binarizer to the dataset.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The dataset to binarize.
        Returns:
            DatsetBinarizer: This instance.
        Raises:
            ValueError: If the bins_per_feature argument does not match the columns of the dataset.

        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        if isinstance(self.bins_per_feature, dict):
            if set(X.columns) != set(self.bins_per_feature.keys()):
                raise ValueError(
                    "The columns of the dataset do not match the keys of the bins_per_feature dict."
                )
        else:
            self.bins_per_feature = {col: self.bins_per_feature for col in X.columns}

        for col, type in zip(X.columns, X.dtypes):
            if (
                type != np.dtype("object")
                and len(np.unique(X[col])) > self.bins_per_feature[col]
            ):
                self.binarize_continuous_feature(X, col, self.bins_per_feature[col])
                self.is_continuous.append(True)
            else:
                self.binarize_categorical_feature(X, col)
                self.is_continuous.append(False)

        return self

    def binarize_continuous_feature(
        self, X: pd.DataFrame, col: str, bins: int
    ) -> pd.DataFrame:
        """
        Binarizes a continuous feature.

        Args:
            X (pd.DataFrame): The full dataset.
            col (str): The column to binarize.
            bins (int): The number of bins to create.

        Returns:
            pd.DataFrame: A binarized dataset for the relevant feature.
        """
        series = X[col]

        end = series.min() - 1

        self.continuous_feature_ranges[col] = []

        for i in range(bins):
            start = end

            if self.mode == "quantile":
                end = np.quantile(series, q=(i + 1) / bins)
            if self.mode == "range":
                end = series.min() + (series.max() - series.min()) * (i + 1) / bins

            if end <= start:
                end = start + 1

            idx = series[(series > start) & (series <= end)].index

            if len(idx) != 0:
                new = pd.Series(index=X.index, data=np.zeros(len(X)), dtype=int)
                new[idx] = 1

                self.continuous_feature_ranges[col].append((start, end))

    def binarize_categorical_feature(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Binarizes a categorical feature.

        Args:
            X (pd.DataFrame): The full dataset to binarize.
            col (str): The feature to binarize.

        Returns:
            pd.DataFrame: A binarized dataset for the relevant feature.
        """
        series = X[col]
        self.categorical_feature_values[col] = np.unique(series)

    def transform(self, X) -> pd.DataFrame:
        """Transforms the dataset.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The dataset to transform.

        Returns:
            pd.DataFrame: A binarized dataset.
        """

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        new_df = pd.DataFrame(index=X.index, dtype=int)

        for i, col in enumerate(X.columns):
            if self.is_continuous[i]:
                new_df = pd.concat(
                    [new_df, self.transform_continuous_feature(X, col)], axis=1
                )

            else:
                new_df = pd.concat(
                    [new_df, self.transform_categorical_feature(X, col)], axis=1
                )
        return new_df

    def transform_continuous_feature(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Transforms a continuous feature.

        Args:
            X (pd.DataFrame): The full dataset.
            col (str): The column to transform.

        Returns:
            pd.DataFrame: A binarized dataset for the relevant feature.
        """

        series = X[col]
        new_df = pd.DataFrame(index=X.index, dtype=int)

        for start, end in self.continuous_feature_ranges[col]:
            idx = series[(series > start) & (series <= end)].index
            new = pd.Series(index=X.index, data=np.zeros(len(X)), dtype=int)
            new[idx] = 1
            new_df[f"{start:.2f}<{col}<={end:.2f}"] = new

        return new_df

    def transform_categorical_feature(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Transforms a categorical feature.

        Args:
            X (pd.DataFrame): The full dataset.
            col (str): The column to transform.

        Returns:
            pd.DataFrame: A binarized dataset for the relevant feature.
        """

        series = X[col]
        new_df = pd.DataFrame(index=X.index, dtype=int)

        for value in self.categorical_feature_values[col]:
            idx = series[series == value].index
            new = pd.Series(index=X.index, data=np.zeros(len(X)), dtype=int)
            new[idx] = 1
            new_df[f"{col}={value}"] = new

        return new_df

    def fit_transform(self, X) -> pd.DataFrame:
        """
        Fits the binarizer and then transforms the dataset.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The dataset to transform.

        Returns:
            pd.DataFrame: A binarized dataset.
        """

        return self.fit(X).transform(X)
