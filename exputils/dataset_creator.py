import numpy as np
import pandas as pd
import os
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


pd.options.mode.chained_assignment = None  # default='warn'

dataset_filename_generator = (
    lambda n_features, n_samples, n_categories, sparsity, feature_noise_percentage, target_white_noise_variance, sorted_dataset, random_seed: f"{n_features}_{n_samples}_{n_categories}_{sparsity}_{feature_noise_percentage}_{target_white_noise_variance}_{sorted_dataset}_{random_seed}"
)


class DatasetCreator:
    def __init__(
        self,
        n_features,
        n_samples,
        n_categories,
        sparsity=1,
        feature_noise_percentage=0,
        target_white_noise_variance=0,
        sorted_dataset=False,
        force_recreation=False,
        random_seed:int = None
    ) -> None:
        """
        Creates a dataset with the given parameters.

        ------------
        Parameters:
        ------------

        n_features: int
            number of features in the dataset
        n_samples: int
            number of samples in the dataset
        n_categories: int
            number of categories in the dataset
        sparsity: float >= 0
            setting sparsity to 0 will make all categories follow the same distribution, setting it very high will make distributions in similar categories very different
        feature_noise_percentage: float between 0 and 1
            percentage of the data that is flipped
        target_white_noise_variance: float >= 0
            variance of the white noise in the target variable
        sorted_dataset: bool
            make the dataset sorted (easier to use when testing c++ code)
        force_recreation: bool
            force the dataset to be recreated
        """

        self.n_features = n_features
        self.n_samples = n_samples
        self.n_categories = n_categories
        self.sparsity = sparsity
        self.feature_noise_percentage = feature_noise_percentage
        self.target_white_noise_variance = target_white_noise_variance
        self.sorted_dataset = sorted_dataset
        self.force_recreation = force_recreation
        self.random_seed = random_seed
        self.rand = RandomState(MT19937(SeedSequence(self.random_seed)))

    def write_dataset(self):
        filename = dataset_filename_generator(
            self.n_features,
            self.n_samples,
            self.n_categories,
            self.sparsity,
            self.feature_noise_percentage,
            self.target_white_noise_variance,
            self.sorted_dataset,
            self.random_seed,
        )

        if self.force_recreation or not self.dataset_exists(filename):
            self.create_dataset(filename)

    def dataset_exists(self, filename):
        return f"dataset_{filename}.csv" in os.listdir(
            "."
        ) and f"distributions_{filename}.csv" in os.listdir(".")

    def create_dataset(self, filename):
        features = [f"feat_{i}" for i in range(self.n_features)]
        columns = [*features, "pred"]

        frame = pd.DataFrame(
            columns=features,
            data=np.zeros((self.n_samples, self.n_features), dtype=bool),
        )
        frame["pred"] = np.zeros(self.n_samples, dtype=float)

        category_weights = self.rand.poisson(lam=4, size=self.n_categories)
        category_weights = category_weights / np.sum(category_weights)

        index = self.rand.choice(
            range(self.n_categories), p=category_weights, size=self.n_samples
        )

        true_frame = pd.DataFrame(
            columns=[*features, "loc", "scale", "num"],
            data=np.zeros((self.n_categories, self.n_features + 3)),
        )

        chosen = set()
        features_for_categories = []
        distribution_info = []
        bin_words = []

        reference = None
        for cat in range(self.n_categories):
            cat_features = self.rand.choice(
                self.n_features, self.rand.choice(range(1, self.n_features + 1))
            )
            while tuple(cat_features) in chosen:
                cat_features = self.rand.choice(
                    self.n_features, self.rand.choice(range(1, self.n_features + 1))
                )
            chosen.add(tuple(cat_features))
            features_for_categories.append(np.array(features)[cat_features])

            binary_word = np.zeros(self.n_features, dtype=bool)
            binary_word[cat_features] = True

            if reference is None:
                distribution_info.append(
                    (self.rand.uniform(3, 15), self.rand.uniform(1, 4))
                )
            else:
                similarities = [
                    np.sum(~np.logical_xor(reference, binary_word))
                    for reference in bin_words
                ]
                i_max = np.argmin(similarities)

                loc_ref, scale_ref = distribution_info[i_max]
                difference = similarities[i_max]

                loc = loc_ref + self.rand.normal(0, self.sparsity * difference * 3)
                scale = scale_ref + self.rand.normal(0, self.sparsity * difference)

                distribution_info.append((loc, scale))
            bin_words.append(binary_word)

        info = [
            (np.where(index == cat)[0], cat_feat)
            for cat, cat_feat in enumerate(features_for_categories)
        ]
        info = zip(info, distribution_info, category_weights)

        for cat, ((cat_idx, cat_feat), (loc, scale), weight) in enumerate(info):
            cat_df = frame.iloc[cat_idx]
            cat_df[cat_feat] = np.ones((len(cat_idx), len(cat_feat)), dtype=bool)
            frame.iloc[cat_idx] = cat_df

            frame["pred"].iloc[cat_idx] = self.rand.normal(
                loc=loc, scale=scale, size=len(cat_idx)
            )

            true_frame.iloc[cat][cat_feat] = 1
            true_frame.iloc[cat]["loc"] = loc
            true_frame.iloc[cat]["scale"] = scale
            true_frame.iloc[cat]["num"] = weight

        mask = self.rand.choice(
            [True, False],
            size=(self.n_samples, self.n_features),
            p=[self.feature_noise_percentage, 1 - self.feature_noise_percentage], 
        )
        values = frame[features].values
        values[mask] = ~values[mask]
        frame[features] = values[:, :]
        frame["pred"] += self.rand.normal(
            loc=0, scale=self.target_white_noise_variance, size=self.n_samples
        )

        frame[features] = frame[features].astype(int)

        if self.sorted_dataset:
            sorted_idx = np.argsort(frame["pred"])
            frame = frame.iloc[sorted_idx]

        frame.to_csv(f"./dataset_{filename}.csv", index=False, header=None)
        true_frame.to_csv(f"./distributions_{filename}.csv")

        return frame


