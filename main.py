import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import *
from sklearn.metrics import *
from matplotlib import pyplot as plt

from config import Config
from src.eda import EDA


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class CustomerSeg:
    def __init__(self, config):
        self.config = config
        self.data = pd.read_csv(config['data']['data_path']).sample(300)
        self.numeric_features_names = config["data"]["numeric_features"]
        self.category_features_names = config["data"]["category_features"]
        self.datetime_features_names = config["data"]["datetime_features"]
        self.model_algs = config["model"]
        self.tune = config["tuning"]["tune"]
        self.tune_alg = config["tuning"]["search_method"]
        self.param_grids = config["tuning"]
        self.metrics = config["evaluation"]
        self.data = self.data[
            self.numeric_features_names + self.category_features_names +\
                self.datetime_features_names
        ]

    def _preprocessing(self):
        """
        This method is use-case dependent
        """
        # Merge
        # self.data = pd.merge(self.data_sales, self.data_segment, 'left', 'member_id_masked')

        # Convert to dt format
        self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'], format='%Y%m%d')

        # Create new features
        self.data['total_sales_after_discount'] = self.data['sales_value_usd'] * self.data['sales_unit']
        self.data['discount_percent'] = self.data['discount_amount_usd'] / self.data['original_price_usd'] * 100
        self.data["day_of_week"] = self.data['transaction_date'].dt.dayofweek
        self.data['is_weekend'] = self.data["day_of_week"] > 4

        # Check dtpes and missing val
        self.data.info()

        # fill na with 0
        self.data.fillna(0, inplace=True)

    def run(self):
        self._preprocessing()
        EDA(self.data).run()
        features_to_drop = self.config['data']['features_to_drop_after_eda']
        self.data.drop(features_to_drop, axis=1, inplace=True)
        for feat in features_to_drop:
            if feat in self.numeric_features_names:
                self.numeric_features_names.remove(feat)
            if feat in self.category_features_names:
                self.category_features_names.remove(feat)
            if feat in self.datetime_features_names:
                self.datetime_features_names.remove(feat)

        # Training
        # Scaling
        scalar = StandardScaler()
        X = scalar.fit_transform(self.data)
        # PCA
        pca = self.config['model'].pop('PCA', None)
        principal_comp_count = 1e3
        if pca is not None:
            pca, threshold = pca
            X = pca.fit_transform(X)
            features = range(pca.n_components_)
            explained_var = pca.explained_variance_ratio_
            fig_pca, ax_pca = plt.subplots()
            ax_pca.bar(features, explained_var)
            ax_pca.set(xlabel='PCA features', ylabel='variance %', xticks=features)

            # Decide #principal components
            explained_var_cumsum = explained_var.cumsum()
            principal_comp_count = np.where(explained_var_cumsum > threshold)[0][0] + 1

        # Train clustering model
        inertias = []
        scores = []
        for model_alg_name, model_alg in self.config['model'].items():
            model_alg, cluster_counts = model_alg
            for cluster_count in cluster_counts:
                clusterer = model_alg(
                    n_clusters=cluster_count,
                    random_state=self.config['seed'])
                y = clusterer.fit_predict(
                        X[:, :principal_comp_count+1]
                    )
                score = self.metrics[0](X, y, **self.metrics[1])

                inertias.append(clusterer.inertia_)
                scores.append(score)

        fig_clusterer, ax_clusterer = plt.subplots(1, 2)
        ax_clusterer[0].plot(cluster_counts, inertias, '-o', color='black')
        ax_clusterer[0].set(
            xlabel='number of clusters, k', ylabel='inertia',
            xticks=cluster_counts)
        ax_clusterer[1].plot(cluster_counts, scores, '-o', color='black')
        ax_clusterer[1].set(
            xlabel='number of clusters, k', ylabel=f'{self.metrics[0].__name__}',
            xticks=cluster_counts)
        
        # Analyse based on best number of cluster
        # Analyse based on given cluster


if __name__ == '__main__':
    CustomerSeg(Config.train).run()
