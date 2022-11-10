import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import *
from sklearn.cluster import *
from sklearn.metrics import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import *
import seaborn as sns

from config import Config
from src.preprocessing import Preprocessing
from src.eda import EDA
from src.post_analysis import PostAnalysis
from src.mlflow_logging import MlflowLogging


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class TrainClustering:
    def __init__(self, config):
        self.config = config
        self.data = pd.read_csv(config['data']['data_path'])#.sample(1000, random_state=0)
        self.numeric_features_names = config["data"]["numeric_features"]
        self.category_features_names = config["data"]["category_features"]
        self.datetime_features_names = config["data"]["datetime_features"]
        self.model_algs = config["model"]
        self.metrics = config["evaluation"]
        self.data = self.data[
            self.numeric_features_names + self.category_features_names +
                self.datetime_features_names
        ]

    def _train(self, X):
        logging.info("Training...")
        # Scaling
        X[['total_sales_after_discount_mean', 'frequency', 'monetary']] =\
            np.log(X[['total_sales_after_discount_mean', 'frequency', 'monetary']])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # PCA
        pca = self.config['model'].pop('PCA', None)
        principal_comp_count = 10000
        explained_variance = 0
        fig_pca, ax_pca = plt.subplots()
        if pca is not None:
            pca, threshold = pca
            X = pca.fit_transform(X)
            features = range(pca.n_components_)
            explained_var = pca.explained_variance_ratio_
            ax_pca.bar(features, explained_var)
            ax_pca.set(xlabel='PCA features', ylabel='variance %', xticks=features)

            # Decide #principal components
            explained_var_cumsum = explained_var.cumsum()
            principal_comp_count = np.where(explained_var_cumsum > threshold)[0][0] + 1
            explained_variance = explained_var_cumsum[principal_comp_count-1]

            logging.info(f"principal_comp_count = {principal_comp_count}")
            logging.info(f"explain_var = {explained_variance:.2f}")

        # Train clustering model
        inertias = []
        metrics = []
        clusterers = []
        labels = []
        for model_alg_name, model_alg in self.config['model'].items():
            model_alg, cluster_counts = model_alg
            for cluster_count in tqdm(cluster_counts):
                clusterer = model_alg(
                    n_clusters=cluster_count,
                    random_state=self.config['seed'])
                y = clusterer.fit_predict(
                        X[:, :principal_comp_count+1]
                    )
                metric = self.metrics[0](X, y, **self.metrics[1])

                inertias.append(clusterer.inertia_)
                metrics.append(metric)
                clusterers.append(clusterer)
                labels.append(y)

        # Plot inertia and clustering metrics
        fig_clusterer, ax_clusterer = plt.subplots(1, 2)
        ax_clusterer[0].plot(cluster_counts, inertias, '-o', color='black')
        ax_clusterer[0].set(
            xlabel='number of clusters, k', ylabel='inertia',
            xticks=cluster_counts)
        ax_clusterer[1].plot(cluster_counts, metrics, '-o', color='black')
        ax_clusterer[1].set(
            xlabel='number of clusters, k',
            ylabel=f'{self.metrics[0].__name__}',
            xticks=cluster_counts)

        # Determine best number of cluster
        # if True, as_scorer means the higher the better the cluster is
        as_scorer = self.metrics[2]
        metrics = np.array(metrics)
        if as_scorer:
            idx = np.argmax(metrics)
        else:
            idx = np.argmin(metrics)
        best_clusterer = clusterers[idx]
        best_cluster_count = best_clusterer.n_clusters
        best_inertia = inertias[idx]
        best_metric = metrics[idx]
        y = labels[idx]

        logging.info(f"best_cluster_count = {best_cluster_count}")
        logging.info(f"best_inertia = {best_inertia:.2f}")
        logging.info(f"best_metric ({self.metrics[0].__name__}) = {best_metric:.2f}")

        return (scaler, pca, principal_comp_count, explained_variance,
                best_clusterer, best_cluster_count, best_inertia, best_metric,
                y, fig_pca, fig_clusterer)

    def _mlflow_logging(
        self, X, y, scaler, pca, principal_comp_count, explained_variance,
        best_clusterer, best_cluster_count,
        best_inertia, best_metric, fig_eda, fig_pca, fig_clusterer,
        fig_postanalysis):
        logging.info("Logging to mlflow...")
        mlflow_logging = MlflowLogging(
            tracking_uri=self.config["mlflow"]["tracking_uri"],
            backend_uri=self.config["mlflow"]["backend_uri"],
            artifact_uri=self.config["mlflow"]["artifact_uri"],
            mlflow_port=self.config["mlflow"]["port"],
            experiment_name=self.config["mlflow"]["experiment_name"],
            run_name=self.config["mlflow"]["run_name"],
            registered_model_name=self.config["mlflow"]["registered_model_name"]
        )
        mlflow_logging.activate_mlflow_server()
        mlflow_logging.logging(
            X, y, scaler, pca, principal_comp_count, explained_variance,
            best_clusterer, best_cluster_count,
            best_inertia, self.metrics[0].__name__, best_metric,
            fig_eda, fig_pca, fig_clusterer, fig_postanalysis
            )

        return None

    def run(self):
        # Preprocessing
        # customers = Preprocessing(self.data).run()
        customers = pd.read_csv('data/03_primary/customers.csv')

        # EDA
        fig_eda = EDA(customers, scale=False).run()

        # Outlier removal
        # first_quantile = customers['monetary'].quantile(0.25)
        # third_quantile = customers['monetary'].quantile(0.75)
        # iqr = third_quantile - first_quantile
        # lower_bound = first_quantile - 1.5*iqr
        # upper_bound = first_quantile + 1.5*iqr
        # customers = customers.loc[(customers['monetary'] >= lower_bound) & (customers['monetary'] <=upper_bound), ]

        # Feature Selection
        X = customers.drop([
            # Drop member_id_masked and customer_type since they are irrelevant
            'member_id_masked', 'customer_type',

            # total_sales_after_discount, original_price_usd, discount_amount_usd are correlated
            # Choose total total_sales_after_discount_mean over original_price and discount_amount_usd as it encapsulates info from the rest
            'original_price_usd_min', 'original_price_usd_mean', 'original_price_usd_median', 'original_price_usd_max',
            'discount_amount_usd_min', 'discount_amount_usd_mean', 'discount_amount_usd_median', 'discount_amount_usd_max',

            # total_sales_after_discount_mean and total_sales_after_discount_median are correlated
            'total_sales_after_discount_median',

            # sales_value_usd and original_price_usd are correlated. Choose sales_value_usd
            'sales_value_usd_median',

            # 'sales_value_usd_min', 'sales_value_usd_mean', 'sales_value_usd_median' 'sales_value_usd_max',

            # weekday_transaction_count_percent and weekend_transaction_count_percent carry same information
            'weekday_transaction_count_percent',

            # sales_unit, discount_percent, weekend_transaction_count_percent, relevancy_in_days are not correlated with others
            'sales_unit_median', 'discount_percent_median',

            # frequency and day<x>_transaction are correlated. Choose frequency
            # day<x>_transaction and monetary are correlated. Choose monetary
            'day0_transaction', 'day1_transaction', 'day2_transaction', 'day3_transaction',
            'day4_transaction', 'day5_transaction', 'day6_transaction',

            # Choose rfm_score over recency_score, frequency_score and monetary_score
            'recency_score', 'frequency_score', 'monetary_score',

            # try
            'sales_value_usd_min', 'sales_value_usd_mean', 'sales_value_usd_max',
            'sales_unit_min', 'sales_unit_mean', 'sales_unit_max',
            'total_sales_after_discount_min', 'total_sales_after_discount_max',
            'discount_percent_min', 'discount_percent_mean', 'discount_percent_max',
            'rfm_score', 'weekend_transaction_count_percent',
            ], axis=1)

        # Training
        scaler, pca, principal_comp_count, explained_variance, best_clusterer,\
            best_cluster_count, best_inertia, best_metric, y, fig_pca,\
                fig_clusterer = self._train(X)

        # Analyse based on best number of cluster
        fig_postanalysis = PostAnalysis(customers, best_clusterer, y).run()

        # calculate accuracy
        y_true = customers.loc[~customers['customer_type'].isna(), 'customer_type'].values
        y_pred = y[~customers['customer_type'].isna()]
        y_pred = np.array(list(map(lambda x: 'A' if x==0 else 'B', y_pred)))
        np.unique(y_true, return_counts=True)
        np.unique(y_pred, return_counts=True)
        logging.info(f'\n{rand_score(y_true, y_pred):.2f}')
        cf_matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(cf_matrix, annot=True)

        # Mlflow logging
        self._mlflow_logging(
            X, y, scaler, pca, principal_comp_count, explained_variance,
            best_clusterer, best_cluster_count,
            best_inertia, best_metric, fig_eda, fig_pca, fig_clusterer,
            fig_postanalysis
        )

        return None


if __name__ == '__main__':
    TrainClustering(Config.train).run()
