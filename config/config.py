import numpy as np
from sklearn.decomposition import *
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Config(object):

    # Training config
    train = dict(

        data=dict(
            data_path="data/01_raw/dfs.csv",

            # To list only needed features
            numeric_features=[
                'sales_value_usd', 'sales_unit', 'original_price_usd',
                'discount_amount_usd',
            ],
            category_features=[
                'transaction_id', 'member_id_masked'
            ],
            datetime_features=[
                'transaction_date'
            ],
            features_to_drop_after_eda=[
                'transaction_id', 'member_id_masked',
                'transaction_date'
            ]
        ),

        model={
            PCA.__name__: [PCA(random_state=0), 0.8],  # alg, choose #components until explained var is larger
            KMeans.__name__: [KMeans, np.arange(2, 10)]  # alg, number of clusters
            # DBSCAN.__name__: DBSCAN(),
        },

        tuning={
            "tune": True,
            "search_method": GridSearchCV,  # RandomizedSearchCV, BayesSearchCV
            # PCA.__name__: dict(
            #     pca__n_components=[2, 5],
            #     ),
            KMeans.__name__: dict(
                model__n_clusters=[2, 5],
                ),
        },

        evaluation=[
            silhouette_score,
            dict(metric='euclidean', random_state=0)
            ],

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            experiment_name="Best Pipeline",
            run_name="trial",
            registered_model_name="my_clustering_model",
            port="5000",
        ),

        seed=0

    )

    # Prediction config
    predict = dict(

        data_path="data/01_raw/housing.csv",

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            model_name="my_clustering_model",
            port="5000",
            model_version="latest"
        ),
    )
