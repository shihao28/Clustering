import numpy as np
from sklearn.decomposition import *
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Config(object):

    # Training config
    train = dict(

        data=dict(
            data_path="data/02_intermediate/dfs_merge.csv",

            # To list only needed features
            numeric_features=[
                'sales_value_usd', 'original_price_usd',
                'discount_amount_usd', 'sales_unit'
            ],
            category_features=[
                'transaction_id', 'member_id_masked',
                'customer_type'
            ],
            datetime_features=[
                'transaction_date'
            ],
        ),

        model={
            # PCA.__name__: [PCA(random_state=0), 0.8],  # alg, #components until explained var is larger
            KMeans.__name__: [KMeans, np.arange(2, 3)]  # alg, number of clusters
            # DBSCAN.__name__: DBSCAN(),
        },

        evaluation=[
            silhouette_score,  # metric function
            dict(metric='euclidean', random_state=0),  # metric parameters
            True,  # Set True to indicate the higher the metrics the better the cluster is
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

        data_path="data/01_raw/dfs.csv",

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            model_name="my_clustering_model",
            port="5000",
            model_version="latest"
        ),
    )
