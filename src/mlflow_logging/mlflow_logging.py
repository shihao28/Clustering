import os
import pandas as pd
from subprocess import Popen, DEVNULL
import mlflow
from mlflow.models.signature import infer_signature
from pathlib import Path
import shutil
import requests
import time
import logging


class MlflowLogging:
    def __init__(
        self, tracking_uri, backend_uri, artifact_uri, mlflow_port,
            experiment_name=None, run_name=None, registered_model_name=None):

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.registered_model_name = registered_model_name

        env = {
            "MLFLOW_TRACKING_URI": f"{tracking_uri}:{mlflow_port}",
            "BACKEND_URI": backend_uri,
            "ARTIFACT_URI": artifact_uri,
            "MLFLOW_PORT": mlflow_port
            }
        os.environ.update(env)
        self.cmd_mlflow_server = (
            f"mlflow server --backend-store-uri {backend_uri} "
            f"--default-artifact-root {artifact_uri} "
            f"--host 0.0.0.0 -p {mlflow_port}")

    def activate_mlflow_server(self):
        with open("stderr.txt", mode="wb") as out, open("stdout.txt", mode="wb") as err:
            Popen(self.cmd_mlflow_server, stdout=out, stderr=err, stdin=DEVNULL,
                  universal_newlines=True, encoding="utf-8",
                  env=os.environ, shell=True)

        # Keep pinging until mlfow server is up
        while True:
            try:
                response = requests.get(f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/experiments/list')
                if str(response) == "<Response [200]>":
                    logging.warning(f'MLFLOW tracking server response: {str(response)}')
                    break
            except requests.exceptions.ConnectionError:
                logging.warning(f'Tracking server {os.getenv("MLFLOW_TRACKING_URI")} is not up and running')
                time.sleep(1)

    def logging(
        self, X, y, scaler, pca, principal_comp_count, explained_variance,
        best_clusterer, best_cluster_count,
        best_inertia, metric_name, best_metric, fig_eda, fig_pca, fig_clusterer,
        fig_postanalysis):

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            use_pca = False if pca is None else True
            mlflow.log_param('use_pca', use_pca)
            if use_pca:
                mlflow.log_metric('principal_comp_count', principal_comp_count)
                mlflow.log_metric('explained_variance', explained_variance)
            mlflow.log_metric('best_cluster_count', best_cluster_count,)
            mlflow.log_metric('best_inertia', best_inertia,)
            mlflow.log_metric(metric_name, best_metric,)

            # Log scaler alg
            mlflow.sklearn.log_model(
                sk_model=scaler, artifact_path="sk_models",
                registered_model_name='my_std_scaler')

            # Log model
            signature = infer_signature(X, y)
            mlflow.sklearn.log_model(
                sk_model=best_clusterer, artifact_path="sk_models",
                signature=signature, input_example=X.sample(5),
                registered_model_name=self.registered_model_name
                )

            # Store plots as artifacts
            artifact_folder = Path("mlflow_tmp")
            artifact_folder.mkdir(parents=True, exist_ok=True)

            # Storing only figures, pd.DataFrames are excluded
            all_figs = fig_eda + (fig_pca,) + (fig_clusterer,) + fig_postanalysis
            for i, fig in enumerate(all_figs):
                fig.savefig(Path(artifact_folder, f"{i}.png"))
            mlflow.log_artifacts(
                artifact_folder, artifact_path="evaluation_artifacts")
            # shutil.rmtree(artifact_folder)
