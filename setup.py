from setuptools import setup, find_packages

setup(
    name="mlflow-http-artifact-repository",
    version="1.0.0",
    description="Http Artifact Repository Server for MLflow",
    packages=find_packages('.'),
    install_requires=["mlflow"],
    setup_requires=["pytest", "pytest-runner", "pytest-httpserver"],
    entry_points={
        "mlflow.http_artifact_repository": [
            "http=mlflow_artifact_repository.client.http_artifact_repository_proxy:HttpArtifactRepository",
            "https=mlflow_artifact_repository.client.http_artifact_repository_proxy:HttpArtifactRepository",
        ],
    },
)
