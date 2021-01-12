from flask import Blueprint, jsonify

from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

endpoint = Blueprint('http_artifact_repository', __name__)
store = LocalArtifactRepository("artifact-store")


@endpoint.route('/<int:experiment_id>/<string:run_id>/artifacts/<path:path>', methods=["GET"])
def retrive_artifacts(experiment_id, run_id, path):
    # store = LocalArtifactRepository(f"store/{experiment_id}/{run_id}")
    # files = store.list_artifacts(path)
    return jsonify([])


@endpoint.route('/<int:experiment_id>/<string:run_id>/artifacts/<path:path>', methods=["POST"])
def log_artifacts(experiment_id, run_id, path):
    return jsonify([])
