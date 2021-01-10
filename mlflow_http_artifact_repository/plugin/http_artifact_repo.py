import logging
import os
import posixpath
import shutil

import requests

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.tracking._tracking_service import utils
from mlflow.utils.file_utils import relative_path_to_artifact_path
from requests_toolbelt.multipart.encoder import MultipartEncoder

logger = logging.Logger(__name__)

class HttpArtifactRepository(ArtifactRepository):
    """
    HTTP Artifact Gateway for MLflow. Artifact URI should provide RESTful APIs
    """

    def __init__(self, artifact_uri):
        super().__init__(artifact_uri)

    def log_artifact(self, local_file, artifact_path=None, *args, **kwargs):
        boundary = None if 'multipart_boundary' not in kwargs else kwargs['multipart_boundary']

        artifact_path = '' if not artifact_path else artifact_path
        local_path = posixpath.abspath(local_file)
        file_name = posixpath.basename(local_path)
        resource_uri = posixpath.join(self.artifact_uri, artifact_path, file_name)

        logger.info(f"log_artifact('{local_file}', '{artifact_path}') -> `{resource_uri}`")

        mpe = MultipartEncoder(fields={
            'artifacts': (file_name, open(local_file, 'rb')),
        }, boundary=boundary)
        resp = requests.post(resource_uri, headers={'Content-Type': mpe.content_type}, data=mpe)
        resp.raise_for_status()
        return resp.ok

    def log_artifacts(self, local_dir, artifact_path=None, *args, **kwargs):
        boundary = None if 'multipart_boundary' not in kwargs else kwargs['multipart_boundary']

        artifact_path = '' if not artifact_path else artifact_path
        local_path = posixpath.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_path):
            if not filenames: continue
            rel_path = os.path.relpath(root, local_path)

            resource_uri = posixpath.join(self.artifact_uri, artifact_path, rel_path)
            resource_uri = resource_uri if resource_uri.endswith("/") else f"{resource_uri}/"
            mpe = MultipartEncoder(fields=[
                ('artifacts', (file_name, open(posixpath.join(root, file_name), 'rb')))
            for file_name in filenames], boundary=boundary)
            resp = requests.post(resource_uri, headers={'Content-Type': mpe.content_type}, data=mpe)
            resp.raise_for_status()
            if not resp.ok: return False

        return True

    def list_artifacts(self, path=None):
        path = "" if not path else path
        resource_uri = posixpath.join(self.artifact_uri, path)
        resource_uri = resource_uri if resource_uri.endswith("/") else f"{resource_uri}/"
        logger.info(f"list_artifacts to `{resource_uri}`")
        resp = requests.get(resource_uri)
        resp.raise_for_status()

        entries = resp.json()
        logger.debug(f"list_artifacts('{path}') = {entries}")
        artifacts = [FileInfo(e['path'], e.get('is_dir', False), e.get('size', None)) for e in entries]
        return artifacts

    def _download_file(self, remote_file_path, local_path):
        download_url = posixpath.join(self.artifact_uri, remote_file_path)
        logger.info(f"download file from `{download_url}`")
        resp = requests.get(download_url, stream=True, headers={
            "Content-Type": "application/octet-stream",
        })
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
