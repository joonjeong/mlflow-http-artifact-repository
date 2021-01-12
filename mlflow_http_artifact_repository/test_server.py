import pytest

from flask import Flask

from .server import endpoint


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(endpoint, url_prefix='/api/1.0/artifact-repository')

    return app.test_client()


def test_list_artifacts(client):
    resp = client.get('/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/model/')
    print(resp.data)
    assert False


def test_log_artifacts(client):
    assert False
