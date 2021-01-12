import uuid

from mlflow.entities import FileInfo
from requests_toolbelt.multipart.encoder import MultipartEncoder

from .http_artifact_repo import HttpArtifactRepository


def test_list_artifacts(httpserver):
    httpserver \
        .expect_request(
            "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/"
        ) \
        .respond_with_json([
            {"path": "model", "is_dir": True},
            {"path": "train.csv", "size": 15},
            {"path": "test.csv", "size": 14},
        ])
    httpserver \
        .expect_request(
            "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/model/"
        ) \
        .respond_with_json([
            {"path": "model/misc", "is_dir": True},
            {"path": "model/v1.pkl", "size": 255},
        ])
    httpserver \
        .expect_request(
            "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/model/misc/"
        ) \
        .respond_with_json([
            {"path": "model/misc/dummy.txt", "size": 0},
        ])
    httpserver \
        .expect_request(
            "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/model/misc/dummy.txt/"
        ) \
        .respond_with_json([])

    http_artifact_repo = HttpArtifactRepository(
        httpserver.url_for("/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts")
    )

    expected = sorted([
        FileInfo("model", True, None),
        FileInfo("train.csv", False, 15),
        FileInfo("test.csv", False, 14),
    ], key=lambda f: f.path)

    actual = sorted(http_artifact_repo.list_artifacts(), key=lambda f: f.path)
    for e, a in zip(expected, actual):
        assert e == a

    expected = sorted([
        FileInfo("model/misc", True, None),
        FileInfo("model/v1.pkl", False, 255),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts('model'), key=lambda f: f.path)
    for e, a in zip(expected, actual):
        assert e == a

    expected = sorted([
        FileInfo("model/misc/dummy.txt", False, 0),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts('model/misc'), key=lambda f: f.path)
    for e, a in zip(expected, actual):
        assert e == a

    actual = sorted(http_artifact_repo.list_artifacts('model/misc/dummy.txt'), key=lambda f: f.path)
    assert len(actual) == 0


def test_log_artifact(tmp_path, httpserver):
    train_csv = tmp_path / 'train.csv'
    train_csv.write_text(uuid.uuid4().hex.upper())
    test_csv = tmp_path / 'test.csv'
    test_csv.write_text(uuid.uuid4().hex.upper())
    models_path = tmp_path / "models"
    models_path.mkdir()
    models_v1_pkl = models_path / "v1.pkl"
    models_v1_pkl.write_text(uuid.uuid4().hex.upper())
    models_v2_pkl = models_path / "v2.pkl"
    models_v2_pkl.write_text(uuid.uuid4().hex.upper())

    multipart_boundary = 'test_log_artifact'
    train_csv_encoder = MultipartEncoder(fields={
        'artifacts': (train_csv.name, train_csv.read_text().encode())
    }, boundary=multipart_boundary)
    test_csv_encoder = MultipartEncoder(fields={
        'artifacts': (test_csv.name, test_csv.read_text().encode())
    }, boundary=multipart_boundary)
    models_v1_pkl_encoder = MultipartEncoder(fields={
        'artifacts': (models_v1_pkl.name, models_v1_pkl.read_text().encode())
    }, boundary=multipart_boundary)
    models_v2_pkl_encoder = MultipartEncoder(fields={
        'artifacts': (models_v2_pkl.name, models_v2_pkl.read_text().encode())
    }, boundary=multipart_boundary)

    http_artifact_repo = HttpArtifactRepository(
        httpserver.url_for("/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts")
    )

    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/v1.pkl",
        method="POST",
        headers={'Content-Type': models_v1_pkl_encoder.content_type},
        data=models_v1_pkl_encoder.to_string(),
    ).respond_with_json({"status": "success"})
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/v2.pkl",
        method="POST",
        headers={'Content-Type': models_v2_pkl_encoder.content_type},
        data=models_v2_pkl_encoder.to_string(),
    ).respond_with_json({"status": "success"})

    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/",
    ).respond_with_json([
        {"path": models_v1_pkl.name, "size": models_v1_pkl.stat().st_size},
        {"path": models_v2_pkl.name, "size": models_v2_pkl.stat().st_size},
    ])
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/data/train.csv",
        method="POST",
        headers={'Content-Type': train_csv_encoder.content_type},
        data=train_csv_encoder.to_string(),
    ).respond_with_json({"status": "success"})
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/data/test.csv",
        method="POST",
        headers={'Content-Type': test_csv_encoder.content_type},
        data=test_csv_encoder.to_string(),
    ).respond_with_json({"status": "success"})
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/",
    ).respond_with_json([
        {"path": "data", "is_dir": True},
        {"path": models_v1_pkl.name, "size": models_v1_pkl.stat().st_size},
        {"path": models_v2_pkl.name, "size": models_v2_pkl.stat().st_size},
    ])

    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/data/",
    ).respond_with_json([
        {"path": f"data/{test_csv.name}", "size": test_csv.stat().st_size},
        {"path": f"data/{train_csv.name}", "size": train_csv.stat().st_size},
    ])

    http_artifact_repo.log_artifact(str(models_v1_pkl), multipart_boundary=multipart_boundary)
    http_artifact_repo.log_artifact(str(models_v2_pkl), multipart_boundary=multipart_boundary)
    expected = sorted([
        FileInfo(models_v1_pkl.name, False, models_v1_pkl.stat().st_size),
        FileInfo(models_v2_pkl.name, False, models_v2_pkl.stat().st_size),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts(), key=lambda f: f.path)

    for e, a in zip(expected, actual):
        assert e == a

    assert http_artifact_repo.log_artifact(str(train_csv), "data", multipart_boundary=multipart_boundary)
    assert http_artifact_repo.log_artifact(str(test_csv), "data", multipart_boundary=multipart_boundary)
    expected = sorted([
        FileInfo("data", True, None),
        FileInfo(models_v1_pkl.name, False, models_v1_pkl.stat().st_size),
        FileInfo(models_v2_pkl.name, False, models_v2_pkl.stat().st_size),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts(), key=lambda f: f.path)
    for e, a in zip(expected, actual):
        assert e == a

    expected = sorted([
        FileInfo(f"data/{test_csv.name}", False, test_csv.stat().st_size),
        FileInfo(f"data/{train_csv.name}", False, train_csv.stat().st_size),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts("data"), key=lambda f: f.path)
    for e, a in zip(expected, actual):
        assert e == a


def test_log_artifacts(tmp_path, httpserver):
    train_csv = tmp_path / 'train.csv'
    train_csv.write_text(uuid.uuid4().hex.upper())
    test_csv = tmp_path / 'test.csv'
    test_csv.write_text(uuid.uuid4().hex.upper())
    models_path = tmp_path / "models"
    models_path.mkdir()
    models_v1_pkl = models_path / "v1.pkl"
    models_v1_pkl.write_text(uuid.uuid4().hex.upper())
    models_v2_pkl = models_path / "v2.pkl"
    models_v2_pkl.write_text(uuid.uuid4().hex.upper())

    multipart_boundary = 'test_log_artifacts'
    root_encoder = MultipartEncoder(fields=[
        ('artifacts', (test_csv.name, test_csv.read_text().encode())),
        ('artifacts', (train_csv.name, train_csv.read_text().encode())),
    ], boundary=multipart_boundary)
    models_encoder = MultipartEncoder(fields=[
        ('artifacts', (models_v2_pkl.name, models_v2_pkl.read_text().encode())),
        ('artifacts', (models_v1_pkl.name, models_v1_pkl.read_text().encode())),
    ], boundary=multipart_boundary)

    http_artifact_repo = HttpArtifactRepository(
        httpserver.url_for("/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts")
    )

    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/",
        method="POST",
        headers={'Content-Type': root_encoder.content_type},
        data=root_encoder.to_string(),
    ).respond_with_json({"status": "success"})
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/models/",
        method="POST",
        headers={'Content-Type': models_encoder.content_type},
        data=models_encoder.to_string(),
    ).respond_with_json({"status": "success"})
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/",
    ).respond_with_json([
        {"path": "models", "is_dir": True},
        {"path": train_csv.name, "size": train_csv.stat().st_size},
        {"path": test_csv.name, "size": test_csv.stat().st_size},
    ])
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/models/",
    ).respond_with_json([
        {"path": f"models/{models_v1_pkl.name}", "size": models_v1_pkl.stat().st_size},
        {"path": f"models/{models_v2_pkl.name}", "size": models_v2_pkl.stat().st_size},
    ])

    http_artifact_repo.log_artifacts(str(tmp_path), multipart_boundary=multipart_boundary)
    expected = sorted([
        FileInfo("models", True, None),
        FileInfo(train_csv.name, False, train_csv.stat().st_size),
        FileInfo(test_csv.name, False, test_csv.stat().st_size),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts(), key=lambda f: f.path)

    for e, a in zip(expected, actual):
        assert e == a

    expected = sorted([
        FileInfo(f"models/{models_v1_pkl.name}", False, models_v1_pkl.stat().st_size),
        FileInfo(f"models/{models_v2_pkl.name}", False, models_v2_pkl.stat().st_size),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts('models'), key=lambda f: f.path)

    for e, a in zip(expected, actual):
        assert e == a


def test_log_artifacts_under_out_dir(tmp_path, httpserver):
    train_csv = tmp_path / 'train.csv'
    train_csv.write_text(uuid.uuid4().hex.upper())
    test_csv = tmp_path / 'test.csv'
    test_csv.write_text(uuid.uuid4().hex.upper())
    models_path = tmp_path / "models"
    models_path.mkdir()
    models_v1_pkl = models_path / "v1.pkl"
    models_v1_pkl.write_text(uuid.uuid4().hex.upper())
    models_v2_pkl = models_path / "v2.pkl"
    models_v2_pkl.write_text(uuid.uuid4().hex.upper())

    multipart_boundary = 'test_log_artifacts'
    root_encoder = MultipartEncoder(fields=[
        ('artifacts', (test_csv.name, test_csv.read_text().encode())),
        ('artifacts', (train_csv.name, train_csv.read_text().encode())),
    ], boundary=multipart_boundary)
    models_encoder = MultipartEncoder(fields=[
        ('artifacts', (models_v2_pkl.name, models_v2_pkl.read_text().encode())),
        ('artifacts', (models_v1_pkl.name, models_v1_pkl.read_text().encode())),
    ], boundary=multipart_boundary)

    http_artifact_repo = HttpArtifactRepository(
        httpserver.url_for("/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts")
    )

    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/out/",
        method="POST",
        headers={'Content-Type': root_encoder.content_type},
        data=root_encoder.to_string(),
    ).respond_with_json({"status": "success"})
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/out/models/",
        method="POST",
        headers={'Content-Type': models_encoder.content_type},
        data=models_encoder.to_string(),
    ).respond_with_json({"status": "success"})
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/",
    ).respond_with_json([
        {"path": "out", "is_dir": True},
    ])
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/out/",
    ).respond_with_json([
        {"path": "models", "is_dir": True},
        {"path": train_csv.name, "size": train_csv.stat().st_size},
        {"path": test_csv.name, "size": test_csv.stat().st_size},
    ])
    httpserver.expect_ordered_request(
        "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/out/models/",
    ).respond_with_json([
        {"path": f"models/{models_v1_pkl.name}", "size": models_v1_pkl.stat().st_size},
        {"path": f"models/{models_v2_pkl.name}", "size": models_v2_pkl.stat().st_size},
    ])

    http_artifact_repo.log_artifacts(str(tmp_path), artifact_path='out', multipart_boundary=multipart_boundary)
    expected = sorted([
        FileInfo("out", True, None),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts(), key=lambda f: f.path)

    for e, a in zip(expected, actual):
        assert e == a

    expected = sorted([
        FileInfo("models", True, None),
        FileInfo(train_csv.name, False, train_csv.stat().st_size),
        FileInfo(test_csv.name, False, test_csv.stat().st_size),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts('out'), key=lambda f: f.path)

    for e, a in zip(expected, actual):
        assert e == a

    expected = sorted([
        FileInfo(f"models/{models_v1_pkl.name}", False, models_v1_pkl.stat().st_size),
        FileInfo(f"models/{models_v2_pkl.name}", False, models_v2_pkl.stat().st_size),
    ], key=lambda f: f.path)
    actual = sorted(http_artifact_repo.list_artifacts('out/models'), key=lambda f: f.path)

    for e, a in zip(expected, actual):
        assert e == a


def test__download_file(tmp_path, httpserver):
    httpserver \
        .expect_request(
            "/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts/model/v1.pkl"
        ) \
        .respond_with_data("some-weights", content_type="text/plain")

    http_artifact_repo = HttpArtifactRepository(
        httpserver.url_for("/api/1.0/artifact-repository/0/YyMOD18lNmU/artifacts")
    )

    download_path = tmp_path / 'v1.pkl'
    http_artifact_repo._download_file('model/v1.pkl', str(download_path))
    assert download_path.exists()
    assert 'some-weights' == download_path.read_text()
