"""Compose editions keep independently managed models and persistent data intact."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def deployment(tmp_path):
    shutil.copy(ROOT / "deploy.sh", tmp_path / "deploy.sh")
    folder = tmp_path / "deployment/compose"
    folder.mkdir(parents=True)
    shutil.copy(ROOT / "deployment/compose/.env.example", folder / ".env.example")
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    docker.chmod(0o755)

    def run(*args):
        return subprocess.run(
            ["bash", str(tmp_path / "deploy.sh"), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": str(binary) + ":" + os.environ["PATH"],
                "CALLS": str(tmp_path / "calls"),
            },
            timeout=10,
        )

    return tmp_path, run


@pytest.mark.parametrize("edition", ["standard", "light"])
def test_edition_dependencies(edition):
    services = yaml.safe_load(
        (ROOT / "deployment/compose/docker-compose.yml").read_text()
    )["services"]
    active = {
        name
        for name, spec in services.items()
        if not spec.get("profiles") or edition in spec["profiles"]
    }
    assert active == (
        {"app", "mineru", "qdrant", "opensearch", "neo4j"}
        if edition == "standard"
        else {"app", "mineru"}
    )
    assert not (
        {"embedding", "reranker", "vl-embedding", "vl-reranker", "vlm"} & set(services)
    )
    assert not any("docker.sock" in str(x) for x in services["app"].get("volumes", []))
    for name in active:
        assert set(services[name].get("depends_on", {})) <= active


@pytest.mark.parametrize("edition", ["light", "standard"])
def test_repeat_configuration_preserves_credentials_and_model_endpoints(
    deployment, edition
):
    root, run = deployment
    app = root / "data/config/knowledge-base.env"
    app.parent.mkdir(parents=True)
    saved = f"KB_STORAGE_BACKEND={edition}\nVLM_BASE_URL=https://vision.example.test/v1\nVLM_MODEL_ID=custom-vision\nVLM_API_KEY=synthetic-key\n"
    app.write_text(saved)
    assert run("config", "--" + edition).returncode == 0
    compose = root / "deployment/compose/.env"
    before = compose.read_bytes()
    assert run("config").returncode == 0
    assert compose.read_bytes() == before
    assert app.read_text() == saved
    assert "NEO4J_PASSWORD=\n" not in compose.read_text()
    assert compose.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "saved,requested,success",
    [
        ("light", "", True),
        ("light", "light", True),
        ("light", "standard", False),
        ("standard", "light", False),
        ("", "", True),
    ],
)
def test_edition_fixed_per_directory(deployment, saved, requested, success):
    root, run = deployment
    state = root / "data/runtime"
    state.mkdir(parents=True)
    if saved:
        (state / "deployment-mode").write_text(saved)
    result = run("config", *(["--" + requested] if requested else []))
    assert (result.returncode == 0) == success, result.stderr


@pytest.mark.parametrize("edition", ["light", "standard"])
def test_up_down_manage_only_compose_and_retain_data(deployment, edition):
    root, run = deployment
    weights = root / "models/independent-model/weights.bin"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic")
    original = root / "data/runtime/mirror/sample.txt"
    original.parent.mkdir(parents=True)
    original.write_text("original")
    assert run("up", "--" + edition, "--build").returncode == 0
    assert run("down").returncode == 0
    calls = (root / "calls").read_text()
    assert (
        "compose build app mineru" in calls
        and "compose up -d" in calls
        and "compose down" in calls
    )
    assert not any(
        x in calls
        for x in ["remove-orphans", "--volumes", "systemctl", "docker rm", "prune"]
    )
    assert original.read_text() == "original" and weights.read_bytes() == b"synthetic"
    assert not (
        root / "data/runtime/qdrant_data"
    ).exists()  # Standard stores are named volumes.


def test_gpu_overlay_and_pdf_opt_in(deployment):
    root, run = deployment
    assert run("config", "--gpu", "--with-pdf-images").returncode == 0
    text = (root / "deployment/compose/.env").read_text()
    assert "COMPOSE_FILE=docker-compose.yml:compose.gpu.yml" in text
    assert "MINERU_FLAVOR=gpu" in text and "WITH_PDF_IMAGES=1" in text
    assert run("config", "--light", "--standard").returncode == 2
