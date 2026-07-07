"""Image source validation for the vision path (issue #74): only data: URIs are
accepted by default; remote URLs and local paths require an explicit opt-in env var,
and are resolved to a PIL.Image *before* anything reaches the GPU worker.
"""

import mlx_vlm.utils as mlx_vlm_utils
import pytest

from ember import server


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Both opt-ins start off, matching the shipped defaults."""
    monkeypatch.setattr(server, "ALLOW_IMAGE_URLS", False)
    monkeypatch.setattr(server, "ALLOW_IMAGE_PATHS", False)


class _FakeImage:
    """Stands in for a PIL.Image -- just needs to be a distinct sentinel object."""


def test_data_uri_always_allowed(monkeypatch):
    fake = _FakeImage()
    monkeypatch.setattr(mlx_vlm_utils, "load_image", lambda src, timeout=10: fake)
    out = server._resolve_images(["data:image/png;base64,abcd"])
    assert out == [fake]


def test_url_rejected_without_opt_in(monkeypatch):
    monkeypatch.setattr(mlx_vlm_utils, "load_image", lambda src, timeout=10: _FakeImage())
    with pytest.raises(ValueError, match="EMBER_ALLOW_IMAGE_URLS"):
        server._resolve_images(["https://example.com/cat.jpg"])


def test_url_allowed_with_opt_in_fetches_via_load_image(monkeypatch):
    fake = _FakeImage()
    seen = []

    def fake_load_image(src, timeout=10):
        seen.append((src, timeout))
        return fake

    monkeypatch.setattr(server, "ALLOW_IMAGE_URLS", True)
    monkeypatch.setattr(mlx_vlm_utils, "load_image", fake_load_image)
    out = server._resolve_images(["https://example.com/cat.jpg"])
    assert out == [fake]
    assert seen == [("https://example.com/cat.jpg", server.IMAGE_FETCH_TIMEOUT_S)]


def test_path_rejected_without_opt_in(monkeypatch):
    monkeypatch.setattr(mlx_vlm_utils, "load_image", lambda src, timeout=10: _FakeImage())
    with pytest.raises(ValueError, match="EMBER_ALLOW_IMAGE_PATHS"):
        server._resolve_images(["/etc/passwd"])


def test_path_allowed_with_opt_in(monkeypatch, tmp_path):
    fake = _FakeImage()
    img_path = tmp_path / "cat.jpg"
    img_path.write_bytes(b"not really a jpeg")
    monkeypatch.setattr(server, "ALLOW_IMAGE_PATHS", True)
    monkeypatch.setattr(mlx_vlm_utils, "load_image", lambda src, timeout=10: fake)
    out = server._resolve_images([str(img_path)])
    assert out == [fake]


def test_load_image_failure_becomes_value_error(monkeypatch):
    monkeypatch.setattr(server, "ALLOW_IMAGE_PATHS", True)

    def boom(src, timeout=10):
        raise OSError("no such file")

    monkeypatch.setattr(mlx_vlm_utils, "load_image", boom)
    with pytest.raises(ValueError, match="failed to load image"):
        server._resolve_images(["/nope.jpg"])


def test_non_string_source_rejected():
    with pytest.raises(ValueError, match="invalid image source"):
        server._resolve_images([{"not": "a string"}])
