"""num_ctx enforcement tests (issue #22): a documented config option that must
actually reject oversized prompts instead of silently doing nothing."""

from ember.server import _num_ctx_error


def test_no_limit_configured_never_errors():
    assert _num_ctx_error(None, 1_000_000) is None
    assert _num_ctx_error(0, 1_000_000) is None


def test_within_budget_is_fine():
    assert _num_ctx_error(8192, 8192) is None
    assert _num_ctx_error(8192, 100) is None


def test_over_budget_errors_with_useful_message():
    err = _num_ctx_error(8192, 8193)
    assert err is not None
    assert "8193" in err
    assert "8192" in err
