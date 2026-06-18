"""CLI tests (parser and helpers — no MLX/server import)."""

import pytest

from ember import cli


def test_build_parser_has_all_commands():
    p = cli.build_parser()
    # extract the names of the registered subcommands
    sub = next(a for a in p._actions if getattr(a, "choices", None) and "serve" in a.choices)
    for name in [
        "serve",
        "ps",
        "status",
        "memory",
        "list",
        "run",
        "warm",
        "unload",
        "clear",
        "config",
        "version",
    ]:
        assert name in sub.choices


def test_run_parses_model_and_optional_prompt():
    args = cli.build_parser().parse_args(["run", "qwen3-8b", "oi"])
    assert args.model == "qwen3-8b" and args.prompt == "oi" and args.func is cli.cmd_run
    args = cli.build_parser().parse_args(["run", "qwen3-8b"])
    assert args.prompt is None


def test_unload_defaults_to_chat():
    assert cli.build_parser().parse_args(["unload"]).target == "chat"
    assert cli.build_parser().parse_args(["unload", "all"]).target == "all"


def test_clear_defaults_to_all_and_validates_choices():
    assert cli.build_parser().parse_args(["clear"]).target == "all"
    assert cli.build_parser().parse_args(["clear", "context"]).target == "context"
    with pytest.raises(SystemExit):  # invalid choice
        cli.build_parser().parse_args(["clear", "model-x"])


def test_serve_flags():
    args = cli.build_parser().parse_args(["serve", "--port", "8010", "--config", "x.yaml"])
    assert args.port == 8010 and args.config == "x.yaml" and args.func is cli.cmd_serve


def test_client_commands_have_url_default():
    args = cli.build_parser().parse_args(["status"])
    assert args.url.startswith("http://")


def test_dur_formatting():
    assert cli._dur(-1) == "∞"
    assert cli._dur(45) == "45s"
    assert cli._dur(120) == "2m"
    assert cli._dur(7200) == "2h"


def test_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(["--help"])
    assert e.value.code == 0


def test_no_command_prints_help(capsys):
    cli.main([])
    out = capsys.readouterr().out
    assert "ember" in out and "<command>" in out
