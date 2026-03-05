from __future__ import annotations

from argparse import Namespace

import cli.describe_image_cli as cli_mod


def test_main_describe_command_falls_back_to_help(monkeypatch):
    printed = {"count": 0}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="describe", image="img.png", query="rewrite"),
    )
    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "print_help",
        lambda _self: printed.__setitem__("count", printed["count"] + 1),
    )

    rc = cli_mod.main()

    assert rc is None
    assert printed["count"] == 1


def test_main_unknown_command_prints_help(monkeypatch):
    printed = {"count": 0}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command=None),
    )
    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "print_help",
        lambda _self: printed.__setitem__("count", printed["count"] + 1),
    )

    rc = cli_mod.main()

    assert rc is None
    assert printed["count"] == 1
