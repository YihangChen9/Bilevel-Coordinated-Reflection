from pathlib import Path
from env.local import LocalEnv


def test_fs_roundtrip(tmp_path: Path):
    env = LocalEnv(root=tmp_path)
    env.fs_write("a.txt", "hi")
    assert env.fs_read("a.txt") == "hi"


def test_run_shell_exit_code(tmp_path: Path):
    env = LocalEnv(root=tmp_path)
    r = env.run_shell("echo ok")
    assert r.exit_code == 0
    assert "ok" in r.stdout


def test_run_shell_captures_stderr(tmp_path: Path):
    env = LocalEnv(root=tmp_path)
    r = env.run_shell("python -c 'import sys; sys.stderr.write(\"bad\"); sys.exit(2)'")
    assert r.exit_code == 2
    assert "bad" in r.stderr
