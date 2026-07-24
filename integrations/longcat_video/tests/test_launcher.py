from pathlib import Path
import os
import subprocess


SCRIPT = Path(__file__).parents[1] / "run_moviegen10.sh"


def test_launcher_preserves_mode_and_task_arguments(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    torchrun = mock_bin / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    torchrun.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT), "bf16", "prefix"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{mock_bin}:/usr/bin:/bin"},
    )

    arguments = result.stdout.splitlines()
    mode_index = arguments.index("--mode")
    task_index = arguments.index("--task")
    assert arguments[mode_index + 1] == "bf16"
    assert arguments[task_index + 1] == "prefix"
