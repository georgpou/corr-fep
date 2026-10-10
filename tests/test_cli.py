import subprocess
import sys


def test_demo_cli():
    output = subprocess.check_output(
        [sys.executable, "-m", "corr_fep.cli", "demo", "--seed", "11", "--n", "20"],
        text=True,
    )
    assert '"mode": "synthetic_demo"' in output


def test_doctor_cli():
    output = subprocess.check_output(
        [sys.executable, "-m", "corr_fep.cli", "doctor"], text=True
    )
    assert '"python_architecture"' in output
