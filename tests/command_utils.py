import subprocess


def exec_command(cmd: str):
    print(f"EXEC: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)
