import typer
from tests.utils.soak.k8s_utils.pod_processes import observe_processes

app = typer.Typer()


@app.command()
def observe(pod_uid: str, pattern: str) -> None:
    print(observe_processes(pod_uid=pod_uid, pattern=pattern).model_dump_json(), flush=True)


if __name__ == "__main__":
    app()
