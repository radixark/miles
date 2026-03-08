import typer

from miles.utils.ft.cli.diag.cluster import cluster
from miles.utils.ft.cli.diag.local import local

app = typer.Typer(help="Node diagnostic commands.")
app.command()(local)
app.command()(cluster)
