import typer

from miles.utils.ft.cli.diag import app as diag_app
from miles.utils.ft.cli.launch import launch

app = typer.Typer(help="Miles Fault Tolerance CLI.")
app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)(launch)
app.add_typer(diag_app, name="diag")
