from miles.utils.ft.models.base import FtBaseModel


class DiagnosticPipelineResult(FtBaseModel):
    bad_node_ids: list[str] = []
    reason: str = ""
