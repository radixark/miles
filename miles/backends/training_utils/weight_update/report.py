from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightUpdateReport:
    weight_version: int | None
    updated_cell_ids: tuple[str, ...]
    failed_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, cell_ids in (
            ("updated_cell_ids", self.updated_cell_ids),
            ("failed_cell_ids", self.failed_cell_ids),
        ):
            assert isinstance(cell_ids, tuple), f"{name} must be a tuple, got {type(cell_ids).__name__}"
            assert len(set(cell_ids)) == len(cell_ids), f"{name} names a cell twice, got {list(cell_ids)}"

        both = sorted(set(self.updated_cell_ids) & set(self.failed_cell_ids))
        assert not both, f"cells {both} are reported as both updated and failed"

        if self.weight_version is None:
            assert not self.failed_cell_ids, (
                f"an update that published no version reached no cell to fail, "
                f"got failures {sorted(self.failed_cell_ids)}"
            )

    @property
    def reported_cell_ids(self) -> frozenset[str]:
        return frozenset(self.updated_cell_ids) | frozenset(self.failed_cell_ids)

    def validate_assignment(self, assigned_cell_ids: Sequence[str]) -> None:
        assigned = frozenset(assigned_cell_ids)
        assert len(assigned) == len(assigned_cell_ids), f"a cell is assigned twice, got {list(assigned_cell_ids)}"

        unknown = sorted(self.reported_cell_ids - assigned)
        assert not unknown, f"cells {unknown} were never assigned to this trainer, which owns {sorted(assigned)}"

        missing = sorted(assigned - self.reported_cell_ids)
        assert (
            not missing
        ), f"cells {missing} were assigned to this trainer but it reported neither success nor failure"


def build_weight_update_report(
    *,
    weight_version: int | None,
    assigned_cell_ids: Sequence[str],
    failed_cell_ids: Iterable[str],
) -> WeightUpdateReport:
    failed = frozenset(failed_cell_ids)
    unknown = sorted(failed - frozenset(assigned_cell_ids))
    assert not unknown, f"cells {unknown} failed without being assigned, which owns {sorted(assigned_cell_ids)}"

    report = WeightUpdateReport(
        weight_version=weight_version,
        updated_cell_ids=tuple(cell_id for cell_id in assigned_cell_ids if cell_id not in failed),
        failed_cell_ids=tuple(cell_id for cell_id in assigned_cell_ids if cell_id in failed),
    )
    report.validate_assignment(assigned_cell_ids)
    return report


def merge_rank_reports(reports: Sequence[WeightUpdateReport], *, debug_name: str) -> WeightUpdateReport:
    assert reports, f"{debug_name} answered with no report at all"

    distinct = {(report.weight_version, report.updated_cell_ids, report.failed_cell_ids) for report in reports}
    assert len(distinct) == 1, f"{debug_name} ranks disagree on what this update achieved, got {sorted(distinct)}"

    return reports[0]


def build_untouched_targets_report(assigned_cell_ids: Sequence[str]) -> WeightUpdateReport:
    return build_weight_update_report(weight_version=None, assigned_cell_ids=assigned_cell_ids, failed_cell_ids=())


def combine_trainer_reports(reports: Sequence[WeightUpdateReport]) -> WeightUpdateReport:
    assert reports, "no trainer cell reported the outcome of this update"

    versions = {report.weight_version for report in reports}
    assert len(versions) == 1, f"the trainer cells published different weight versions, got {sorted(versions)}"
    [weight_version] = versions

    return WeightUpdateReport(
        weight_version=weight_version,
        updated_cell_ids=tuple(cell_id for report in reports for cell_id in report.updated_cell_ids),
        failed_cell_ids=tuple(cell_id for report in reports for cell_id in report.failed_cell_ids),
    )
