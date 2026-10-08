from collections import defaultdict
from collections.abc import Iterable

import torch

from miles.backends.training_utils.weight_update.protocols.utils.loader_probe import HfNameMapping


class ModelParamStager:
    """Hold each parameter group's HF tensors until the group is complete.

    Parameters sharing HF inputs form one group; loading only part would leave parameters unbound or incomplete.
    """

    def __init__(self, hf_name_mapping: HfNameMapping) -> None:
        self._param_group_by_hf_name, self._hf_names_by_param_group = _build_param_groups_by_shared_hf_inputs(hf_name_mapping)
        self._staged_hf_tensors_by_param_group: dict[tuple[str, ...], list[tuple[str, torch.Tensor]]] = {}
        self._missing_hf_names_by_param_group: dict[tuple[str, ...], set[str]] = {}

    @property
    def param_groups(self) -> list[tuple[str, ...]]:
        return list(self._hf_names_by_param_group)

    def stage(
        self, hf_tensors: Iterable[tuple[str, torch.Tensor]]
    ) -> dict[tuple[str, ...], list[tuple[str, torch.Tensor]]]:
        """Return newly completed parameter groups and their HF tensors."""
        ready_hf_tensors_by_param_group = {}
        for hf_name, tensor in hf_tensors:
            param_group = self._param_group_by_hf_name[hf_name]
            missing_hf_names = self._missing_hf_names_by_param_group.setdefault(
                param_group, set(self._hf_names_by_param_group[param_group])
            )
            missing_hf_names.discard(hf_name)
            self._staged_hf_tensors_by_param_group.setdefault(param_group, []).append((hf_name, tensor))
            if not missing_hf_names:
                del self._missing_hf_names_by_param_group[param_group]
                ready_hf_tensors_by_param_group[param_group] = self._staged_hf_tensors_by_param_group.pop(param_group)
        return ready_hf_tensors_by_param_group

    def assert_all_done(self) -> None:
        assert not self._missing_hf_names_by_param_group, (
            "p2p update ended with params still missing HF tensors, which would leave their bytes unwritten: "
            + ", ".join(
                f"{param_group} lacks {sorted(missing)[:3]}"
                for param_group, missing in list(self._missing_hf_names_by_param_group.items())[:5]
            )
        )


def _build_param_groups_by_shared_hf_inputs(
    hf_name_mapping: HfNameMapping,
) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, ...], frozenset[str]]]:
    # union-find over params: two params sharing an HF name load together
    root_by_param_name = {param_name: param_name for param_name in hf_name_mapping.hf_names_by_param_name}

    def find_root(param_name: str) -> str:
        while root_by_param_name[param_name] != param_name:
            root_by_param_name[param_name] = root_by_param_name[root_by_param_name[param_name]]
            param_name = root_by_param_name[param_name]
        return param_name

    for param_names in hf_name_mapping.param_names_by_hf_name.values():
        first_param_name, *other_param_names = sorted(param_names)
        for param_name in other_param_names:
            root_by_param_name[find_root(param_name)] = find_root(first_param_name)

    param_names_by_root = defaultdict(list)
    for param_name in root_by_param_name:
        param_names_by_root[find_root(param_name)].append(param_name)
    param_group_by_hf_name, hf_names_by_param_group = {}, {}
    for param_names in param_names_by_root.values():
        param_group = tuple(sorted(param_names))
        hf_names = frozenset().union(*(hf_name_mapping.hf_names_by_param_name[name] for name in param_group))
        hf_names_by_param_group[param_group] = hf_names
        for hf_name in hf_names:
            param_group_by_hf_name[hf_name] = param_group
    return param_group_by_hf_name, hf_names_by_param_group
