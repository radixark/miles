"""The admission tables in core must agree with the trainer's loss registry."""

from miles.backends.training_utils.loss_hub.tinker_losses import TINKER_LOSS_FUNCTIONS
from miles.tinker.core.types import LOSS_FN_INPUTS, LOSS_INPUT_KEYS
from miles.tinker.runtime import DATUM_TO_BATCH_KEYS


def test_every_registered_loss_has_an_admission_entry():
    assert set(LOSS_FN_INPUTS) == set(TINKER_LOSS_FUNCTIONS)


def test_wire_keys_map_onto_trainer_batch_keys():
    assert set(LOSS_INPUT_KEYS.values()) == set(DATUM_TO_BATCH_KEYS)
