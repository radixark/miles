"""Unit tests for the disaggregated-OPD-teacher local unembedding.

`TeacherUnembedding` reconstructs full-vocab teacher log-probs from received hidden
states (already post-final-norm, see the module docstring) using a local linear
projection; `load_teacher_output_layer` reads just that one tensor out of a Megatron
torch_dist checkpoint. Both are pure torch/`torch.distributed.checkpoint` (no
`megatron.core` dependency), so unlike the rest of the Megatron backend, this module
can actually run in a plain CPU sandbox.
"""

import torch
import torch.distributed.checkpoint as dist_cp

from miles.backends.training_utils.opd_teacher_unembedding import TeacherUnembedding, load_teacher_output_layer


def test_teacher_unembedding_matches_reference_log_softmax():
    torch.manual_seed(0)
    hidden_size, vocab_size, num_tokens = 8, 17, 5
    hidden_states = torch.randn(num_tokens, hidden_size)
    output_layer_weight = torch.randn(vocab_size, hidden_size)

    unembedding = TeacherUnembedding(output_layer_weight)
    log_probs = unembedding(hidden_states)

    expected_logits = hidden_states @ output_layer_weight.T
    expected_log_probs = torch.log_softmax(expected_logits, dim=-1)

    assert log_probs.shape == (num_tokens, vocab_size)
    assert torch.allclose(log_probs, expected_log_probs, atol=1e-5)
    # log-probs of a valid distribution: exp(.) sums to 1 across the vocab dim.
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(num_tokens), atol=1e-4)


def test_teacher_unembedding_upcasts_bf16_hidden_states():
    hidden_size, vocab_size = 4, 6
    hidden_states = torch.randn(3, hidden_size, dtype=torch.bfloat16)
    unembedding = TeacherUnembedding(torch.randn(vocab_size, hidden_size))

    log_probs = unembedding(hidden_states)

    assert log_probs.dtype == torch.float32


def test_load_teacher_output_layer_reads_only_the_output_layer_tensor(tmp_path):
    hidden_size, vocab_size = 4, 6
    output_layer_weight = torch.randn(vocab_size, hidden_size)
    unrelated_tensor = torch.randn(3, 3)

    checkpoint_dir = tmp_path / "iter_0000042"
    checkpoint_dir.mkdir()
    dist_cp.save(
        {
            "output_layer.weight": output_layer_weight,
            "some_other_layer.weight": unrelated_tensor,
        },
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("42")

    loaded_output_layer = load_teacher_output_layer(
        str(tmp_path), None, hidden_size=hidden_size, vocab_size=vocab_size, dtype=torch.float32
    )

    assert torch.equal(loaded_output_layer, output_layer_weight)


def test_load_teacher_output_layer_falls_back_to_tied_embedding(tmp_path):
    hidden_size, vocab_size = 4, 6
    tied_embedding_weight = torch.randn(vocab_size, hidden_size)

    checkpoint_dir = tmp_path / "iter_0000001"
    checkpoint_dir.mkdir()
    dist_cp.save(
        {"embedding.word_embeddings.weight": tied_embedding_weight},
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1")

    loaded_output_layer = load_teacher_output_layer(
        str(tmp_path), None, hidden_size=hidden_size, vocab_size=vocab_size, dtype=torch.float32
    )

    assert torch.equal(loaded_output_layer, tied_embedding_weight)


def test_load_teacher_output_layer_reads_a_release_checkpoint(tmp_path):
    hidden_size, vocab_size = 4, 6
    output_layer_weight = torch.randn(vocab_size, hidden_size)

    checkpoint_dir = tmp_path / "release"
    checkpoint_dir.mkdir()
    dist_cp.save(
        {"output_layer.weight": output_layer_weight},
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")

    loaded_output_layer = load_teacher_output_layer(
        str(tmp_path), None, hidden_size=hidden_size, vocab_size=vocab_size, dtype=torch.float32
    )

    assert torch.equal(loaded_output_layer, output_layer_weight)


def test_load_teacher_output_layer_respects_explicit_ckpt_step(tmp_path):
    hidden_size, vocab_size = 2, 3
    output_layer_weight = torch.randn(vocab_size, hidden_size)

    checkpoint_dir = tmp_path / "iter_0000007"
    checkpoint_dir.mkdir()
    dist_cp.save(
        {"output_layer.weight": output_layer_weight},
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )
    # No latest_checkpointed_iteration.txt on purpose: explicit ckpt_step must bypass it.

    loaded_output_layer = load_teacher_output_layer(
        str(tmp_path), 7, hidden_size=hidden_size, vocab_size=vocab_size, dtype=torch.float32
    )

    assert torch.equal(loaded_output_layer, output_layer_weight)
