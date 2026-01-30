"""
Visualize indexer replay comparison between orig (forward computed) and replay (loaded from dump).

Usage:
    python tests/deepseekv4/adhoc_indexer_replay_visualizer.py --dump-dir /tmp/sglang_dump_megatron_tp2cp2 --rank 0

Remote execution (run from local, execute on remote, pull image back and open):
    ssh sglang-di-b200-sun "docker exec -i miles_tom python3 /host_home/primary_synced/miles-sunrise/tests/deepseekv4/adhoc_indexer_replay_visualizer.py --dump-dir /tmp/sglang_dump_megatron_tp2cp2 --rank 0 --output /host_home/indexer_replay_comparison.png" && scp sglang-di-b200-sun:/data/tom/indexer_replay_comparison.png /tmp/ && open /tmp/indexer_replay_comparison.png
"""

import argparse
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_tensors(dump_dir: str, rank: int):
    """Load orig and replay tensors for a specific rank."""
    orig_pattern = f"{dump_dir}/forward_pass_id=*___rank={rank}___name=orig_top_indices___dump_index=*___layer_id=*.pt"
    replay_pattern = f"{dump_dir}/forward_pass_id=*___rank={rank}___name=replay_top_indices___dump_index=*___layer_id=*.pt"
    
    orig_files = sorted(glob.glob(orig_pattern))
    replay_files = sorted(glob.glob(replay_pattern))
    
    if not orig_files:
        raise FileNotFoundError(f"No orig files found matching {orig_pattern}")
    if not replay_files:
        raise FileNotFoundError(f"No replay files found matching {replay_pattern}")
    
    # Use first file found
    orig = torch.load(orig_files[0], weights_only=False)
    replay = torch.load(replay_files[0], weights_only=False)
    
    print(f"Loaded orig from: {orig_files[0]}")
    print(f"Loaded replay from: {replay_files[0]}")
    
    return orig, replay


def visualize(orig: torch.Tensor, replay: torch.Tensor, output_path: str):
    """Create visualization comparing orig and replay indices."""
    print(f"orig shape: {orig.shape}, replay shape: {replay.shape}")
    print(f"orig range: [{orig.min().item()}, {orig.max().item()}]")
    print(f"replay range: [{replay.min().item()}, {replay.max().item()}]")
    
    # Squeeze batch dim if present
    if orig.dim() == 3:
        orig = orig.squeeze(0)
    if replay.dim() == 3:
        replay = replay.squeeze(0)
    
    orig = orig.cpu().numpy()  # (seq_len, topk)
    replay = replay.cpu().numpy()  # (seq_len, topk)
    
    seq_len, topk = orig.shape
    
    # Compute match ratio per position (diagonal comparison)
    match_count = (orig == replay).sum(axis=1)
    match_ratio = match_count / topk
    
    # Compute pairwise overlap matrix: overlap[x, y] = |orig[x] ∩ replay[y]| / topk
    sample_size = min(256, seq_len)
    sample_indices = np.linspace(0, seq_len - 1, sample_size, dtype=int)
    
    print(f"Computing pairwise overlap matrix ({sample_size}x{sample_size})...")
    overlap_matrix = np.zeros((sample_size, sample_size), dtype=np.float32)
    
    for i, xi in enumerate(sample_indices):
        orig_set_i = set(orig[xi, :])
        for j, yj in enumerate(sample_indices):
            replay_set_j = set(replay[yj, :])
            overlap = len(orig_set_i & replay_set_j)
            overlap_matrix[i, j] = overlap / topk
    
    best_match_idx = np.argmax(overlap_matrix, axis=1)
    best_match_pos = sample_indices[best_match_idx]
    diagonal_overlap = np.diag(overlap_matrix)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original diagnostics
    # Plot 1: Match ratio per sequence position
    ax1 = axes[0, 0]
    ax1.plot(match_ratio)
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Match Ratio')
    ax1.set_title(f'Match Ratio per Position (mean={match_ratio.mean():.3f})')
    ax1.axhline(y=0.7, color='r', linestyle='--', label='threshold=0.7')
    ax1.legend()
    
    # Plot 2: Heatmap of differences
    ax2 = axes[0, 1]
    diff = (orig != replay).astype(float)
    show_len = min(200, seq_len)
    show_topk = min(100, topk)
    im2 = ax2.imshow(diff[:show_len, :show_topk], aspect='auto', cmap='Reds')
    ax2.set_xlabel('Top-k Index')
    ax2.set_ylabel('Sequence Position')
    ax2.set_title(f'Difference Heatmap (first {show_len}x{show_topk})')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Value distribution at a mismatched position
    mismatch_pos = np.where(match_ratio < 0.5)[0]
    pos = mismatch_pos[0] if len(mismatch_pos) > 0 else seq_len // 2
    ax3 = axes[0, 2]
    ax3.hist(orig[pos, :], bins=50, alpha=0.7, label='orig (forward)')
    ax3.hist(replay[pos, :], bins=50, alpha=0.7, label='replay (loaded)')
    ax3.set_xlabel('Index Value')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Value Distribution at Position {pos}')
    ax3.legend()
    
    # Row 2: Shuffle analysis
    # Plot 4: Pairwise overlap heatmap (THE KEY PLOT)
    ax4 = axes[1, 0]
    im4 = ax4.imshow(overlap_matrix, aspect='auto', cmap='viridis', origin='lower',
                     extent=[sample_indices[0], sample_indices[-1], sample_indices[0], sample_indices[-1]])
    ax4.set_xlabel('Replay Position (y)')
    ax4.set_ylabel('Orig Position (x)')
    ax4.set_title(f'TopK Overlap: |orig[x] ∩ replay[y]| / {topk}\n(diagonal=same pos, off-diagonal=shuffle)')
    plt.colorbar(im4, ax=ax4, label='Overlap Ratio')
    ax4.plot([sample_indices[0], sample_indices[-1]], [sample_indices[0], sample_indices[-1]], 
             'r--', linewidth=1, label='x=y (no shuffle)')
    ax4.legend()
    
    # Plot 5: Best match analysis
    ax5 = axes[1, 1]
    ax5.scatter(sample_indices, best_match_pos, alpha=0.5, s=10, label='Best matching replay pos')
    ax5.plot([0, seq_len], [0, seq_len], 'r--', linewidth=1, label='x=y (identity)')
    ax5.set_xlabel('Orig Position')
    ax5.set_ylabel('Best Matching Replay Position')
    ax5.set_title(f'Best Match: argmax_y overlap(orig[x], replay[y])\nDiag mean={diagonal_overlap.mean():.3f}')
    ax5.legend()
    
    # Plot 6: Chunk distribution
    pos_to_show = mismatch_pos[0] if len(mismatch_pos) > 0 else 0
    key_seq_len = max(orig.max(), replay.max()) + 1
    chunk_size = key_seq_len // 4 if key_seq_len >= 4 else key_seq_len
    orig_chunks = orig[pos_to_show, :] // chunk_size
    replay_chunks = replay[pos_to_show, :] // chunk_size
    ax6 = axes[1, 2]
    x = np.arange(4)
    width = 0.35
    orig_chunk_counts = [np.sum(orig_chunks == i) for i in range(4)]
    replay_chunk_counts = [np.sum(replay_chunks == i) for i in range(4)]
    ax6.bar(x - width/2, orig_chunk_counts, width, label='orig', alpha=0.7)
    ax6.bar(x + width/2, replay_chunk_counts, width, label='replay', alpha=0.7)
    ax6.set_xlabel('Chunk')
    ax6.set_ylabel('Count')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['chunk_0', 'chunk_1', 'chunk_2', 'chunk_3'])
    ax6.set_title(f'Chunk Distribution at Position {pos_to_show}')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'Saved to {output_path}')
    
    # Print statistics
    print(f'\n=== Statistics ===')
    print(f'Total positions: {seq_len}')
    print(f'Positions with match_ratio >= 0.7: {(match_ratio >= 0.7).sum()}')
    print(f'Positions with match_ratio < 0.7: {(match_ratio < 0.7).sum()}')
    print(f'Min match ratio: {match_ratio.min():.3f} at position {match_ratio.argmin()}')
    print(f'Max match ratio: {match_ratio.max():.3f} at position {match_ratio.argmax()}')
    
    # Print sample values
    print(f'\n=== Sample Values ===')
    for pos in [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]:
        print(f'Position {pos}: orig[:5]={orig[pos, :5]}, replay[:5]={replay[pos, :5]}, match={match_ratio[pos]:.3f}')


def main():
    parser = argparse.ArgumentParser(description='Visualize indexer replay comparison')
    parser.add_argument('--dump-dir', type=str, default='/tmp/sglang_dump_megatron_tp2cp2',
                        help='Directory containing dump files')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank to visualize')
    parser.add_argument('--output', type=str, default='/tmp/indexer_replay_comparison.png',
                        help='Output image path')
    args = parser.parse_args()
    
    orig, replay = load_tensors(args.dump_dir, args.rank)
    visualize(orig, replay, args.output)


if __name__ == '__main__':
    main()
