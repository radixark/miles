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
    
    # Compute match ratio per position
    match_count = (orig == replay).sum(axis=1)
    match_ratio = match_count / topk
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
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
    im = ax2.imshow(diff[:show_len, :show_topk], aspect='auto', cmap='Reds')
    ax2.set_xlabel('Top-k Index')
    ax2.set_ylabel('Sequence Position')
    ax2.set_title(f'Difference Heatmap (first {show_len}x{show_topk})')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Value distribution at a mismatched position
    mismatch_pos = np.where(match_ratio < 0.5)[0]
    if len(mismatch_pos) > 0:
        pos = mismatch_pos[0]
    else:
        pos = seq_len // 2
    ax3 = axes[0, 2]
    ax3.hist(orig[pos, :], bins=50, alpha=0.7, label='orig (forward)')
    ax3.hist(replay[pos, :], bins=50, alpha=0.7, label='replay (loaded)')
    ax3.set_xlabel('Index Value')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Value Distribution at Position {pos}')
    ax3.legend()
    
    # Plot 4: Scatter plot - orig vs replay indices (overlap visualization)
    ax4 = axes[1, 0]
    # For each position, plot orig indices vs replay indices
    sample_positions = np.linspace(0, seq_len - 1, min(10, seq_len), dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_positions)))
    for i, pos in enumerate(sample_positions):
        ax4.scatter(orig[pos, :20], replay[pos, :20], alpha=0.5, c=[colors[i]], label=f'pos {pos}', s=20)
    max_val = max(orig.max(), replay.max())
    ax4.plot([0, max_val], [0, max_val], 'r--', label='y=x (perfect match)')
    ax4.set_xlabel('Orig Index (forward computed)')
    ax4.set_ylabel('Replay Index (loaded from dump)')
    ax4.set_title('Orig vs Replay Indices (first 20 topk, sampled positions)')
    ax4.legend(fontsize=6)
    
    # Plot 5: Sorted indices comparison - shows ordering
    ax5 = axes[1, 1]
    pos_to_show = mismatch_pos[0] if len(mismatch_pos) > 0 else 0
    orig_sorted = np.sort(orig[pos_to_show, :])
    replay_sorted = np.sort(replay[pos_to_show, :])
    ax5.scatter(orig_sorted, replay_sorted, alpha=0.5, s=10)
    ax5.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax5.set_xlabel('Orig Index (sorted)')
    ax5.set_ylabel('Replay Index (sorted)')
    ax5.set_title(f'Sorted Indices at Position {pos_to_show}')
    ax5.legend()
    
    # Plot 6: Index value transformation visualization
    ax6 = axes[1, 2]
    # Show how indices are distributed across chunks
    key_seq_len = max(orig.max(), replay.max()) + 1
    chunk_size = key_seq_len // 4 if key_seq_len >= 4 else key_seq_len
    
    orig_chunks = orig[pos_to_show, :] // chunk_size
    replay_chunks = replay[pos_to_show, :] // chunk_size
    
    chunk_labels = ['chunk_0', 'chunk_1', 'chunk_2', 'chunk_3']
    x = np.arange(4)
    width = 0.35
    
    orig_chunk_counts = [np.sum(orig_chunks == i) for i in range(4)]
    replay_chunk_counts = [np.sum(replay_chunks == i) for i in range(4)]
    
    ax6.bar(x - width/2, orig_chunk_counts, width, label='orig', alpha=0.7)
    ax6.bar(x + width/2, replay_chunk_counts, width, label='replay', alpha=0.7)
    ax6.set_xlabel('Chunk')
    ax6.set_ylabel('Count')
    ax6.set_xticks(x)
    ax6.set_xticklabels(chunk_labels)
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
