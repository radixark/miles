"""
Performance visualization for OpenEvolve evolution runs
"""

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
# Import error filtering functionality
try:
    from openevolve.modular_utils.error_constants import ErrorThresholds, get_visualization_safe_score
except ImportError:
    raise ImportError("Error filtering module not found - ensure openevolve.modular_utils.error_constants is available")

logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """
    Creates performance visualizations for OpenEvolve experiments
    
    Supports both individual experiment analysis and multi-experiment comparisons
    with log-scale and linear-scale plotting options.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir) if output_dir else Path("./visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Check matplotlib availability
        self.matplotlib_available = self._check_matplotlib()
        if not self.matplotlib_available:
            logger.warning("⚠️ matplotlib not available - visualizations will be disabled")
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            return True
        except ImportError:
            return False
    
    def _extract_iteration_from_dirname(self, dirname: str) -> Optional[int]:
        """Extract iteration number from directory name like 'iter_123_gen_04_abc123_175332' or 'step01_gen00_id...'"""
        # Support both iter_XXX and stepXX formats
        match = re.match(r'iter_(\d+)_', dirname)
        if match:
            return int(match.group(1))

        # Support stepXX format from gym recorder
        match = re.match(r'step(\d+)_', dirname)
        if match:
            return int(match.group(1))

        return None
    
    def _filter_error_scores(self, scores: List[float], fallback_strategy: str = 'min_valid') -> Tuple[List[float], dict]:
        """
        Filter out error codes from score lists for visualization with statistics
        
        Args:
            scores: List of scores that may contain error codes
            fallback_strategy: Strategy for replacing error codes ('min_valid', 'zero', 'none')
        
        Returns:
            Tuple of (filtered_scores, statistics)
        """
        if not scores:
            return scores, {'valid_count': 0, 'error_count': 0, 'error_rate': 0.0}
        
        # Separate valid scores from error codes
        valid_scores = []
        error_count = 0
        
        for score in scores:
            if isinstance(score, (int, float)) and not ErrorThresholds.is_error_code(int(score)):
                valid_scores.append(score)
            else:
                error_count += 1
        
        # Determine fallback value
        if fallback_strategy == 'min_valid' and valid_scores:
            fallback = min(valid_scores)
        elif fallback_strategy == 'zero':
            fallback = 0.0
        elif fallback_strategy == 'none':
            fallback = None
        else:
            fallback = 0.0
        
        # Apply filtering
        filtered_scores = []
        for score in scores:
            if isinstance(score, (int, float)) and not ErrorThresholds.is_error_code(int(score)):
                filtered_scores.append(score)
            else:
                if fallback is not None:
                    filtered_scores.append(fallback)
                else:
                    filtered_scores.append(None)  # Keep as None for proper filtering later
        
        statistics = {
            'valid_count': len(valid_scores),
            'error_count': error_count,
            'total_count': len(scores),
            'error_rate': error_count / len(scores) if scores else 0.0
        }
        
        return filtered_scores, statistics
    
    def _load_historical_records(self, historical_dir: Path) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Load all historical records and return sorted list of (iteration, metadata)
        Handles multiple generations per iteration by selecting the one with lowest generation
        """
        records = []
        iteration_counts = {}
        
        if not historical_dir.exists():
            logger.debug(f"Historical records directory not found: {historical_dir}")
            return records
        
        # Get all iteration directories (support both iter_ and step formats)
        iter_dirs = [d for d in historical_dir.iterdir() if d.is_dir() and (d.name.startswith('iter_') or d.name.startswith('step'))]
        
        # First pass: check for duplicates
        for iter_dir in iter_dirs:
            iteration = self._extract_iteration_from_dirname(iter_dir.name)
            if iteration is not None:
                iteration_counts[iteration] = iteration_counts.get(iteration, 0) + 1
        
        # Report duplicates
        duplicates = {k: v for k, v in iteration_counts.items() if v > 1}
        if duplicates:
            logger.debug(f"Found duplicate iterations: {duplicates}")
            for iter_num, count in duplicates.items():
                logger.debug(f"  Iteration {iter_num} appears {count} times")
        
        # Second pass: load records (prefer lowest generation for duplicates)
        seen_iterations = {}  # iteration -> (generation, dir_name)
        for iter_dir in iter_dirs:
            iteration = self._extract_iteration_from_dirname(iter_dir.name)
            if iteration is None:
                continue
                
            # Extract generation from directory name (e.g., iter_167_gen_05_abc_123456 -> 5)
            gen_match = re.search(r'_gen_(\d+)_', iter_dir.name)
            generation = int(gen_match.group(1)) if gen_match else 0
            
            # Keep the record with the lowest generation for each iteration
            if iteration not in seen_iterations or generation < seen_iterations[iteration][0]:
                if iteration in seen_iterations:
                    logger.debug(f"Replacing iteration {iteration} gen_{seen_iterations[iteration][0]} with gen_{generation}")
                seen_iterations[iteration] = (generation, iter_dir.name)
        
        # Third pass: load the selected records
        for iter_dir in iter_dirs:
            iteration = self._extract_iteration_from_dirname(iter_dir.name)
            if iteration is None:
                continue
                
            # Only process if this is the selected directory for this iteration
            if seen_iterations.get(iteration, (None, None))[1] != iter_dir.name:
                continue
                
            metadata_file = iter_dir / 'metadata.json'
            if not metadata_file.exists():
                logger.debug(f"No metadata.json in {iter_dir}")
                continue
                
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                records.append((iteration, metadata))
            except Exception as e:
                logger.debug(f"Failed to load {metadata_file}: {e}")
                continue
        
        # Sort by iteration number
        records.sort(key=lambda x: x[0])
        logger.info(f"Loaded {len(records)} unique historical records as fallback")
        
        return records
    
    def _enhance_tracker_with_historical_records(self, tracker, output_dir: Optional[str] = None):
        """
        Enhance tracker with historical records if performance data seems incomplete
        """
        from openevolve.analysis.performance_tracker import IterationRecord
        
        if not output_dir:
            return tracker
            
        historical_dir = Path(output_dir) / "historical_records"
        
        # Check if we should use historical records as fallback
        use_historical = False
        
        # Case 1: No performance records at all
        if not tracker.iteration_records:
            use_historical = True
            logger.info("No performance records found - using historical_records")
        
        # Case 2: Historical records show more iterations than performance records
        elif historical_dir.exists():
            historical_records = self._load_historical_records(historical_dir)
            if historical_records:
                max_historical_iter = max(r[0] for r in historical_records)
                max_performance_iter = max(r.iteration for r in tracker.iteration_records) if tracker.iteration_records else -1
                
                if max_historical_iter > max_performance_iter + 10:  # Significant gap
                    use_historical = True
                    logger.warning(f"Performance records incomplete: max iter {max_performance_iter} vs historical {max_historical_iter}")
                    logger.info("Using historical_records to fill gaps")
        
        if not use_historical:
            return tracker
        
        # Load historical records and reconstruct performance data
        historical_records = self._load_historical_records(historical_dir)
        if not historical_records:
            logger.warning("No historical records found for fallback")
            return tracker
        
        # Clear existing records and rebuild from historical data
        tracker.iteration_records = []
        tracker.best_score = 0.0
        tracker.best_program_id = ""
        tracker.best_iteration = 0
        
        for iteration, metadata in historical_records:
            program_id = metadata.get('program_id', 'unknown')
            timestamp = metadata.get('timestamp', 0.0)
            metrics = metadata.get('metrics', {})
            generation = metadata.get('generation', 0)
            iteration_found = metadata.get('iteration_found', iteration)
            parent_id = metadata.get('parent_id')
            
            # Check if this is a new best
            is_new_best = False
            if 'combined_score' in metrics:
                score = metrics['combined_score']
                if isinstance(score, (int, float)) and score > tracker.best_score:
                    is_new_best = True
                    tracker.best_score = score
                    tracker.best_program_id = program_id
                    tracker.best_iteration = iteration
            
            # Create record
            record = IterationRecord(
                iteration=iteration,
                timestamp=timestamp,
                program_id=program_id,
                metrics=metrics,
                generation=generation,
                iteration_found=iteration_found,
                parent_id=parent_id,
                prompt_tokens=0,  # Not available in historical records
                response_tokens=0,  # Not available in historical records
                total_tokens=0,  # Not available in historical records
                evaluation_time=metrics.get('eval_time', 0.0),
                llm_time=0.0,  # Not available in historical records
                improvement={},  # Could compute but not implemented
                is_new_best=is_new_best
            )
            
            tracker.iteration_records.append(record)
        
        logger.info(f"Enhanced tracker with {len(tracker.iteration_records)} records from historical data")
        logger.info(f"Best score: {tracker.best_score:.4f} at iteration {tracker.best_iteration}")
        
        return tracker
    
    def plot_evolution_progress(
        self, 
        tracker, 
        metric_name: str = "combined_score",
        scale: str = "linear",
        save_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Plot evolution progress over iterations with error filtering
        
        Args:
            tracker: PerformanceTracker instance
            metric_name: Name of metric to plot
            scale: "linear" or "log" scale
            save_path: Optional path to save plot
            output_dir: Output directory for fallback to historical records
            
        Returns:
            Path to saved plot
        """
        if not self.matplotlib_available:
            logger.error("❌ Cannot create plot - matplotlib not available")
            return ""
        
        import matplotlib.pyplot as plt
        
        # Enhance tracker with historical records if needed
        tracker = self._enhance_tracker_with_historical_records(tracker, output_dir)
        
        if not tracker.iteration_records:
            logger.warning("⚠️ No data to plot")
            return ""
        
        # Extract data - ensure records are sorted by iteration
        sorted_records = sorted(tracker.iteration_records, key=lambda r: r.iteration)
        
        iterations = []
        raw_scores = []
        
        for record in sorted_records:
            iterations.append(record.iteration)
            
            if metric_name in record.metrics:
                score = record.metrics[metric_name]
                if isinstance(score, (int, float)) and not (score != score):  # Check for NaN
                    raw_scores.append(score)
                else:
                    raw_scores.append(None)
            else:
                raw_scores.append(None)
        
        # Filter error codes using our modular system
        filtered_scores, error_stats = self._filter_error_scores(raw_scores, 'min_valid')
        
        # Compute best score progression with filtered data
        best_scores = []
        current_best = 0.0

        # Check if we have explicit best_score_so_far data (from gym recorder)
        explicit_best_scores = []
        has_explicit_best_scores = False
        for record in sorted_records:
            if "best_score_so_far" in record.metrics:
                explicit_best_scores.append(record.metrics["best_score_so_far"])
                has_explicit_best_scores = True
            else:
                explicit_best_scores.append(None)

        # Use explicit best scores if available (gym mode), otherwise compute from current scores (OpenEvolve mode)
        if has_explicit_best_scores and any(score is not None for score in explicit_best_scores):
            best_scores = explicit_best_scores
        else:
            # Original OpenEvolve logic - compute cumulative best from current scores
            for score in filtered_scores:
                if score is not None and score > current_best:
                    current_best = score
                best_scores.append(current_best)
        
        # Create single plot (iteration only)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create title with error statistics
        title = f'Evolution Progress - {tracker.experiment_id}'
        if isinstance(error_stats, dict) and error_stats.get('error_count', 0) > 0:
            error_pct = error_stats['error_rate'] * 100
            title += f' ({error_stats["valid_count"]} valid, {error_stats["error_count"]} errors, {error_pct:.1f}%)'
        
        fig.suptitle(title, fontsize=16)
        
        # Filter None values for plotting
        plot_iterations = []
        plot_scores = []
        for it, score in zip(iterations, filtered_scores):
            if score is not None:
                plot_iterations.append(it)
                plot_scores.append(score)
        
        if plot_scores:
            # Plot individual scores and best progression
            ax.plot(plot_iterations, plot_scores, 'b-', alpha=0.6, label=f'{metric_name} (individual)', marker='o', markersize=3)
            ax.plot(iterations, best_scores, 'r-', linewidth=2, label=f'{metric_name} (best so far)')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(f'{metric_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Handle scale
            if scale == "log":
                min_score = min(plot_scores + best_scores)
                if min_score <= 0:
                    # Use symlog scale for negative values
                    data_range = max(plot_scores + best_scores) - min_score
                    linthresh = max(0.1, data_range * 0.01)
                    ax.set_yscale('symlog', linthresh=linthresh)
                    logger.info(f"Using symlog scale for negative scores (min: {min_score:.3f}, linthresh: {linthresh:.3f})")
                else:
                    ax.set_yscale('log')
            
            # Add best score annotation
            best_score_value = max(best_scores)
            best_idx = best_scores.index(best_score_value)
            ax.annotate(f'Best: {best_score_value:.4f}', 
                       xy=(iterations[best_idx], best_score_value),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = str(self.output_dir / f"{tracker.experiment_id}_progress_{scale}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved evolution progress plot: {save_path}")
        return save_path
    
    def plot_token_usage(self, tracker, save_path: Optional[str] = None) -> str:
        """
        Plot token usage over time
        
        Args:
            tracker: PerformanceTracker instance
            save_path: Optional path to save plot
            
        Returns:
            Path to saved plot
        """
        if not self.matplotlib_available:
            logger.error("❌ Cannot create plot - matplotlib not available")
            return ""
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not tracker.iteration_records:
            logger.warning("⚠️ No data to plot")
            return ""
        
        # Extract data
        iterations = [r.iteration for r in tracker.iteration_records]
        prompt_tokens = [r.prompt_tokens for r in tracker.iteration_records]
        response_tokens = [r.response_tokens for r in tracker.iteration_records]
        total_tokens = [r.total_tokens for r in tracker.iteration_records]
        
        # Calculate cumulative tokens
        cumulative_tokens = np.cumsum(total_tokens)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Token Usage - {tracker.experiment_id}', fontsize=16)
        
        # Plot 1: Per-iteration token usage
        ax1.bar(iterations, prompt_tokens, label='Prompt tokens', alpha=0.7)
        ax1.bar(iterations, response_tokens, bottom=prompt_tokens, label='Response tokens', alpha=0.7)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Tokens per iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative token usage
        ax2.plot(iterations, cumulative_tokens, 'g-', linewidth=2, label='Cumulative tokens')
        ax2.fill_between(iterations, cumulative_tokens, alpha=0.3)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative tokens')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add final total annotation
        final_total = cumulative_tokens[-1]
        ax2.annotate(f'Total: {final_total:,} tokens', 
                    xy=(iterations[-1], final_total),
                    xytext=(-50, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = str(self.output_dir / f"{tracker.experiment_id}_tokens.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved token usage plot: {save_path}")
        return save_path
    
    def plot_performance_vs_tokens(self, tracker, save_path: Optional[str] = None) -> str:
        """
        Plot performance metrics vs cumulative token usage
        Shows how performance improves as more compute (tokens) is used
        
        Args:
            tracker: PerformanceTracker instance
            save_path: Optional path to save plot
            
        Returns:
            Path to saved plot
        """
        if not self.matplotlib_available:
            logger.error("❌ Cannot create plot - matplotlib not available")
            return ""
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not tracker.iteration_records:
            logger.warning("⚠️ No data to plot")
            return ""
        
        # Extract data
        records = sorted(tracker.iteration_records, key=lambda r: r.iteration)
        total_tokens = [r.total_tokens for r in records]
        cumulative_tokens = np.cumsum(total_tokens)
        
        # Extract performance metrics (look for common performance indicators)
        performance_metrics = {}
        for record in records:
            for key, value in record.metrics.items():
                if isinstance(value, (int, float)) and key not in ['eval_time', 'complexity', 'diversity']:
                    if key not in performance_metrics:
                        performance_metrics[key] = []
                    performance_metrics[key].append(value)
        
        if not performance_metrics:
            logger.warning("⚠️ No numerical performance metrics found")
            return ""
        
        # Create subplot for each metric
        num_metrics = len(performance_metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'Performance vs Token Usage - {tracker.experiment_id}', fontsize=16)
        
        for idx, (metric_name, values) in enumerate(performance_metrics.items()):
            ax = axes[idx]
            
            # Skip if we don't have enough data
            if len(values) != len(cumulative_tokens):
                continue
                
            # Plot performance vs tokens
            ax.scatter(cumulative_tokens, values, alpha=0.6, s=50)
            ax.plot(cumulative_tokens, values, '-', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('Cumulative Token Usage')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            # Add trend line if there are enough points
            if len(values) > 5:
                try:
                    # Fit a trend line
                    z = np.polyfit(cumulative_tokens, values, 1)
                    p = np.poly1d(z)
                    ax.plot(cumulative_tokens, p(cumulative_tokens), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.2e})')
                    ax.legend()
                except:
                    pass
            
            # Highlight best performance
            best_idx = np.argmax(values) if 'sum' in metric_name.lower() or 'score' in metric_name.lower() else np.argmax(values)
            ax.scatter(cumulative_tokens[best_idx], values[best_idx], color='red', s=100, marker='*', label=f'Best: {values[best_idx]:.4f}')
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = str(self.output_dir / f"{tracker.experiment_id}_performance_vs_tokens.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved performance vs tokens plot: {save_path}")
        return save_path

    
    def create_summary_report(self, tracker, save_path: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """
        Create a comprehensive summary report with success rates
        
        Args:
            tracker: PerformanceTracker instance
            save_path: Optional path to save report
            output_dir: Output directory for computing success rates
            
        Returns:
            Path to saved report
        """
        if save_path is None:
            save_path = str(self.output_dir / f"{tracker.experiment_id}_report.html")
        
        summary = tracker.get_summary()
        
        # Compute success rates if output directory is provided
        success_rates = None
        if output_dir:
            try:
                from openevolve.analysis.success_rates import compute_success_rates
                success_rates = compute_success_rates(Path(output_dir))
            except Exception as e:
                logger.warning(f"Failed to compute success rates: {e}")
                success_rates = {
                    "extraction_rate": 0.0,
                    "validity_rate": 0.0,
                    "has_data": False
                }
        
        # Create HTML report
        success_rates_section = ""
        if success_rates and success_rates.get("has_data", False):
            extraction_rate = success_rates["extraction_rate"]
            validity_rate = success_rates["validity_rate"]
            success_rates_section = f"""
    <div class="section">
        <h3>🔧 Code Generation Quality</h3>
        <div class="metric {'success' if extraction_rate > 0.8 else 'warning' if extraction_rate > 0.5 else ''}">
            Extraction Success Rate: {extraction_rate:.1%} ({success_rates.get('extraction_count', 0)}/{success_rates.get('total_expected_iterations', 0)})
        </div>
        <div class="metric {'success' if validity_rate > 0.8 else 'warning' if validity_rate > 0.5 else ''}">
            Validity Success Rate: {validity_rate:.1%} ({success_rates.get('validity_count', 0)}/{success_rates.get('total_expected_iterations', 0)})
        </div>
    </div>
"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpenEvolve Performance Report - {tracker.experiment_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .metric.success {{ background-color: #e8f5e8; border-left: 4px solid #4CAF50; }}
        .metric.warning {{ background-color: #fff3cd; border-left: 4px solid #ff9800; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OpenEvolve Performance Report</h1>
        <h2>Experiment: {tracker.experiment_id}</h2>
    </div>
    
    <div class="section">
        <h3>📈 Core Performance Metrics</h3>
        <div class="metric">Total Iterations: {summary.get('total_iterations', 0)}</div>
        <div class="metric">Best Score: {summary.get('best_score', 0):.4f}</div>
        <div class="metric">Best Iteration: {summary.get('best_iteration', 0)}</div>
        <div class="metric">Total Tokens: {summary.get('total_tokens', 0):,}</div>
        <div class="metric">Total Time: {summary.get('total_time', 0):.2f} seconds</div>
        <div class="metric">Average Tokens per Iteration: {summary.get('avg_tokens_per_iter', 0):.1f}</div>
        <div class="metric">Average Time per Iteration: {summary.get('avg_time_per_iter', 0):.2f} seconds</div>
    </div>
    
    {success_rates_section}
    
    <div class="section">
        <h3>📋 Recent Performance History</h3>
        <table>
            <tr><th>Iteration</th><th>Program ID</th><th>Score</th><th>Tokens</th><th>Time (s)</th></tr>
"""
        
        # Add recent iterations to table
        recent_records = tracker.iteration_records[-10:]  # Last 10 iterations
        for record in recent_records:
            score = record.metrics.get('combined_score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            
            html_content += f"""
            <tr>
                <td>{record.iteration}</td>
                <td>{record.program_id[:8]}...</td>
                <td>{score_str}</td>
                <td>{record.total_tokens}</td>
                <td>{record.evaluation_time + record.llm_time:.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📋 Saved summary report: {save_path}")
        return save_path