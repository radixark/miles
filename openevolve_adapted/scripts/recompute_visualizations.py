#!/usr/bin/env python3
"""
Recompute visualizations from historical_records

This script reconstructs performance data from historical_records when the 
performance tracker data is incomplete, then generates updated visualizations.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import glob
import re

# Add openevolve to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openevolve.analysis.performance_tracker import PerformanceTracker, IterationRecord
from openevolve.analysis.visualizer import PerformanceVisualizer
from openevolve.analysis.success_rates import compute_success_rates

logger = logging.getLogger(__name__)


def extract_iteration_from_dirname(dirname: str) -> Optional[int]:
    """Extract iteration number from directory name like 'iter_123_gen_04_abc123_175332'"""
    match = re.match(r'iter_(\d+)_', dirname)
    if match:
        return int(match.group(1))
    return None


def load_historical_records(historical_dir: Path) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Load all historical records and return sorted list of (iteration, metadata)
    """
    records = []
    iteration_counts = {}
    
    if not historical_dir.exists():
        logger.warning(f"Historical records directory not found: {historical_dir}")
        return records
    
    # Get all iteration directories
    iter_dirs = [d for d in historical_dir.iterdir() if d.is_dir() and d.name.startswith('iter_')]
    
    # First pass: check for duplicates
    for iter_dir in iter_dirs:
        iteration = extract_iteration_from_dirname(iter_dir.name)
        if iteration is not None:
            iteration_counts[iteration] = iteration_counts.get(iteration, 0) + 1
    
    # Report duplicates
    duplicates = {k: v for k, v in iteration_counts.items() if v > 1}
    if duplicates:
        logger.warning(f"Found duplicate iterations: {duplicates}")
        for iter_num, count in duplicates.items():
            logger.warning(f"  Iteration {iter_num} appears {count} times")
    
    # Second pass: load records (prefer lowest generation for duplicates)
    seen_iterations = {}  # iteration -> (generation, dir_name)
    for iter_dir in iter_dirs:
        iteration = extract_iteration_from_dirname(iter_dir.name)
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
        iteration = extract_iteration_from_dirname(iter_dir.name)
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
            logger.warning(f"Failed to load {metadata_file}: {e}")
            continue
    
    # Sort by iteration number
    records.sort(key=lambda x: x[0])
    logger.info(f"Loaded {len(records)} unique historical records")
    
    return records


# Success rate calculation functions moved to openevolve.analysis.success_rates


def reconstruct_performance_data(historical_dir: Path, experiment_id: str) -> PerformanceTracker:
    """
    Reconstruct PerformanceTracker from historical records
    """
    # Load historical records
    records = load_historical_records(historical_dir)
    
    if not records:
        raise ValueError("No historical records found to reconstruct performance data")
    
    # Create a dummy performance tracker (we'll manually populate it)
    output_dir = historical_dir.parent
    tracker = PerformanceTracker(str(output_dir), experiment_id)
    tracker.iteration_records = []
    
    # Convert historical records to IterationRecords
    for iteration, metadata in records:
        # Extract basic info
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
        
        # Create record (tokens info not available in historical records)
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
    
    logger.info(f"Reconstructed performance data: {len(tracker.iteration_records)} iterations")
    logger.info(f"Best score: {tracker.best_score:.4f} at iteration {tracker.best_iteration}")
    
    return tracker


def create_enhanced_visualizations(
    tracker: PerformanceTracker, 
    output_dir: Path,
    extraction_success_rate: float,
    validity_success_rate: float
) -> None:
    """
    Create enhanced visualizations with additional metrics
    """
    visualizer = PerformanceVisualizer(str(output_dir / "visualizations"))
    
    # Create standard plots using the enhanced visualizer (with historical records fallback)
    visualizer.plot_evolution_progress(tracker, scale="linear", output_dir=str(output_dir))
    visualizer.plot_evolution_progress(tracker, scale="log", output_dir=str(output_dir))
    
    # Create enhanced summary report with additional metrics
    create_enhanced_summary_report(tracker, output_dir, extraction_success_rate, validity_success_rate)


def create_enhanced_summary_report(
    tracker: PerformanceTracker,
    output_dir: Path,
    extraction_success_rate: float,
    validity_success_rate: float
) -> None:
    """
    Create enhanced HTML summary report with additional metrics
    """
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    report_path = viz_dir / f"{tracker.experiment_id}_enhanced_report.html"
    
    summary = tracker.get_summary()
    
    # Calculate additional statistics
    valid_scores = [r.metrics.get('combined_score', 0) for r in tracker.iteration_records 
                   if 'combined_score' in r.metrics and isinstance(r.metrics['combined_score'], (int, float))]
    
    score_improvement = 0.0
    if len(valid_scores) >= 2:
        score_improvement = valid_scores[-1] - valid_scores[0]
    
    # Create enhanced HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced OpenEvolve Report - {tracker.experiment_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ margin: 30px 0; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
        .metric {{ background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; }}
        .metric-success {{ background-color: #e8f5e8; border-left: 4px solid #4CAF50; }}
        .metric-warning {{ background-color: #fff3cd; border-left: 4px solid #ff9800; }}
        .metric-title {{ font-weight: bold; font-size: 1.1em; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #1976D2; }}
        .success-value {{ color: #388E3C; }}
        .warning-value {{ color: #F57C00; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .highlight {{ background-color: #ffeb3b !important; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Enhanced OpenEvolve Performance Report</h1>
        <h2>📊 Experiment: {tracker.experiment_id}</h2>
        <p><em>Reconstructed from historical records with enhanced metrics</em></p>
    </div>
    
    <div class="section">
        <h3>📈 Core Performance Metrics</h3>
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-title">Total Iterations</div>
                <div class="metric-value">{summary.get('total_iterations', 0):,}</div>
            </div>
            <div class="metric metric-success">
                <div class="metric-title">Best Score</div>
                <div class="metric-value success-value">{summary.get('best_score', 0):.6f}</div>
            </div>
            <div class="metric">
                <div class="metric-title">Best Iteration</div>
                <div class="metric-value">{summary.get('best_iteration', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-title">Score Improvement</div>
                <div class="metric-value {'success-value' if score_improvement > 0 else 'warning-value'}">{score_improvement:+.6f}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>🔧 Code Generation Quality</h3>
        <div class="metric-grid">
            <div class="metric {'metric-success' if extraction_success_rate > 0.8 else 'metric-warning' if extraction_success_rate > 0.5 else ''}">
                <div class="metric-title">Extraction Success Rate</div>
                <div class="metric-value {'success-value' if extraction_success_rate > 0.8 else 'warning-value' if extraction_success_rate > 0.5 else ''}">{extraction_success_rate:.1%}</div>
            </div>
            <div class="metric {'metric-success' if validity_success_rate > 0.8 else 'metric-warning' if validity_success_rate > 0.5 else ''}">
                <div class="metric-title">Validity Success Rate</div>
                <div class="metric-value {'success-value' if validity_success_rate > 0.8 else 'warning-value' if validity_success_rate > 0.5 else ''}">{validity_success_rate:.1%}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>📋 Recent Performance History</h3>
        <table>
            <tr>
                <th>Iteration</th>
                <th>Program ID</th>
                <th>Combined Score</th>
                <th>Validity</th>
                <th>Generation</th>
                <th>Eval Time (s)</th>
                <th>New Best?</th>
            </tr>
"""
    
    # Add recent iterations to table (last 20)
    recent_records = tracker.iteration_records[-20:] if len(tracker.iteration_records) > 20 else tracker.iteration_records
    for record in recent_records:
        score = record.metrics.get('combined_score', 'N/A')
        validity = record.metrics.get('validity', 'N/A')
        eval_time = record.metrics.get('eval_time', record.evaluation_time)
        
        score_str = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
        validity_str = f"{validity:.2f}" if isinstance(validity, (int, float)) else str(validity)
        eval_time_str = f"{eval_time:.2f}" if isinstance(eval_time, (int, float)) else str(eval_time)
        
        row_class = "highlight" if record.is_new_best else ""
        best_marker = "🎯" if record.is_new_best else ""
        
        html_content += f"""
            <tr class="{row_class}">
                <td>{record.iteration}</td>
                <td title="{record.program_id}">{record.program_id[:12]}...</td>
                <td>{score_str}</td>
                <td>{validity_str}</td>
                <td>{record.generation}</td>
                <td>{eval_time_str}</td>
                <td>{best_marker}</td>
            </tr>
"""
    
    html_content += f"""
        </table>
    </div>
    
    <div class="section">
        <h3>ℹ️ Data Sources</h3>
        <ul>
            <li><strong>Performance data:</strong> Reconstructed from historical_records/</li>
            <li><strong>Extraction rates:</strong> Based on extracted_prompts/ directories</li>
            <li><strong>Compilation rates:</strong> Based on validity metrics in metadata</li>
            <li><strong>Token usage:</strong> Not available (historical records limitation)</li>
        </ul>
    </div>
    
    <div class="section">
        <h3>📊 Visualization Files</h3>
        <ul>
            <li><strong>Linear Progress:</strong> {tracker.experiment_id}_progress_linear.png</li>
            <li><strong>Log Progress:</strong> {tracker.experiment_id}_progress_log.png</li>
        </ul>
    </div>
    
    <p><em>Report generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
</body>
</html>
"""
    
    # Save HTML report
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📋 Saved enhanced report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Recompute visualizations from historical records")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to openevolve output directory (containing historical_records/)"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID (auto-detected from directory name if not provided)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    # Auto-detect experiment ID if not provided
    experiment_id = args.experiment_id
    if not experiment_id:
        experiment_id = output_dir.name
        logger.info(f"Auto-detected experiment ID: {experiment_id}")
    
    historical_dir = output_dir / "historical_records"
    
    try:
        logger.info(f"🔄 Recomputing visualizations for: {output_dir}")
        
        # Compute success rates using centralized module
        success_rates = compute_success_rates(output_dir)
        extraction_rate = success_rates["extraction_rate"] 
        validity_rate = success_rates["validity_rate"]
        
        # Reconstruct performance data
        tracker = reconstruct_performance_data(historical_dir, experiment_id)
        
        # Create enhanced visualizations
        create_enhanced_visualizations(tracker, output_dir, extraction_rate, validity_rate)
        
        logger.info("✅ Visualization recomputation completed successfully!")
        logger.info(f"📁 Check visualizations/ directory in {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to recompute visualizations: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()