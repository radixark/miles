"""
Performance tracking for OpenEvolve evolution runs
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openevolve.utils.format_utils import format_metrics_safe

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Record for a single iteration's performance data"""
    iteration: int
    timestamp: float
    program_id: str
    metrics: Dict[str, Any]
    generation: int
    iteration_found: int
    parent_id: Optional[str] = None
    
    # Token usage
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    evaluation_time: float = 0.0
    llm_time: float = 0.0
    
    # Performance tracking
    improvement: Dict[str, float] = field(default_factory=dict)
    is_new_best: bool = False


class PerformanceTracker:
    """
    Real-time performance tracking for OpenEvolve
    
    Tracks iteration-by-iteration metrics, token usage, and compute costs
    with automatic saving and summary generation.
    """
    
    def __init__(self, output_dir: str, experiment_id: Optional[str] = None):
        """Initialize performance tracker"""
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        
        # Create performance directory
        self.performance_dir = self.output_dir / "performance"
        self.performance_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.iteration_records: List[IterationRecord] = []
        self.start_time = time.time()
        self.best_score = 0.0
        self.best_program_id = ""
        self.best_iteration = 0
        
        # File paths
        self.records_file = self.performance_dir / f"{self.experiment_id}_iterations.json"
        self.summary_file = self.performance_dir / f"{self.experiment_id}_summary.json"
        self.csv_file = self.performance_dir / f"{self.experiment_id}_data.csv"
        
        logger.info(f"📊 Performance tracker initialized: {self.experiment_id}")
        logger.debug(f"📁 Data directory: {self.performance_dir}")
        
    def record_iteration(
        self,
        iteration: int,
        program_id: str,
        metrics: Dict[str, Any],
        generation: int,
        iteration_found: int,
        parent_id: Optional[str] = None,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        evaluation_time: float = 0.0,
        llm_time: float = 0.0,
        improvement: Optional[Dict[str, float]] = None
    ) -> IterationRecord:
        """Record performance data for a single iteration"""
        
        # Calculate total tokens
        total_tokens = prompt_tokens + response_tokens
        
        # Check if this is a new best score
        is_new_best = False
        if "combined_score" in metrics:
            score = metrics["combined_score"]
            if isinstance(score, (int, float)) and score > self.best_score:
                is_new_best = True
                self.best_score = score
                self.best_program_id = program_id
                self.best_iteration = iteration
        
        # Create record
        record = IterationRecord(
            iteration=iteration,
            timestamp=time.time(),
            program_id=program_id,
            metrics=metrics,
            generation=generation,
            iteration_found=iteration_found,
            parent_id=parent_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            evaluation_time=evaluation_time,
            llm_time=llm_time,
            improvement=improvement or {},
            is_new_best=is_new_best
        )
        
        self.iteration_records.append(record)
        
        # Auto-save every 10 iterations
        if len(self.iteration_records) % 10 == 0:
            self.save_data()
        
        if is_new_best:
            logger.info(f"🎯 New best score: {self.best_score:.4f} at iteration {iteration}")
        
        return record
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        if not self.iteration_records:
            return {"status": "no_data"}
        
        latest = self.iteration_records[-1]
        total_tokens = sum(r.total_tokens for r in self.iteration_records)
        total_time = time.time() - self.start_time
        
        return {
            "experiment_id": self.experiment_id,
            "total_iterations": len(self.iteration_records),
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "latest_score": latest.metrics.get("combined_score", 0),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "avg_tokens_per_iter": total_tokens / len(self.iteration_records),
            "avg_time_per_iter": total_time / len(self.iteration_records)
        }
    
    def save_data(self) -> None:
        """Save current data to JSON and CSV files"""
        try:
            # Save records as JSON
            records_data = []
            for record in self.iteration_records:
                records_data.append({
                    "iteration": record.iteration,
                    "timestamp": record.timestamp,
                    "program_id": record.program_id,
                    "metrics": record.metrics,
                    "generation": record.generation,
                    "iteration_found": record.iteration_found,
                    "parent_id": record.parent_id,
                    "prompt_tokens": record.prompt_tokens,
                    "response_tokens": record.response_tokens,
                    "total_tokens": record.total_tokens,
                    "evaluation_time": record.evaluation_time,
                    "llm_time": record.llm_time,
                    "improvement": record.improvement,
                    "is_new_best": record.is_new_best
                })
            
            with open(self.records_file, 'w') as f:
                json.dump(records_data, f, indent=2)
            
            # Save summary
            summary = {
                "experiment_id": self.experiment_id,
                "start_time": self.start_time,
                "end_time": time.time(),
                "total_iterations": len(self.iteration_records),
                "best_score": self.best_score,
                "best_program_id": self.best_program_id,
                "best_iteration": self.best_iteration,
                "total_tokens": sum(r.total_tokens for r in self.iteration_records),
                "total_time": time.time() - self.start_time
            }
            
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save CSV
            self._save_csv()
            
            logger.debug(f"💾 Saved performance data: {len(self.iteration_records)} iterations")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save performance data: {e}")
    
    def _save_csv(self) -> None:
        """Save data in CSV format for analysis"""
        import csv
        
        if not self.iteration_records:
            return
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                "iteration", "timestamp", "program_id", "generation", "iteration_found",
                "parent_id", "prompt_tokens", "response_tokens", "total_tokens",
                "evaluation_time", "llm_time", "is_new_best"
            ]
            
            # Add metric columns
            if self.iteration_records:
                for metric_name in self.iteration_records[0].metrics.keys():
                    header.append(f"metric_{metric_name}")
            
            writer.writerow(header)
            
            # Data rows
            for record in self.iteration_records:
                row = [
                    record.iteration, record.timestamp, record.program_id,
                    record.generation, record.iteration_found, record.parent_id,
                    record.prompt_tokens, record.response_tokens, record.total_tokens,
                    record.evaluation_time, record.llm_time, record.is_new_best
                ]
                
                # Add metric values
                if self.iteration_records:
                    for metric_name in self.iteration_records[0].metrics.keys():
                        row.append(record.metrics.get(metric_name, ""))
                
                writer.writerow(row)
    
    def finalize(self) -> None:
        """Finalize experiment and save final data"""
        self.save_data()
        
        total_tokens = sum(r.total_tokens for r in self.iteration_records)
        total_time = time.time() - self.start_time
        
        logger.info(f"🏁 Experiment {self.experiment_id} completed:")
        logger.info(f"   Iterations: {len(self.iteration_records)}")
        logger.info(f"   Best score: {self.best_score:.4f}")
        logger.info(f"   Total tokens: {total_tokens:,}")
        logger.info(f"   Total time: {total_time:.2f}s")
    
    @classmethod
    def load_experiment(cls, performance_dir: str, experiment_id: str) -> "PerformanceTracker":
        """Load existing experiment data"""
        performance_path = Path(performance_dir)
        records_file = performance_path / f"{experiment_id}_iterations.json"
        
        if not records_file.exists():
            raise FileNotFoundError(f"Experiment data not found: {experiment_id}")
        
        # Create tracker
        tracker = cls(str(performance_path.parent), experiment_id)
        
        # Load records
        with open(records_file, 'r') as f:
            records_data = json.load(f)
        
        tracker.iteration_records = []
        for data in records_data:
            record = IterationRecord(
                iteration=data["iteration"],
                timestamp=data["timestamp"],
                program_id=data["program_id"],
                metrics=data["metrics"],
                generation=data["generation"],
                iteration_found=data["iteration_found"],
                parent_id=data.get("parent_id"),
                prompt_tokens=data.get("prompt_tokens", 0),
                response_tokens=data.get("response_tokens", 0),
                total_tokens=data.get("total_tokens", 0),
                evaluation_time=data.get("evaluation_time", 0.0),
                llm_time=data.get("llm_time", 0.0),
                improvement=data.get("improvement", {}),
                is_new_best=data.get("is_new_best", False)
            )
            tracker.iteration_records.append(record)
        
        # Update best score tracking
        for record in tracker.iteration_records:
            if record.is_new_best and "combined_score" in record.metrics:
                score = record.metrics["combined_score"]
                if isinstance(score, (int, float)) and score > tracker.best_score:
                    tracker.best_score = score
                    tracker.best_program_id = record.program_id
                    tracker.best_iteration = record.iteration
        
        logger.info(f"📂 Loaded experiment: {experiment_id} ({len(tracker.iteration_records)} iterations)")
        
        return tracker