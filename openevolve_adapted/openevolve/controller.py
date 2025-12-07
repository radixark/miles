"""
Main controller for OpenEvolve
"""

import asyncio
import logging
import os
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.process_parallel import ProcessParallelController
from openevolve.utils.code_utils import (
    extract_code_language,
)
from openevolve.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)
from openevolve.analysis.performance_tracker import PerformanceTracker
from openevolve.modular_utils.error_constants import ErrorThresholds, get_visualization_safe_score

logger = logging.getLogger(__name__)


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Safely format metrics, handling both numeric and string values"""
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")
    return ", ".join(formatted_parts)


def _format_improvement(improvement: Dict[str, Any]) -> str:
    """Safely format improvement metrics"""
    formatted_parts = []
    for name, diff in improvement.items():
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            try:
                formatted_parts.append(f"{name}={diff:+.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={diff}")
        else:
            formatted_parts.append(f"{name}={diff}")
    return ", ".join(formatted_parts)


class OpenEvolve:
    """
    Main controller for OpenEvolve

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
    ):
        # Load configuration
        if config is not None:
            # Use provided Config object directly
            self.config = config
        else:
            # Load from file or use defaults
            self.config = load_config(config_path)

        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            base_dir = "openevolve_output"
            if self.config.output_postfix:
                base_dir = f"openevolve_output_{self.config.output_postfix}"
            self.output_dir = os.path.join(os.path.dirname(initial_program_path), base_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up checkpoint directory
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        
        # Set up historical records directory
        self.historical_records_dir = os.path.join(self.output_dir, "historical_records")
        
        # Track programs evaluated per iteration for checkpoint runtime environment saving
        self.iteration_programs = {}  # iteration -> program_id

        # Set up performance tracking
        self.performance_tracker = PerformanceTracker(
            self.output_dir, 
            experiment_id=self.config.output_postfix or f"openevolve_{int(time.time())}"
        )

        # Set up logging
        self._setup_logging()

        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            import random
            import numpy as np
            import hashlib

            # Set global random seeds
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

            # Create hash-based seeds for different components
            base_seed = str(self.config.random_seed).encode("utf-8")
            llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (2**31)

            # Propagate seed to LLM configurations
            self.config.llm.random_seed = llm_seed
            for model_cfg in self.config.llm.models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed

            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")
            logger.debug(f"Generated LLM seed: {llm_seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Pass random seed to database if specified
        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.database = ProgramDatabase(self.config.database)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
        )
        self.evaluation_file = evaluation_file

        logger.info(f"Initialized OpenEvolve with {initial_program_path}")

        # Initialize improved parallel processing components
        self.parallel_controller = None

    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # Add file handler
        log_file = os.path.join(log_dir, f"openevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """
        Run the evolution process with improved parallel processing

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)
            checkpoint_path: Path to resume from checkpoint

        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations

        # Determine starting iteration
        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                language=self.config.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
            )

            self.database.add(initial_program)
            
            # Check if combined_score is present in the metrics
            if "combined_score" not in initial_metrics:
                # Calculate average of numeric metrics
                numeric_metrics = [
                    v for v in initial_metrics.values() 
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_metrics:
                    avg_score = sum(numeric_metrics) / len(numeric_metrics)
                    logger.warning(
                        f"⚠️  No 'combined_score' metric found in evaluation results. "
                        f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                        f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                        f"metric that properly weights different aspects of program performance."
                    )
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} "
                f"with {len(self.database.programs)} existing programs)"
            )

        # Initialize improved parallel processing
        try:
            self.parallel_controller = ProcessParallelController(
                self.config, self.evaluation_file, self.database
            )

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.parallel_controller.request_shutdown()

                # Set up a secondary handler for immediate exit if user presses Ctrl+C again
                def force_exit_handler(signum, frame):
                    logger.info("Force exit requested - terminating immediately")
                    import sys

                    sys.exit(0)

                signal.signal(signal.SIGINT, force_exit_handler)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.parallel_controller.start()

            # When starting from iteration 0, we've already done the initial program evaluation
            # So we need to adjust the start_iteration for the actual evolution
            evolution_start = start_iteration
            evolution_iterations = max_iterations
            
            # If we just added the initial program at iteration 0, start evolution from iteration 1
            if should_add_initial and start_iteration == 0:
                evolution_start = 1
                # User expects max_iterations evolutionary iterations AFTER the initial program
                # So we don't need to reduce evolution_iterations
                
            # Run evolution with improved parallel processing and checkpoint callback
            await self._run_evolution_with_checkpoints(
                evolution_start, evolution_iterations, target_score
            )

        finally:
            # Clean up parallel processing resources
            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None
            
            # Clean up any remaining runtime environments
            if self.evaluator:
                self.evaluator.cleanup_runtime_environments()

        # Get the best program
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        # Check if there's a better program by combined_score that wasn't tracked
        if best_program and "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this program is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs "
                        f"{best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )
            self._save_best_program(best_program)
            
            # Finalize performance tracking and create final visualizations
            self.performance_tracker.finalize()
            self._create_performance_visualizations(is_final=True)
            
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            # Still finalize performance tracking
            self.performance_tracker.finalize()
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        llm_time: float = 0.0,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
            prompt_tokens: Number of prompt tokens used
            response_tokens: Number of response tokens generated
            llm_time: Time spent on LLM calls
        """
        # Calculate improvement using safe formatting
        improvement_dict = {}
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)
        
        # Calculate numeric improvements for tracking
        for key in child.metrics:
            if key in parent.metrics:
                try:
                    parent_val = parent.metrics[key]
                    child_val = child.metrics[key]
                    if isinstance(parent_val, (int, float)) and isinstance(child_val, (int, float)):
                        improvement_dict[key] = child_val - parent_val
                except (TypeError, ValueError):
                    pass

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )
        
        # Record performance data
        self.performance_tracker.record_iteration(
            iteration=iteration,
            program_id=child.id,
            metrics=child.metrics,
            generation=child.generation,
            iteration_found=child.iteration_found,
            parent_id=parent.id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            evaluation_time=elapsed_time - llm_time,
            llm_time=llm_time,
            improvement=improvement_dict
        )
        
        # Record which program was evaluated in this iteration for checkpoint runtime environment
        self.iteration_programs[iteration] = child.id
        logger.debug(f"📝 Recorded iteration {iteration} -> program {child.id}")
        
        # Historical records will be saved during checkpoint

    def set_iteration_program(self, iteration: int, program_id: str) -> None:
        """Record which program was evaluated in a specific iteration."""
        self.iteration_programs[iteration] = program_id
        logger.debug(f"📝 Set iteration {iteration} -> program {program_id}")

    def _save_checkpoint(self, iteration: int) -> None:
        """Save a checkpoint at the current iteration."""
        import json
        import time
        import glob
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save database (programs)
        self.database.save(checkpoint_path, iteration)

        # Debug: List all pending runtime environments  
        if self.evaluator:
            pending_envs = self.evaluator.list_pending_runtime_environments()
            logger.info(f"🗃️ Pending runtime environments at checkpoint {iteration}: {list(pending_envs.keys())}")
            logger.info(f"🗃️ Total pending environments: {len(pending_envs)}")
            
            # Log details of each pending environment
            for env_id, env_path in pending_envs.items():
                exists = os.path.exists(env_path) if env_path else False
                logger.info(f"   📁 {env_id}: {env_path} (exists: {exists})")

        # Get runtime environment info from parallel controller
        if hasattr(self, 'parallel_controller') and hasattr(self.parallel_controller, '_runtime_env_sync'):
            sync_data = self.parallel_controller._runtime_env_sync
            logger.info(f"🔍 Available sync data: {list(sync_data.keys()) if sync_data else 'None'}")
            
            # Find the program for this iteration
            iteration_program_id = None
            runtime_env_path = None
            
            for program_id, data in sync_data.items():
                if data['iteration'] == iteration:
                    iteration_program_id = program_id
                    runtime_env_path = data['path']
                    logger.info(f"📊 Found exact match for iteration {iteration}: {program_id}")
                    break
                    
            # If no exact match, try to find the closest iteration (fallback)
            if not iteration_program_id and sync_data:
                logger.info(f"🔄 No exact match for iteration {iteration}, looking for closest...")
                closest_iteration = None
                for program_id, data in sync_data.items():
                    sync_iteration = data.get('iteration', -1)
                    if closest_iteration is None or abs(sync_iteration - iteration) < abs(closest_iteration - iteration):
                        closest_iteration = sync_iteration
                        iteration_program_id = program_id
                        runtime_env_path = data['path']
                logger.info(f"🔄 Using closest iteration {closest_iteration} for checkpoint {iteration}")
            
            if iteration_program_id and runtime_env_path:
                logger.info(f"📊 Iteration {iteration} evaluated program: {iteration_program_id}")
                logger.info(f"🗂️ Using synced runtime environment: {runtime_env_path}")
                
                # Check if path exists before using it
                if os.path.exists(runtime_env_path):
                    # Manually sync to main evaluator
                    if self.evaluator:
                        self.evaluator._pending_runtime_environments[iteration_program_id] = runtime_env_path
                        self.iteration_programs[iteration] = iteration_program_id
                        logger.info(f"✅ Synced runtime environment for {iteration_program_id} to main evaluator")
                    
                    self._save_runtime_environment(checkpoint_path, iteration_program_id)
                else:
                    logger.warning(f"⚠️ Runtime environment path doesn't exist: {runtime_env_path}")
            else:
                logger.warning(f"⚠️ No runtime environment sync data found for iteration {iteration}")
        else:
            logger.warning(f"⚠️ No parallel controller or sync data available")

        # Save the best program found so far (but don't save its runtime environment here)
        best_program = self.database.get_best_program()
        logger.info(f"👑 Using tracked best program ID: {best_program.id if best_program else 'None'}")

        if best_program:
            logger.info(f"💾 Saving best program {best_program.id} to checkpoint {iteration}")
            
            # Save the best program at this checkpoint
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            # Save program info
            best_program_info = {
                "id": best_program.id,
                "iteration": iteration,
                "metrics": best_program.metrics,
                "timestamp": best_program.timestamp,
                "generation": best_program.generation,
                "iteration_found": best_program.iteration_found,
                "parent_id": best_program.parent_id,
                "complexity": best_program.complexity,
                "diversity": best_program.diversity
            }
            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                json.dump(best_program_info, f, indent=2)

        # Save checkpoint metadata
        metadata = {
            "iteration": iteration,
            "timestamp": time.time(),
            "total_programs": len(self.database.programs),
            "best_program_id": best_program.id if best_program else None,
            "best_score": best_program.metrics.get("combined_score", 0) if best_program else 0
        }
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 🚀 NEW: Save all program source codes for easy comparison
        self._save_all_program_codes(checkpoint_path, iteration)

        # # 🚀 NEW: Save historical records for all programs
        # self._save_all_historical_records(iteration)
        
        # 🚀 NEW: Update performance data from database
        self._update_performance_from_database(iteration)

        # 🚀 NEW: Update best folder at each checkpoint
        if best_program:
            self._save_best_program(best_program)

        # 🚀 NEW: Update performance visualizations at each checkpoint
        if self.config.enable_realtime_visualizations:
            self._create_performance_visualizations(is_final=False)

        # 🚀 NEW: Clean up old checkpoints to save space
        self._cleanup_old_checkpoints(iteration)

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _save_runtime_environment(self, checkpoint_path: str, program_id: str) -> None:
        """
        Save runtime environment for a program to the checkpoint directory
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            program_id: ID of the program
        """
        if not self.evaluator:
            return
            
        logger.info(f"🔍 Attempting to save runtime environment for program {program_id}")
        
        # Get all available runtime environments
        all_envs = self.evaluator.list_pending_runtime_environments()
        logger.info(f"🗃️ Available runtime environments: {list(all_envs.keys())}")
        
        # First try to get from pending environments
        runtime_env_dir = all_envs.get(program_id)
        logger.info(f"🎯 Runtime environment for {program_id}: {runtime_env_dir}")
        
        # If not found in pending, search temp directories as backup
        if not runtime_env_dir or not os.path.exists(runtime_env_dir):
            logger.warning(f"🔍 Runtime environment not found in pending, searching temp directories...")
            
            # Search for runtime environment in temp directories
            temp_dirs = []
            for temp_root in ['/tmp']:
                if os.path.exists(temp_root):
                    try:
                        import glob
                        pattern = os.path.join(temp_root, f"runtime_env_{program_id}_*")
                        matches = glob.glob(pattern)
                        temp_dirs.extend(matches)
                        logger.info(f"   🗂️ Found {len(matches)} potential directories for {program_id}")
                    except Exception as e:
                        logger.debug(f"Error searching temp dirs: {e}")
            
            if temp_dirs:
                # Use the most recent directory
                runtime_env_dir = max(temp_dirs, key=os.path.getmtime)
                logger.info(f"🚨 Using found runtime environment: {runtime_env_dir}")
            else:
                logger.warning(f"⚠️ No runtime environment found anywhere for {program_id}")
        
        if runtime_env_dir and os.path.exists(runtime_env_dir):
            try:
                # Create runtime environment directory in checkpoint
                checkpoint_runtime_dir = os.path.join(checkpoint_path, "runtime_environment")
                
                logger.info(f"📁 Copying runtime environment from {runtime_env_dir} to {checkpoint_runtime_dir}")
                
                # Copy runtime environment to checkpoint
                shutil.copytree(runtime_env_dir, checkpoint_runtime_dir, dirs_exist_ok=True)
                
                # Count files for logging
                file_count = sum(len(files) for _, _, files in os.walk(checkpoint_runtime_dir))
                
                logger.info(f"✅ Saved runtime environment for program {program_id} to checkpoint: {file_count} files")
                
                # Log what specific run timestamp was saved
                checkpoints_in_env = os.path.join(checkpoint_runtime_dir, "checkpoints")
                if os.path.exists(checkpoints_in_env):
                    run_dirs = [d for d in os.listdir(checkpoints_in_env) if d.startswith("run_")]
                    if run_dirs:
                        logger.info(f"📅 Saved runtime environment contains: {run_dirs}")
                
                return  # Successfully saved
                
            except Exception as e:
                logger.warning(f"❌ Failed to save runtime environment for program {program_id}: {e}")
                import traceback
                logger.warning(f"❌ Error details: {traceback.format_exc()}")
        else:
            # This indicates a problem - log diagnostic info
            logger.warning(f"⚠️ No runtime environment found for program {program_id}")
            logger.info(f"🔍 Runtime environment path was: {runtime_env_dir}")
            logger.info(f"🔍 Path exists: {os.path.exists(runtime_env_dir) if runtime_env_dir else 'N/A'}")
            
            if all_envs:
                logger.info(f"🔍 Other available runtime environments: {list(all_envs.keys())}")
            else:
                logger.warning(f"⚠️ No runtime environments available at all!")

    def _extract_prompts_for_historical_record(self, program_data: dict, record_path: str) -> None:
        """
        Extract prompts for historical record in the same format as checkpoints.
        """
        try:
            prompts = program_data.get("prompts", {})
            if not prompts:
                return
            
            extracted_dir = os.path.join(record_path, "extracted_prompts")
            os.makedirs(extracted_dir, exist_ok=True)
            
            # Create program-specific directory using program ID
            program_id = program_data.get("id", "unknown")
            program_dir = os.path.join(extracted_dir, program_id[:8])
            os.makedirs(program_dir, exist_ok=True)
            
            # Save program info
            info = {
                "id": program_data.get("id", "unknown"),
                "generation": program_data.get("generation", "unknown"),
                "timestamp": program_data.get("timestamp", "unknown"),
                "iteration_found": program_data.get("iteration_found", "unknown"),
                "historical_iteration": program_data.get("historical_iteration", "unknown"),
                "metrics": program_data.get("metrics", {}),
                "parent_id": program_data.get("parent_id", "unknown")
            }
            
            info_path = os.path.join(program_dir, "program_info.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("=== HISTORICAL RECORD PROGRAM INFORMATION ===\n\n")
                for key, value in info.items():
                    if key == "metrics" and isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for metric_key, metric_value in value.items():
                            f.write(f"  {metric_key}: {metric_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Extract and save prompts
            for prompt_type, prompt_data in prompts.items():
                prompt_dir = os.path.join(program_dir, prompt_type)
                os.makedirs(prompt_dir, exist_ok=True)
                
                if isinstance(prompt_data, dict):
                    for section, content in prompt_data.items():
                        if isinstance(content, str) and content.strip():
                            output_path = os.path.join(prompt_dir, f"{section}.txt")
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(f"=== {prompt_type.upper()}_{section.upper()} ===\n\n")
                                f.write(content)
                                f.write("\n")
                        elif isinstance(content, list):
                            # Handle response arrays
                            for i, response in enumerate(content):
                                if isinstance(response, str) and response.strip():
                                    output_path = os.path.join(prompt_dir, f"{section}_{i+1}.txt")
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(f"=== {prompt_type.upper()}_{section.upper()}_{i+1} ===\n\n")
                                        f.write(response)
                                        f.write("\n")
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract prompts for historical record: {e}")

    def _save_all_historical_records(self, checkpoint_iteration: int) -> None:
        """
        Save historical records for all programs by copying from the current checkpoint
        This ensures we get the same complete data as checkpoints (including prompts)
        """
        import json
        import datetime
        import shutil
        
        try:
            # Create historical records directory if it doesn't exist
            os.makedirs(self.historical_records_dir, exist_ok=True)
            
            # Get the current checkpoint directory (where programs were just saved)
            current_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_iteration}")
            programs_dir = os.path.join(current_checkpoint_path, "programs")
            
            if not os.path.exists(programs_dir):
                logger.warning(f"⚠️ No programs directory found in current checkpoint: {programs_dir}")
                return
            
            # Get all program files from current checkpoint (these have complete data)
            program_files = [f for f in os.listdir(programs_dir) if f.endswith('.json')]
            logger.info(f"📚 Saving historical records for {len(program_files)} programs from checkpoint {checkpoint_iteration}")
            
            saved_count = 0
            for program_file in program_files:
                try:
                    program_path = os.path.join(programs_dir, program_file)
                    with open(program_path, 'r', encoding='utf-8') as f:
                        program_data = json.load(f)
                    
                    # Extract info for directory naming
                    program_id = program_data.get('id', 'unknown')
                    program_iteration = program_data.get('iteration_found', checkpoint_iteration)
                    program_generation = program_data.get('generation', 0)
                    timestamp = program_data.get('timestamp', time.time())
                    
                    # Create consistent directory name
                    dt = datetime.datetime.fromtimestamp(timestamp)
                    time_str = dt.strftime("%H%M%S")
                    record_dirname = f"iter_{program_iteration:02d}_gen_{program_generation:02d}_{program_id[:8]}_{time_str}"
                    record_path = os.path.join(self.historical_records_dir, record_dirname)
                    
                    # Skip if already exists to avoid duplicates
                    if os.path.exists(record_path):
                        continue
                    
                    os.makedirs(record_path, exist_ok=True)
                    
                    # Save program code
                    program_codes_dir = os.path.join(record_path, "program_codes")
                    os.makedirs(program_codes_dir, exist_ok=True)
                    code_filename = f"program{self.file_extension}"
                    code_path = os.path.join(program_codes_dir, code_filename)
                    with open(code_path, 'w') as f:
                        f.write(program_data.get('code', ''))
                    
                    # Save complete program JSON (including prompts from checkpoint)
                    programs_record_dir = os.path.join(record_path, "programs")
                    os.makedirs(programs_record_dir, exist_ok=True)
                    
                    # Use the complete program data from checkpoint (which includes prompts)
                    program_json = program_data.copy()  # Copy all data from checkpoint
                    program_json["checkpoint_iteration"] = checkpoint_iteration
                    
                    json_path = os.path.join(programs_record_dir, f"{program_id}.json")
                    with open(json_path, 'w') as f:
                        json.dump(program_json, f, indent=2)
                    
                    # Extract prompts (now with complete checkpoint data)
                    self._extract_prompts_for_historical_record(program_json, record_path)
                    
                    # Save iteration metadata
                    metadata = {
                        "iteration_found": program_iteration,
                        "checkpoint_iteration": checkpoint_iteration,
                        "program_id": program_id,
                        "timestamp": timestamp,
                        "metrics": program_data.get("metrics", {}),
                        "generation": program_generation,
                        "parent_id": program_data.get("parent_id", None)
                    }
                    
                    metadata_path = os.path.join(record_path, "metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save historical record for program from {program_file}: {e}")
            
            logger.info(f"📚 Saved {saved_count} new historical records")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save historical records: {e}")
    
    def _update_performance_from_database(self, checkpoint_iteration: int) -> None:
        """
        Update performance data retrospectively from database programs
        This is called during checkpoint to ensure performance tracking works with parallel processing
        """
        try:
            # Get all programs from database sorted by iteration_found
            all_programs = list(self.database.programs.values())
            all_programs.sort(key=lambda p: p.iteration_found)
            
            # Update performance tracker with all programs if it has fewer records than database
            current_records = len(self.performance_tracker.iteration_records)
            database_programs = len(all_programs)
            
            if current_records < database_programs:
                logger.info(f"📊 Updating performance data: {current_records} -> {database_programs} records")
                
                # Add missing records
                for program in all_programs:
                    # Check if this program is already recorded
                    already_recorded = any(
                        r.program_id == program.id 
                        for r in self.performance_tracker.iteration_records
                    )
                    
                    if not already_recorded:
                        # Calculate improvement compared to parent
                        improvement = {}
                        if program.parent_id:
                            parent = self.database.get(program.parent_id)
                            if parent:
                                for key in program.metrics:
                                    if key in parent.metrics:
                                        try:
                                            parent_val = parent.metrics[key]
                                            child_val = program.metrics[key]
                                            if isinstance(parent_val, (int, float)) and isinstance(child_val, (int, float)):
                                                improvement[key] = child_val - parent_val
                                        except (TypeError, ValueError):
                                            pass
                        
                        # Try to get token data from runtime environment
                        prompt_tokens = 0
                        response_tokens = 0
                        llm_time = 0.0
                        
                        # Check if we have runtime environment data for this program
                        if (hasattr(self, 'parallel_controller') and 
                            hasattr(self.parallel_controller, '_runtime_env_sync') and
                            program.id in (self.parallel_controller._runtime_env_sync or {})):
                            
                            runtime_env_path = self.parallel_controller._runtime_env_sync[program.id]['path']
                            if runtime_env_path and os.path.exists(runtime_env_path):
                                try:
                                    # Load token data from runtime environment artifacts
                                    checkpoints_dir = os.path.join(runtime_env_path, "checkpoints")
                                    if os.path.exists(checkpoints_dir):
                                        # Find the latest run directory
                                        run_dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith("run_")]
                                        if run_dirs:
                                            latest_run = max(run_dirs)
                                            run_path = os.path.join(checkpoints_dir, latest_run)
                                            
                                            # Look for artifacts.json which might contain token information
                                            artifacts_file = os.path.join(run_path, "artifacts.json")
                                            if os.path.exists(artifacts_file):
                                                with open(artifacts_file, 'r', encoding='utf-8') as f:
                                                    artifacts = json.load(f)
                                                    # Extract token data if available
                                                    if 'prompt_tokens' in artifacts:
                                                        prompt_tokens = artifacts.get('prompt_tokens', 0)
                                                    if 'response_tokens' in artifacts:
                                                        response_tokens = artifacts.get('response_tokens', 0)
                                                    if 'llm_time' in artifacts:
                                                        llm_time = artifacts.get('llm_time', 0.0)
                                                    
                                                    logger.debug(f"📊 Loaded token data for program {program.id}: {prompt_tokens}+{response_tokens} tokens")
                                except Exception as e:
                                    logger.debug(f"⚠️ Could not load token data for program {program.id}: {e}")

                        # Add record to performance tracker
                        self.performance_tracker.record_iteration(
                            iteration=program.iteration_found,
                            program_id=program.id,
                            metrics=program.metrics,
                            generation=program.generation,
                            iteration_found=program.iteration_found,
                            parent_id=program.parent_id,
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                            evaluation_time=program.metrics.get('eval_time', 0.0),
                            llm_time=llm_time,
                            improvement=improvement
                        )
                
                logger.info(f"📊 Performance data updated to {len(self.performance_tracker.iteration_records)} records")
            
            # Save updated performance data
            self.performance_tracker.save_data()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update performance data: {e}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load state from a checkpoint directory"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.database.load(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully (iteration {self.database.last_iteration})")

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:
        """Run evolution with checkpoint saving support"""
        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()

        # Run the evolution process with checkpoint callback
        await self.parallel_controller.run_evolution(
            start_iteration, max_iterations, target_score, checkpoint_callback=self._save_checkpoint
        )

        # Check if shutdown was requested
        if self.parallel_controller.shutdown_event.is_set():
            logger.info("Evolution stopped due to shutdown request")
            return

        # Save final checkpoint if needed
        # Note: start_iteration here is the evolution start (1 for fresh start, not 0)
        # max_iterations is the number of evolution iterations to run
        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save runtime environment if available
        self._save_runtime_environment(best_dir, program.id)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")

    def _save_all_program_codes(self, checkpoint_path: str, iteration: int) -> None:
        """
        Save all program source codes with consistent iter_{N} naming and generate extracted prompts.
        Also rename JSON files to match code files for consistency.
        """
        import json
        import datetime
        import shutil
        
        try:
            programs_dir = os.path.join(checkpoint_path, "programs")
            codes_dir = os.path.join(checkpoint_path, "program_codes")
            extracted_dir = os.path.join(checkpoint_path, "extracted_prompts")
            
            os.makedirs(codes_dir, exist_ok=True)
            os.makedirs(extracted_dir, exist_ok=True)
            
            if not os.path.exists(programs_dir):
                logger.warning(f"⚠️ Programs directory not found: {programs_dir}")
                return
                
            # Get all program files
            program_files = [f for f in os.listdir(programs_dir) if f.endswith('.json')]
            logger.info(f"📋 Processing {len(program_files)} programs with new naming scheme")
            
            # Track processed programs for renaming
            program_mapping = {}  # old_filename -> new_filename
            
            for program_file in program_files:
                try:
                    program_path = os.path.join(programs_dir, program_file)
                    with open(program_path, 'r') as f:
                        program_data = json.load(f)
                    
                    program_id = program_data.get('id', program_file.replace('.json', ''))
                    program_code = program_data.get('code', '')
                    program_iteration = program_data.get('iteration_found', iteration)  # Use iteration_found or current iteration
                    program_generation = program_data.get('generation', 0)  # Evolution generation
                    timestamp = program_data.get('timestamp', 0)
                    
                    # Create consistent filename: iter_{iteration}_gen_{generation}_{id}_{time}
                    if timestamp:
                        # Convert timestamp to readable format
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        time_str = dt.strftime("%H%M%S")
                        base_filename = f"iter_{program_iteration:02d}_gen_{program_generation:02d}_{program_id[:8]}_{time_str}"
                    else:
                        base_filename = f"iter_{program_iteration:02d}_gen_{program_generation:02d}_{program_id[:8]}"
                    
                    # Save program code
                    code_filename = f"{base_filename}{self.file_extension}"
                    code_path = os.path.join(codes_dir, code_filename)
                    with open(code_path, 'w') as f:
                        f.write(program_code)
                    
                    # Save JSON with consistent naming
                    json_filename = f"{base_filename}.json"
                    new_json_path = os.path.join(programs_dir, json_filename)
                    
                    # Only rename if it's different
                    if program_file != json_filename:
                        program_mapping[program_file] = json_filename
                        shutil.move(program_path, new_json_path)
                        logger.debug(f"Renamed {program_file} -> {json_filename}")
                    
                    # Extract and save prompts
                    self._extract_prompts_for_program(program_data, extracted_dir, base_filename)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process {program_file}: {e}")
            
            logger.info(f"✅ Processed {len(program_files)} programs:")
            logger.info(f"   📁 Saved program codes to {codes_dir}")
            logger.info(f"   📄 Renamed {len(program_mapping)} JSON files for consistency")
            logger.info(f"   🗂️ Extracted prompts to {extracted_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to process program codes: {e}")
    
    def _extract_prompts_for_program(self, program_data: dict, extracted_dir: str, base_filename: str) -> None:
        """
        Extract prompts for a single program and save as text files.
        """
        try:
            prompts = program_data.get("prompts", {})
            if not prompts:
                return
            
            # Create program-specific directory
            program_dir = os.path.join(extracted_dir, base_filename)
            os.makedirs(program_dir, exist_ok=True)
            
            # Save program info
            info = {
                "id": program_data.get("id", "unknown"),
                "generation": program_data.get("generation", "unknown"),
                "timestamp": program_data.get("timestamp", "unknown"),
                "iteration_found": program_data.get("iteration_found", "unknown"),
                "metrics": program_data.get("metrics", {}),
                "parent_id": program_data.get("parent_id", "unknown")
            }
            
            info_path = os.path.join(program_dir, "program_info.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("=== PROGRAM INFORMATION ===\n\n")
                for key, value in info.items():
                    if key == "metrics" and isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for metric_key, metric_value in value.items():
                            f.write(f"  {metric_key}: {metric_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Extract and save prompts
            for prompt_type, prompt_data in prompts.items():
                prompt_dir = os.path.join(program_dir, prompt_type)
                os.makedirs(prompt_dir, exist_ok=True)
                
                if isinstance(prompt_data, dict):
                    for section, content in prompt_data.items():
                        if isinstance(content, str) and content.strip():
                            output_path = os.path.join(prompt_dir, f"{section}.txt")
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(f"=== {prompt_type.upper()}_{section.upper()} ===\n\n")
                                f.write(content)
                                f.write("\n")
                        elif isinstance(content, list):
                            # Handle response arrays
                            for i, response in enumerate(content):
                                if isinstance(response, str) and response.strip():
                                    output_path = os.path.join(prompt_dir, f"{section}_{i+1}.txt")
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(f"=== {prompt_type.upper()}_{section.upper()}_{i+1} ===\n\n")
                                        f.write(response)
                                        f.write("\n")
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract prompts for {base_filename}: {e}")
    
    def _cleanup_old_checkpoints(self, current_iteration: int) -> None:
        """
        Clean up old checkpoints to save space by removing redundant files.
        Preserve full checkpoint data at regular intervals based on codes_interval_ratio.
        """
        try:
            if not os.path.exists(self.checkpoint_dir):
                return
            
            # Get all checkpoint directories
            checkpoint_dirs = []
            for item in os.listdir(self.checkpoint_dir):
                if item.startswith("checkpoint_") and os.path.isdir(os.path.join(self.checkpoint_dir, item)):
                    try:
                        checkpoint_num = int(item.split("_")[1])
                        checkpoint_dirs.append((checkpoint_num, item))
                    except (ValueError, IndexError):
                        continue
            
            # Sort by checkpoint number
            checkpoint_dirs.sort()
            
            # Calculate preservation interval
            preservation_interval = self.config.checkpoint_interval * self.config.codes_interval_ratio
            
            # Keep only essential files in older checkpoints (not the current one)
            cleaned_count = 0
            preserved_count = 0
            
            for checkpoint_num, checkpoint_name in checkpoint_dirs:
                # Skip current checkpoint and the most recent 2 checkpoints for safety
                if checkpoint_num >= current_iteration - self.config.checkpoint_interval:
                    continue
                
                # Check if this checkpoint should preserve full data
                should_preserve_full = (checkpoint_num % preservation_interval) == 0
                
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
                essential_files = {"best_program", "best_program_info.json", "metadata.json"}
                
                if should_preserve_full:
                    logger.debug(f"🔒 Preserving full checkpoint data for {checkpoint_name}")
                    preserved_count += 1
                    continue
                
                try:
                    # Remove programs directory (large and redundant)
                    programs_dir = os.path.join(checkpoint_path, "programs")
                    if os.path.exists(programs_dir):
                        import shutil
                        shutil.rmtree(programs_dir)
                        logger.debug(f"🗑️ Removed programs directory from {checkpoint_name}")
                    
                    # Remove program_codes directory (also redundant)
                    codes_dir = os.path.join(checkpoint_path, "program_codes")
                    if os.path.exists(codes_dir):
                        import shutil
                        shutil.rmtree(codes_dir)
                        logger.debug(f"🗑️ Removed program_codes directory from {checkpoint_name}")
                    
                    # Remove extracted_prompts directory (can be regenerated if needed)
                    extracted_dir = os.path.join(checkpoint_path, "extracted_prompts")
                    if os.path.exists(extracted_dir):
                        import shutil
                        shutil.rmtree(extracted_dir)
                        logger.debug(f"🗑️ Removed extracted_prompts directory from {checkpoint_name}")
                    
                    # Remove runtime environment directories (also redundant)
                    for item in os.listdir(checkpoint_path):
                        item_path = os.path.join(checkpoint_path, item)
                        if os.path.isdir(item_path) and item not in {".", ".."}:
                            if not any(item.startswith(essential) for essential in essential_files):
                                import shutil
                                shutil.rmtree(item_path)
                                logger.debug(f"🗑️ Removed directory {item} from {checkpoint_name}")
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean checkpoint {checkpoint_name}: {e}")
            
            if cleaned_count > 0 or preserved_count > 0:
                logger.info(f"🧹 Cleaned {cleaned_count} checkpoints, preserved {preserved_count} full checkpoints (interval: {preservation_interval})")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup old checkpoints: {e}")

    def _calculate_visualization_fallback(self) -> float:
        """
        Calculate fallback value for error codes in visualizations.
        Uses -0.1 as default, but if -0.1 is larger than the minimum valid score,
        uses twice the minimum valid score (making it smaller).

        Returns:
            Fallback value for error codes in visualizations
        """
        try:
            # Get all valid scores from performance tracker
            valid_scores = []
            for record in self.performance_tracker.iteration_records:
                if 'combined_score' in record.metrics:
                    score = record.metrics['combined_score']
                    if isinstance(score, (int, float)) and not ErrorThresholds.is_error_code(int(score)):
                        valid_scores.append(score)

            # Default fallback value
            default_fallback = -0.1

            if not valid_scores:
                # No valid scores found, use default
                logger.debug(f"📊 No valid scores found, using default fallback: {default_fallback}")
                return default_fallback

            min_valid_score = min(valid_scores)

            if default_fallback < min_valid_score:
                # -0.1 is smaller than minimum valid score, use it
                logger.debug(f"📊 Using default fallback {default_fallback} (min_valid: {min_valid_score:.4f})")
                return default_fallback
            else:
                # -0.1 is larger than minimum valid score, use twice the minimum to make it smaller
                fallback = 2 * min_valid_score
                logger.debug(f"📊 Default fallback {default_fallback} >= min_valid {min_valid_score:.4f}, using {fallback:.4f}")
                return fallback

        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate visualization fallback: {e}")
            return -0.1  # Safe default

    def _create_performance_visualizations(self, is_final: bool = False) -> None:
        """
        Create performance visualizations (real-time updates during evolution)

        Args:
            is_final: True if this is the final visualization at experiment end
        """
        try:
            from openevolve.analysis.visualizer import PerformanceVisualizer

            # Skip if not enough data yet
            if len(self.performance_tracker.iteration_records) < 2:
                logger.debug("📊 Skipping visualization - not enough data yet")
                return

            # Calculate fallback value for error codes based on valid scores
            fallback_value = self._calculate_visualization_fallback()

            # Create custom PerformanceVisualizer that uses our fallback
            visualizer = PerformanceVisualizer(os.path.join(self.output_dir, "visualizations"))

            # Monkey patch the get_visualization_safe_score function to use our fallback
            original_safe_score_func = visualizer._filter_error_scores
            def custom_filter_error_scores(scores, fallback_strategy='min_valid'):
                # Use our custom fallback instead of the default strategy
                filtered_scores = []
                error_count = 0
                valid_scores = []

                for score in scores:
                    if isinstance(score, (int, float)) and not ErrorThresholds.is_error_code(int(score)):
                        valid_scores.append(score)
                        filtered_scores.append(score)
                    else:
                        error_count += 1
                        filtered_scores.append(fallback_value)

                statistics = {
                    'valid_count': len(valid_scores),
                    'error_count': error_count,
                    'total_count': len(scores),
                    'error_rate': error_count / len(scores) if scores else 0.0
                }

                return filtered_scores, statistics

            visualizer._filter_error_scores = custom_filter_error_scores

            # Create evolution progress plots (always create these)
            linear_plot = visualizer.plot_evolution_progress(
                self.performance_tracker,
                metric_name="combined_score",
                scale="linear",
                output_dir=self.output_dir
            )

            log_plot = visualizer.plot_evolution_progress(
                self.performance_tracker,
                metric_name="combined_score",
                scale="log",
                output_dir=self.output_dir
            )

            # Create performance vs tokens plot (user requested this instead of token usage)
            perf_vs_tokens_plot = visualizer.plot_performance_vs_tokens(self.performance_tracker)

            # Also create traditional token usage plot for completeness
            token_plot = visualizer.plot_token_usage(self.performance_tracker)

            # Create detailed summary report only for final visualization
            if is_final:
                report = visualizer.create_summary_report(self.performance_tracker, output_dir=self.output_dir)
                logger.info("📊 Created final performance visualizations:")
                logger.info(f"   Linear plot: {linear_plot}")
                logger.info(f"   Log plot: {log_plot}")
                logger.info(f"   Token usage: {token_plot}")
                logger.info(f"   Summary report: {report}")
                logger.info(f"   Error code fallback value: {fallback_value:.4f}")
            else:
                # For real-time updates, just log briefly
                current_iteration = len(self.performance_tracker.iteration_records)
                best_score = self.performance_tracker.best_score
                logger.info(f"📊 Updated visualizations (iter {current_iteration}, best: {best_score:.4f})")

        except Exception as e:
            logger.warning(f"⚠️ Failed to create visualizations: {e}")
            logger.debug(f"Visualization error details: {e}", exc_info=True)
