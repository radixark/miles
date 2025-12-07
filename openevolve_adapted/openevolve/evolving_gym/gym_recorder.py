"""
Simple recorder for evolving gym that reuses controller functionality
"""
import os
import time
import json
import glob
import logging
from typing import Optional, Dict, Any
from openevolve.utils.plot_utils import scan_best_metadata_files, plot_single_run_curve

logger = logging.getLogger(__name__)


class GymRecorder:
    """Simple recorder that adapts controller functions for gym use"""

    def __init__(self, gym, output_dir: Optional[str] = None):
        """
        Initialize recorder

        Args:
            gym: SingleTaskEvolvingGym instance
            output_dir: Output directory (defaults to gym_output in current dir)
        """
        self.gym = gym
        self.output_dir = output_dir or os.path.join(os.getcwd(), "gym_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up historical records directory
        self.historical_records_dir = os.path.join(self.output_dir, "historical_records")
        os.makedirs(self.historical_records_dir, exist_ok=True)

        logger.info(f"GymRecorder initialized with output_dir: {self.output_dir}")

    def record_step(self,
                   training_step: int,
                   save_checkpoint: bool = False,
                   save_historical_records: bool = False,
                   step_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record current gym state

        Args:
            training_step: Real training step number (required)
            save_checkpoint: Whether to save checkpoint
            save_historical_records: Whether to save historical records
            step_metrics: Optional metrics for this step (e.g., batch results)
        """
        logger.info(f"Recording gym progress: training_step={training_step}")

        try:
            # Create a minimal controller-like object to reuse existing functions
            controller_adapter = self._create_controller_adapter()

            # # not very useful now
            # if save_checkpoint:
            #     self._save_checkpoint_adapted(controller_adapter, training_step)

            # Save historical records. Be careful, it would generate many files!!!
            if save_historical_records:
                self._save_historical_records_adapted(controller_adapter, training_step)

            # Always save best program
            self._save_best_program_adapted(controller_adapter, training_step)

            # Update performance visualization
            self.plot_performance_curve()

            logger.info(f"✅ Recorded step: {training_step}")

        except Exception as e:
            logger.warning(f"Recording failed for step {training_step}: {e}")


    def _create_controller_adapter(self):
        """Create minimal adapter to reuse controller functions"""
        class ControllerAdapter:
            def __init__(self, gym, output_dir):
                self.database = gym.database
                self.config = gym.config
                self.output_dir = output_dir
                self.evaluator = gym.evaluator

        return ControllerAdapter(self.gym, self.output_dir)

    def _save_historical_records_adapted(self, adapter, training_step: int):
        """Save historical records for programs including extracted_prompts and runtime_environment"""
        try:
            import shutil
            from datetime import datetime

            # Create records for new programs since last step
            # Skip initial program (generation 0) unless it's the very first step
            for program_id, program in adapter.database.programs.items():
                # Skip generation 0 programs after the first step to avoid duplicates
                if program.generation == 0 and training_step > 1:
                    continue
                # Create unique record path
                record_dirname = f"step{training_step:04d}_gen{program.generation:02d}_id{program_id[:8]}_pid{program.parent_id[:8] if program.parent_id else 'init'}"
                record_path = os.path.join(self.historical_records_dir, record_dirname)

                # Skip if already exists
                if os.path.exists(record_path):
                    continue

                os.makedirs(record_path, exist_ok=True)

                # Save program code
                program_codes_dir = os.path.join(record_path, "program_codes")
                os.makedirs(program_codes_dir, exist_ok=True)

                code_filename = f"{program_id}.py"
                with open(os.path.join(program_codes_dir, code_filename), "w") as f:
                    f.write(program.code)

                # Save program metadata
                metadata = {
                    "id": program.id,
                    "generation": program.generation,
                    "metrics": program.metrics,
                    "parent_id": program.parent_id,
                    "iteration_found": program.iteration_found,
                    "training_step": training_step,
                    "timestamp": time.time()
                }

                with open(os.path.join(record_path, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                # Extract prompts if available (from gym's prompt tracking)
                self._extract_prompts_for_record(program, record_path, training_step)

                # Save runtime environment if available and enabled
                self._save_runtime_environment(adapter, program_id, record_path)

            logger.info(f"Historical records saved for step {training_step}")

        except Exception as e:
            logger.warning(f"Historical records save failed: {e}")

    def _extract_prompts_for_record(self, program, record_path: str, training_step: int):
        """Extract prompts for debugging (adapted from controller.py)"""
        try:
            # Check if program has prompts data from multiple sources
            prompts = {}

            # First, try to get prompts from gym's prompt tracking system
            if hasattr(self.gym, 'get_program_prompts'):
                prompt_data = self.gym.get_program_prompts(program.id)

                if prompt_data and 'prompt' in prompt_data:
                    # Convert gym prompt data to controller.py format
                    gym_prompt = prompt_data['prompt']

                    if isinstance(gym_prompt, dict):
                        prompts = {
                            "evolution_prompt": {
                                "system": gym_prompt.get("system", ""),
                                "user": gym_prompt.get("user", ""),
                                "response": prompt_data.get("llm_response", "")
                            },
                            "metadata": {
                                "parent_id": prompt_data.get("parent_id", ""),
                                "evolution_round": prompt_data.get("evolution_round", 0),
                                "island": prompt_data.get("island", 0),
                                "timestamp": prompt_data.get("timestamp", 0)
                            }
                        }

            # Fallback: Try to get prompts from database's prompts_by_program structure
            if not prompts and hasattr(self.gym, 'database') and hasattr(self.gym.database, 'prompts_by_program'):
                if self.gym.database.prompts_by_program and program.id in self.gym.database.prompts_by_program:
                    db_prompts = self.gym.database.prompts_by_program[program.id]
                    if db_prompts:
                        # Convert database format to our expected format
                        for template_key, prompt_data in db_prompts.items():
                            prompts[template_key] = {
                                "system": prompt_data.get("system", ""),
                                "user": prompt_data.get("user", ""),
                                "responses": prompt_data.get("responses", [])
                            }
                        logger.debug(f"Found {len(prompts)} prompts in database for program {program.id[:8]}")

            # Fallback: Try to get prompts from artifacts_json if available
            if not prompts and hasattr(program, 'artifacts_json') and program.artifacts_json:
                try:
                    artifacts = json.loads(program.artifacts_json) if isinstance(program.artifacts_json, str) else program.artifacts_json
                    prompts = artifacts.get("prompts", {})
                except Exception as e:
                    logger.debug(f"Could not parse prompts from artifacts: {e}")

            # Fallback: Try to get prompts from metadata if available
            if not prompts and hasattr(program, 'metadata') and program.metadata:
                prompts = program.metadata.get("prompts", {})

            if not prompts:
                logger.debug(f"No prompts found for program {program.id[:8]}")
                return

            extracted_dir = os.path.join(record_path, "extracted_prompts")
            os.makedirs(extracted_dir, exist_ok=True)

            # Create program-specific directory using program ID
            program_dir = os.path.join(extracted_dir, program.id[:8])
            os.makedirs(program_dir, exist_ok=True)

            # Save program info
            info = {
                "id": program.id,
                "generation": program.generation,
                "timestamp": program.timestamp,
                "iteration_found": program.iteration_found,
                "training_step": training_step,
                "metrics": program.metrics or {},
                "parent_id": program.parent_id
            }

            info_path = os.path.join(program_dir, "program_info.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("=== GYM HISTORICAL RECORD PROGRAM INFORMATION ===\n\n")
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

            logger.debug(f"Extracted prompts for program {program.id[:8]} to {program_dir}")

        except Exception as e:
            logger.warning(f"Failed to extract prompts for program {program.id[:8]}: {e}")

    def _save_checkpoint_adapted(self, adapter, training_step: int):
        """Adapted checkpoint saving"""
        try:
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"checkpoint_step_{training_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save database state
            database_data = {
                "programs": {pid: p.to_dict() for pid, p in adapter.database.programs.items()},
                "last_iteration": adapter.database.last_iteration,
                "current_island": adapter.database.current_island,
                "training_step": training_step,
                "timestamp": time.time()
            }

            with open(os.path.join(checkpoint_dir, "database.json"), "w") as f:
                json.dump(database_data, f, indent=2)

            logger.info(f"Checkpoint saved: {checkpoint_dir}")

        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")


    def print_database_score_distribution(self):
        """Print current database score distribution for debugging"""
        try:
            programs = self.gym.database.programs
            if not programs:
                print("📊 Database is empty")
                return

            scores = []
            generations = []
            valid_programs = 0

            for program_id, program in programs.items():
                if program.metrics and 'combined_score' in program.metrics:
                    score = program.metrics['combined_score']
                    if isinstance(score, (int, float)) and score > -999:
                        scores.append(score)
                        generations.append(program.generation)
                        valid_programs += 1

            if not scores:
                print(f"📊 Database: {len(programs)} programs, but no valid scores")
                return

            print(f"\n📊 DATABASE SCORE DISTRIBUTION:")
            print(f"   Total programs: {len(programs)}")
            print(f"   Valid scores: {valid_programs}")
            print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
            print(f"   Average score: {sum(scores)/len(scores):.4f}")

            # Generation distribution
            gen_counts = {}
            for gen in generations:
                gen_counts[gen] = gen_counts.get(gen, 0) + 1

            print(f"   Generation distribution:")
            for gen in sorted(gen_counts.keys()):
                print(f"     Gen {gen}: {gen_counts[gen]} programs")

            # Score ranges
            if len(scores) >= 5:
                sorted_scores = sorted(scores, reverse=True)
                print(f"   Top 5 scores: {[f'{s:.4f}' for s in sorted_scores[:5]]}")
                print(f"   Bottom 5 scores: {[f'{s:.4f}' for s in sorted_scores[-5:]]}")

                # Percentile statistics
                n = len(sorted_scores)
                percentiles = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
                percentile_scores = []
                for p in percentiles:
                    idx = int((100 - p) / 100 * (n - 1))
                    percentile_scores.append(f"{p}%:{sorted_scores[idx]:.4f}")
                print(f"   Percentiles: {', '.join(percentile_scores)}")

            # Prompt coverage
            if hasattr(self.gym, 'database') and hasattr(self.gym.database, 'prompts_by_program'):
                if self.gym.database.prompts_by_program:
                    prompt_count = len(self.gym.database.prompts_by_program)
                    print(f"   Programs with prompts in database: {prompt_count}/{len(programs)} ({prompt_count/len(programs)*100:.1f}%)")
                else:
                    print(f"   Programs with prompts in database: 0/{len(programs)} (0.0%)")

            # New: Print historical best program information
            historical_info = self.gym.database.get_historical_best_info()
            print(f"🏆 [HISTORICAL] {historical_info}")

        except Exception as e:
            print(f"❌ Error printing score distribution: {e}")

    def _save_best_program_adapted(self, adapter, training_step: int, save_runtime_env: bool = True):
        """Save best program from current database including runtime environment"""
        try:
            best_program = adapter.database.get_best_program()
            if best_program:
                best_dir = os.path.join(self.output_dir, "best_program", f"best_step_{training_step}")
                os.makedirs(best_dir, exist_ok=True)

                # Save best program code
                filename = f"best_program_step_{training_step}.py"
                with open(os.path.join(best_dir, filename), "w") as f:
                    f.write(best_program.code)

                # Save metadata
                metadata = {
                    "id": best_program.id,
                    "generation": best_program.generation,
                    "metrics": best_program.metrics,
                    "iteration_found": best_program.iteration_found,
                    "training_step": training_step,
                    "timestamp": time.time()
                }

                with open(os.path.join(best_dir, f"best_metadata_step_{training_step}.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                # Save runtime environment for best program (default enabled)
                if save_runtime_env:
                    self._save_runtime_environment(adapter, best_program.id, best_dir, is_best=True)

                logger.info(
                    f"Best program saved at training_step={training_step} "
                    f"(program from iteration={best_program.iteration_found}, "
                    f"score={best_program.metrics.get('combined_score', 'N/A')})"
                )

        except Exception as e:
            logger.warning(f"Best program save failed: {e}")

    def _save_runtime_environment(self, adapter, program_id: str, target_dir: str, is_best: bool = False):
        """Save runtime environment to target directory (adapted from controller.py)"""
        try:
            # Check if evaluator has pending runtime environments
            if not hasattr(adapter, 'evaluator') or not adapter.evaluator:
                return

            if not hasattr(adapter.evaluator, '_pending_runtime_environments'):
                return

            # Get runtime environment for this program
            all_envs = adapter.evaluator.list_pending_runtime_environments()
            runtime_env_dir = all_envs.get(program_id)

            if not runtime_env_dir or not os.path.exists(runtime_env_dir):
                if is_best:
                    logger.debug(f"Runtime environment not found in pending, searching temp directories...")
                    import glob
                    temp_dirs = []
                    for temp_root in ['/tmp']:
                        if os.path.exists(temp_root):
                            try:
                                pattern = os.path.join(temp_root, f"runtime_env_{program_id}_*")
                                matches = glob.glob(pattern)
                                temp_dirs.extend(matches)
                            except Exception as e:
                                logger.debug(f"Error searching temp dirs: {e}")

                    if temp_dirs:
                        runtime_env_dir = max(temp_dirs, key=os.path.getmtime)
                        logger.debug(f"Using found runtime environment: {runtime_env_dir}")
                    else:
                        logger.debug(f"No runtime environment found for program {program_id[:8]}")
                        return
                else:
                    logger.debug(f"No runtime environment found for program {program_id[:8]}")
                    return

            # Create runtime environment directory in target folder
            target_runtime_dir = os.path.join(target_dir, "runtime_environment")

            logger.debug(f"Copying runtime environment from {runtime_env_dir} to {target_runtime_dir}")

            # Copy runtime environment
            import shutil
            shutil.copytree(runtime_env_dir, target_runtime_dir, dirs_exist_ok=True)

            # Count files for logging
            file_count = sum(len(files) for _, _, files in os.walk(target_runtime_dir))

            if is_best:
                logger.info(f"Saved runtime environment for program {program_id[:8]}: {file_count} files")
            else:
                logger.debug(f"Saved runtime environment for program {program_id[:8]}: {file_count} files")

        except Exception as e:
            logger.warning(f"Failed to save runtime environment for best program {program_id[:8]}: {e}")

    def plot_performance_curve(self):
        """
        Plot best score progression curve from saved metadata files

        Reads all best_metadata_step_*.json files and creates a simple
        performance visualization showing score vs training_step.
        """
        # Scan and cache data
        data_points = scan_best_metadata_files(self.output_dir, save_cache=True)

        if len(data_points) < 2:
            logger.info(f"Skipping visualization - only {len(data_points)} data points found")
            return

        # Plot curve
        viz_dir = os.path.join(self.output_dir, "visualizations")
        output_path = os.path.join(viz_dir, "performance_curve.jpg")

        plot_single_run_curve(data_points, output_path)

        steps = [step for step, _ in data_points]
        logger.info(f"Performance curve saved to: {output_path}")
        logger.info(f"Plotted {len(data_points)} data points from step {min(steps)} to {max(steps)}")