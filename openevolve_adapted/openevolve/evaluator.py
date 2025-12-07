"""
Evaluation system for OpenEvolve
"""

import asyncio
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import traceback

from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.database import ProgramDatabase
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.utils.async_utils import TaskPool, run_in_executor
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.format_utils import format_metrics_safe

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates programs and assigns scores

    The evaluator is responsible for executing programs, measuring their performance,
    and assigning scores based on the evaluation criteria.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        llm_ensemble: Optional[LLMEnsemble] = None,
        prompt_sampler: Optional[PromptSampler] = None,
        database: Optional[ProgramDatabase] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database

        # Create a task pool for parallel evaluation
        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)

        # Set up evaluation function if file exists
        self._load_evaluation_function()

        # Pending artifacts storage for programs
        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}
        
        # Pending runtime environments storage for programs
        self._pending_runtime_environments: Dict[str, str] = {}

        logger.info(f"Initialized evaluator with {evaluation_file}")

    def _copy_evaluation_files(self, temp_dir: str) -> None:
        """
        Copy specified folders and files to the temporary directory for evaluation

        Args:
            temp_dir: Path to the temporary directory
        """

        if not self.config.copy_folders and not self.config.copy_files:
            # print(f"[VERBOSE-COPY] No copy_folders or copy_files configured, skipping file copying")
            logger.debug("No copy_folders or copy_files configured, skipping file copying")
            return
            
        # Get the evaluation file directory as the base directory
        eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
        logger.info(f"Debug: Copying folders and files for evaluation")
        logger.info(f"Debug: Evaluation directory: {eval_dir}")
        logger.info(f"Debug: Target temp directory: {temp_dir}")
        
        # List what's in the evaluation directory
        logger.info(f"Debug: Contents of evaluation directory:")
        try:
            for item in os.listdir(eval_dir):
                item_path = os.path.join(eval_dir, item)
                if os.path.isdir(item_path):
                    logger.info(f"Debug:   [DIR]  {item}/")
                else:
                    logger.info(f"Debug:   [FILE] {item}")
        except Exception as e:
            logger.warning(f"Debug: Failed to list evaluation directory: {e}")
        
        # Copy folders
        for folder in self.config.copy_folders:
            # Resolve source path: support both relative (../initial_programs) and sibling (initial_programs) paths
            if folder.startswith("../"):
                # For paths like "../initial_programs", resolve relative to evaluators/ parent
                source_path = os.path.normpath(os.path.join(eval_dir, folder))
            else:
                # For simple names like "initial_programs", check if sibling to evaluators/
                candidate_path = os.path.join(eval_dir, folder)
                sibling_path = os.path.normpath(os.path.join(eval_dir, "..", folder))
                source_path = candidate_path if os.path.exists(candidate_path) else sibling_path

            # Always use just the folder basename for target (avoids ../.. in temp_dir path)
            folder_basename = os.path.basename(folder)
            target_path = os.path.join(temp_dir, folder_basename)

            logger.info(f"Debug: Attempting to copy folder '{folder}'")
            logger.info(f"Debug:   Source: {source_path}")
            logger.info(f"Debug:   Target: {target_path}")
            
            if os.path.exists(source_path) and os.path.isdir(source_path):
                try:
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                    logger.info(f"Debug: ✅ Successfully copied folder {folder}")
                    
                    # List contents of copied folder
                    try:
                        contents = os.listdir(target_path)
                        logger.info(f"Debug: Contents of copied {folder}: {contents}")
                        
                        # For files folder, look deeper
                        if folder == "files":
                            for item in contents:
                                item_path = os.path.join(target_path, item)
                                if os.path.isdir(item_path):
                                    sub_contents = os.listdir(item_path)
                                    logger.info(f"Debug:   {item}/: {sub_contents}")
                    except Exception as e:
                        logger.warning(f"Debug: Failed to list copied folder contents: {e}")
                        
                except Exception as e:
                    logger.warning(f"Debug: ❌ Failed to copy folder {folder}: {e}")
            else:
                logger.warning(f"Debug: ❌ Folder {folder} not found at {source_path}")
        
        # Copy individual files
        for file in self.config.copy_files:
            source_path = os.path.join(eval_dir, file)
            target_path = os.path.join(temp_dir, file)
            
            logger.info(f"Debug: Attempting to copy file '{file}'")
            logger.info(f"Debug:   Source: {source_path}")
            logger.info(f"Debug:   Target: {target_path}")
            
            if os.path.exists(source_path) and os.path.isfile(source_path):
                try:
                    # Create target directory if it doesn't exist
                    target_dir = os.path.dirname(target_path)
                    if target_dir:
                        os.makedirs(target_dir, exist_ok=True)
                    
                    shutil.copy2(source_path, target_path)
                    logger.info(f"Debug: ✅ Successfully copied file {file}")
                except Exception as e:
                    logger.warning(f"Debug: ❌ Failed to copy file {file}: {e}")
            else:
                logger.warning(f"Debug: ❌ File {file} not found at {source_path}")
        
        # # Final verification - list everything in temp_dir
        # logger.info(f"Debug: Final verification - contents of temp directory {temp_dir}:")
        # try:
        #     for root, dirs, files in os.walk(temp_dir):
        #         level = root.replace(temp_dir, '').count(os.sep)
        #         indent = '  ' * level
        #         logger.info(f"Debug: {indent}{os.path.basename(root)}/")
        #         subindent = '  ' * (level + 1)
        #         for file in files:
        #             logger.info(f"Debug: {subindent}{file}")
        # except Exception as e:
        #     logger.warning(f"Debug: Failed to walk temp directory: {e}")

    def _load_evaluation_function(self) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} does not contain an 'evaluate' function"
                )

            self.evaluate_function = module.evaluate
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")

            # Validate cascade configuration
            self._validate_cascade_configuration(module)
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

    def _validate_cascade_configuration(self, module) -> None:
        """
        Validate cascade evaluation configuration and warn about potential issues

        Args:
            module: The loaded evaluation module
        """
        if self.config.cascade_evaluation:
            # Check if cascade functions exist
            has_stage1 = hasattr(module, "evaluate_stage1")
            has_stage2 = hasattr(module, "evaluate_stage2")
            has_stage3 = hasattr(module, "evaluate_stage3")

            if not has_stage1:
                logger.warning(
                    f"Configuration has 'cascade_evaluation: true' but evaluator "
                    f"'{self.evaluation_file}' does not define 'evaluate_stage1' function. "
                    f"This will fall back to direct evaluation, making the cascade setting useless. "
                    f"Consider setting 'cascade_evaluation: false' or implementing cascade functions."
                )
            elif not (has_stage2 or has_stage3):
                logger.warning(
                    f"Evaluator '{self.evaluation_file}' defines 'evaluate_stage1' but no additional "
                    f"cascade stages (evaluate_stage2, evaluate_stage3). Consider implementing "
                    f"multi-stage evaluation for better cascade benefits."
                )
            else:
                logger.debug(
                    f"Cascade evaluation properly configured with available stage functions"
                )

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate a program and return scores

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"

        # Retry logic for evaluation
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            # Create a temporary directory for the program and supporting files
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "program.py")
            
            try:
                # Write the program code to the temporary file
                with open(temp_file_path, 'w') as temp_file:
                    temp_file.write(program_code)
                
                # Copy evaluation files if configured
                self._copy_evaluation_files(temp_dir)

                # Run evaluation
                if self.config.cascade_evaluation:
                    # Run cascade evaluation
                    result = await self._cascade_evaluate(temp_file_path)
                else:
                    # Run direct evaluation
                    result = await self._direct_evaluate(temp_file_path)

                # Process the result based on type
                eval_result = self._process_evaluation_result(result)

                # Check if this was a timeout and capture artifacts if enabled
                if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    self._pending_artifacts[program_id].update(
                        {
                            "timeout": True,
                            "timeout_duration": self.config.timeout,
                            "failure_stage": "evaluation",
                            "error_type": "timeout",
                        }
                    )

                # Add LLM feedback if configured
                llm_eval_result = None
                if self.config.use_llm_feedback and self.llm_ensemble:
                    llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                    llm_eval_result = self._process_evaluation_result(llm_result)

                    # Combine metrics
                    for name, value in llm_result.metrics.items():
                        eval_result.metrics[f"llm_{name}"] = value * self.config.llm_feedback_weight

                # Store artifacts if enabled and present
                if (
                    artifacts_enabled
                    and (
                        eval_result.has_artifacts()
                        or (llm_eval_result and llm_eval_result.has_artifacts())
                    )
                    and program_id
                ):
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    # Merge eval_result artifacts with llm artifacts if they exist
                    if eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned artifacts: "
                            f"{eval_result.artifacts}"
                        )

                    if llm_eval_result and llm_eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(llm_eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned LLM artifacts: "
                            f"{llm_eval_result.artifacts}"
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )

                # Log final state of pending runtime environments after evaluation
                if program_id:
                    current_pending = list(self._pending_runtime_environments.keys())
                    logger.debug(f"🔚 Final pending runtime environments after evaluating {program_id}: {current_pending}")

                # Return just metrics for backward compatibility
                return eval_result.metrics

            except asyncio.TimeoutError:
                # Handle timeout specially - don't retry, just return timeout result
                logger.warning(f"Evaluation timed out after {self.config.timeout}s")

                # Capture timeout artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

                return {"error": 0.0, "timeout": True}

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {str(e)}"
                )
                traceback.print_exc()

                # Capture failure artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": attempt + 1,
                    }

                # If this is not the last attempt, wait a bit before retrying
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)  # Wait 1 second before retry

            finally:
                # Collect runtime environment for all programs (successful or failed)
                if program_id:
                    self._collect_runtime_environment(temp_dir, program_id)
                
                # Clean up temporary directory (unless preserve_temp_directories is enabled)
                if os.path.exists(temp_dir):
                    if self.config.preserve_temp_directories:
                        logger.info(f"Preserving temporary directory for debugging: {temp_dir}")
                    else:
                        shutil.rmtree(temp_dir)

        # All retries failed
        logger.error(
            f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        )
        return {"error": 0.0}

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        else:
            # Error case - return error metrics
            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Get and clear pending artifacts for a program

        Args:
            program_id: Program ID

        Returns:
            Artifacts dictionary or None if not found
        """
        return self._pending_artifacts.pop(program_id, None)

    async def _direct_evaluate(
        self, program_path: str
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Directly evaluate a program using the evaluation function with timeout

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            Exception: If evaluation function raises an exception
        """

        # Get the temporary directory from the program path
        temp_dir = os.path.dirname(program_path)

        # Create a coroutine that runs the evaluation function in an executor
        async def run_evaluation():
            loop = asyncio.get_event_loop()
            # Check if the evaluation function accepts temp_dir parameter
            import inspect
            sig = inspect.signature(self.evaluate_function)
            if "temp_dir" in sig.parameters:
                return await loop.run_in_executor(None, self.evaluate_function, program_path, temp_dir)
            else:
                return await loop.run_in_executor(None, self.evaluate_function, program_path)

        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        result = await asyncio.wait_for(run_evaluation(), timeout=self.config.timeout)

        # Return result as-is to be processed by _process_evaluation_result
        # This supports both dict and EvaluationResult returns, just like _cascade_evaluate
        return result

    async def _cascade_evaluate(
        self, program_path: str
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Run cascade evaluation with increasingly challenging test cases

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts
        """
        # Import the evaluation module to get cascade functions if they exist
        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for cascade evaluation")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                return await self._direct_evaluate(program_path)

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if cascade functions exist
            if not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(program_path)

            # Get the temporary directory from the program path
            temp_dir = os.path.dirname(program_path)

            # Run first stage with timeout
            try:

                async def run_stage1():
                    loop = asyncio.get_event_loop()
                    # Check if the evaluation function accepts temp_dir parameter
                    import inspect
                    sig = inspect.signature(module.evaluate_stage1)
                    if temp_dir in sig.parameters:
                        return await loop.run_in_executor(None, module.evaluate_stage1, program_path, temp_dir)
                    else:
                        return await loop.run_in_executor(None, module.evaluate_stage1, program_path)

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=self.config.timeout)
                stage1_eval_result = self._process_evaluation_result(stage1_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 1 evaluation timed out after {self.config.timeout}s")
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                    artifacts={
                        "failure_stage": "stage1",
                        "timeout": True,
                    },
                )
            except Exception as e:
                logger.error(f"Error in stage 1 evaluation: {str(e)}")
                # Capture stage 1 failure with enhanced context
                error_context = self._create_cascade_error_context("stage1", e)
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0},
                    artifacts={
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        **error_context,
                    },
                )

            # Check threshold
            if not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result

            # Check if second stage exists
            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result

            # Run second stage with timeout
            try:

                async def run_stage2():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage2, program_path)

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=self.config.timeout)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                stage1_eval_result.metrics["timeout"] = True
                return stage1_eval_result
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {str(e)}")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                return stage1_eval_result

            # Merge results from stage 1 and 2
            merged_metrics = {}
            # Convert all values to float to avoid type errors
            for name, value in stage1_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            for name, value in stage2_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            # Merge artifacts
            merged_artifacts = {}
            merged_artifacts.update(stage1_eval_result.artifacts)
            merged_artifacts.update(stage2_eval_result.artifacts)

            merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)

            # Check threshold for stage 3
            if len(self.config.cascade_thresholds) < 2 or not self._passes_threshold(
                merged_result.metrics, self.config.cascade_thresholds[1]
            ):
                return merged_result

            # Check if third stage exists
            if not hasattr(module, "evaluate_stage3"):
                return merged_result

            # Run third stage with timeout
            try:

                async def run_stage3():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage3, program_path)

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=self.config.timeout)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                merged_result.metrics["timeout"] = True
                return merged_result
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {str(e)}")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                return merged_result

            # Merge stage 3 results
            for name, value in stage3_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_result.metrics[name] = float(value)

            merged_result.artifacts.update(stage3_eval_result.artifacts)

            return merged_result

        except Exception as e:
            logger.error(f"Error in cascade evaluation: {str(e)}")
            # Return proper cascade failure result with enhanced context
            error_context = self._create_cascade_error_context("cascade_setup", e)
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )

    async def _llm_evaluate(self, program_code: str, program_id: str = "") -> Dict[str, float]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        if not self.llm_ensemble:
            return {}

        try:
            # Create prompt for LLM
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )

            # Get LLM response
            responses = await self.llm_ensemble.generate_all_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )

            # Log prompt and response to database
            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluation",
                    prompt=prompt,
                    responses=responses,
                )

            # Extract JSON from response
            try:
                # Try to find JSON block
                json_pattern = r"```json\n(.*?)\n```"
                import re

                artifacts = {}
                avg_metrics = {}
                for i, response in enumerate(responses):
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to extract JSON directly
                        json_str = response
                        # Remove non-JSON parts
                        start_idx = json_str.find("{")
                        end_idx = json_str.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]

                    # Parse JSON
                    result = json.loads(json_str)

                    # All non-numeric values are artifacts, all numeric values are metrics
                    metrics = {}
                    for key, value in result.items():
                        if not isinstance(value, (int, float)):
                            artifacts[key] = value
                        else:
                            metrics[key] = float(value)

                    # Weight of the model in the ensemble
                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0

                    # Average the metrics
                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight

                return EvaluationResult(
                    metrics=avg_metrics,
                    artifacts=artifacts,
                )

            except Exception as e:
                logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            traceback.print_exc()
            return {}

    def _create_cascade_error_context(self, stage: str, error: Exception) -> dict:
        """
        Create rich error context for cascade failures

        Args:
            stage: The stage where the error occurred
            error: The exception that was raised

        Returns:
            Dictionary with enhanced error context
        """
        import time

        return {
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "cascade_config": self.config.cascade_evaluation,
            "cascade_thresholds": getattr(self.config, "cascade_thresholds", []),
            "timeout_config": self.config.timeout,
            "evaluation_file": self.evaluation_file,
        }

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:
        """
        Check if metrics pass a threshold

        Args:
            metrics: Dictionary of metric name to score
            threshold: Threshold to pass

        Returns:
            True if metrics pass threshold
        """
        if not metrics:
            return False

        # Calculate average score, skipping non-numeric values and 'error' key
        valid_metrics = []
        for name, value in metrics.items():
            # Skip 'error' keys and ensure values are numeric
            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid_metrics.append(float(value))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue

        if not valid_metrics:
            return False

        avg_score = sum(valid_metrics) / len(valid_metrics)
        return avg_score >= threshold

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple programs in parallel

        Args:
            programs: List of (program_code, program_id) tuples

        Returns:
            List of metric dictionaries
        """
        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)

    def _collect_runtime_environment(self, temp_dir: str, program_id: str) -> None:
        """
        Collect runtime environment files from temporary directory
        
        Args:
            temp_dir: Temporary directory path
            program_id: Program ID for identification
        """
        if not self.config.collect_runtime_environments:
            logger.debug(f"Runtime environment collection disabled for program {program_id}")
            return
            
        if not program_id:
            logger.warning("No program_id provided for runtime environment collection")
            return
            
        logger.debug(f"🔄 Starting runtime environment collection for program {program_id} from {temp_dir}")

        # List what's available in temp_dir first
        try:
            temp_contents = os.listdir(temp_dir) if os.path.exists(temp_dir) else []
            logger.debug(f"📁 Temp directory contents: {temp_contents}")
        except Exception as e:
            logger.warning(f"Failed to list temp directory contents: {e}")
        
        # Create a unique directory for this program's runtime environment
        runtime_env_dir = tempfile.mkdtemp(prefix=f"runtime_env_{program_id}_")
        logger.debug(f"📂 Created runtime environment directory: {runtime_env_dir}")

        try:
            collected_files = 0

            # Collect files based on configured patterns
            logger.debug(f"🔍 Searching for patterns: {self.config.runtime_environment_patterns}")
            for pattern in self.config.runtime_environment_patterns:
                pattern_path = os.path.join(temp_dir, pattern)
                logger.debug(f"  Checking pattern: {pattern_path}")
                matched_paths = glob.glob(pattern_path, recursive=True)
                logger.debug(f"  Found {len(matched_paths)} matches: {matched_paths}")
                
                for matched_path in matched_paths:
                    if os.path.exists(matched_path):
                        # Calculate relative path from temp_dir
                        rel_path = os.path.relpath(matched_path, temp_dir)
                        target_path = os.path.join(runtime_env_dir, rel_path)
                        
                        # Create target directory if needed
                        target_parent = os.path.dirname(target_path)
                        if target_parent:
                            os.makedirs(target_parent, exist_ok=True)
                        
                        try:
                            if os.path.isfile(matched_path):
                                shutil.copy2(matched_path, target_path)
                                collected_files += 1
                                logger.debug(f"  ✅ Collected file: {rel_path}")
                            elif os.path.isdir(matched_path):
                                shutil.copytree(matched_path, target_path, dirs_exist_ok=True)
                                # Count files in the directory
                                for root, dirs, files in os.walk(target_path):
                                    collected_files += len(files)
                                logger.debug(f"  ✅ Collected directory: {rel_path}")
                        except Exception as e:
                            logger.warning(f"  ❌ Failed to collect {rel_path}: {e}")
            
            if collected_files > 0:
                # Ensure thread-safe addition to pending runtime environments
                self._pending_runtime_environments[program_id] = runtime_env_dir
                logger.debug(f"✅ Collected runtime environment for program {program_id}: {collected_files} files in {runtime_env_dir}")
                logger.debug(f"🔗 Added to pending environments. Current count: {len(self._pending_runtime_environments)}")
                logger.debug(f"🗂️ All pending IDs: {list(self._pending_runtime_environments.keys())}")
                
                # List what we collected
                try:
                    collected_contents = []
                    for root, dirs, files in os.walk(runtime_env_dir):
                        for d in dirs[:3]:  # First 3 dirs
                            rel_path = os.path.relpath(os.path.join(root, d), runtime_env_dir)
                            collected_contents.append(f"[DIR] {rel_path}/")
                        for f in files[:3]:  # First 3 files
                            rel_path = os.path.relpath(os.path.join(root, f), runtime_env_dir)
                            collected_contents.append(f"[FILE] {rel_path}")
                    logger.debug(f"📋 Sample collected items: {collected_contents}")
                except:
                    pass
            else:
                # No files collected, clean up the empty directory
                shutil.rmtree(runtime_env_dir, ignore_errors=True)
                logger.warning(f"❌ No runtime environment files found for program {program_id}")
                logger.info(f"🔍 Available patterns were: {self.config.runtime_environment_patterns}")
                logger.info(f"🔍 Temp directory was: {temp_dir}")
                
        except Exception as e:
            logger.error(f"💥 Error collecting runtime environment for program {program_id}: {e}")
            import traceback
            logger.error(f"💥 Error details: {traceback.format_exc()}")
            # Clean up on error
            if os.path.exists(runtime_env_dir):
                shutil.rmtree(runtime_env_dir, ignore_errors=True)

    def get_pending_runtime_environment(self, program_id: str, clear: bool = False) -> Optional[str]:
        """
        Get pending runtime environment for a program
        
        Args:
            program_id: Program ID
            clear: Whether to clear the entry (default False - changed to preserve environments)
            
        Returns:
            Runtime environment directory path or None if not found
        """
        if clear:
            return self._pending_runtime_environments.pop(program_id, None)
        else:
            return self._pending_runtime_environments.get(program_id, None)

    def cleanup_runtime_environments(self) -> None:
        """
        Clean up all pending runtime environment directories
        """
        logger.info(f"Cleaning up {len(self._pending_runtime_environments)} pending runtime environments")
        
        for program_id, runtime_env_dir in self._pending_runtime_environments.items():
            if os.path.exists(runtime_env_dir):
                try:
                    if not self.config.preserve_temp_directories:
                        shutil.rmtree(runtime_env_dir)
                        logger.debug(f"Cleaned up runtime environment for program {program_id}")
                    else:
                        logger.info(f"Preserving runtime environment for program {program_id}: {runtime_env_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up runtime environment for program {program_id}: {e}")
        
        if not self.config.preserve_temp_directories:
            self._pending_runtime_environments.clear()
        else:
            logger.info("Preserving all runtime environment references for debugging")

    def list_pending_runtime_environments(self) -> Dict[str, str]:
        """
        List all pending runtime environments for debugging
        
        Returns:
            Dictionary mapping program IDs to runtime environment paths
        """
        return dict(self._pending_runtime_environments)
