import asyncio
import uuid
import time
import logging
from typing import Dict, Tuple, Optional

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.prompt.sampler import PromptSampler
from openevolve.evaluator import Evaluator
from openevolve.iteration import Result
from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    extract_code_language,
    check_code_identical
)
from openevolve.modular_utils.error_constants import ErrorCodes, ErrorMessages
from openevolve.modular_utils.score_transform import create_score_transform_config_from_dict
from openevolve.utils.metrics_utils import create_evaluation_metrics
from openevolve.utils.performance_utils import timed_async_operation, timed_operation

logger = logging.getLogger(__name__)


class SingleTaskEvolvingGym:
    """
    Evolving gym for single-task RL-style batching.

    Key features
    - Uses asyncio.Semaphore to limit concurrent evaluations;
    - Added concurrent lock and TTL/capacity limits for prompt debug cache to avoid race conditions and memory growth.
    - Evaluation supports timeout (read from config, default 300s).
    - Result return type changed to Optional[Result]
    """

    # Prompt debug cache strategy
    _TEMP_PROMPT_TTL_SEC = 300          # Temporary keys live for 5 minutes
    _PROMPT_CACHE_MAX = 5000            # Max 5000 entries (including temporary + permanent items)

    def __init__(self,
                 initial_program_path: str,
                 evaluation_file: str,
                 config_path: Optional[str] = None,
                 config: Optional[Config] = None,
                 max_concurrent_evaluations: int = 4,
                 log_prompts: bool = True,
                 lazy_output_penalty_level: int = 2,
                 database_reinit_ratio: float = 0.0,
                 smallest_restart_step: int = 0,
                 largest_restart_step: int = None,
                 add_historical_programs: int = 0,
                 reward_process_type: str = "original_reward",
                 seed: Optional[int] = None):
        """
        Initialize gym (no I/O or async evaluation; trigger initial evaluation via initialize/create)

        Args:
            initial_program_path: Path to initial program file
            evaluation_file: Path to evaluator file
            config_path: Optional config file path
            config: Optional config object
            max_concurrent_evaluations: Maximum concurrent evaluations allowed
            log_prompts: Whether to log prompt/debug info
            lazy_output_penalty_level: Lazy output penalty level (0=no penalty, 1=check parent, 2=check database)
            database_reinit_ratio: Database reinit ratio (0.0 disables reinit)
            smallest_restart_step: Minimum steps between reinitializations
            largest_restart_step: Maximum steps between reinitializations (None for infinity)
            add_historical_programs: Number of historical programs to add after reinit
            reward_process_type: Reward processing type (original_reward, rl_normalized_reward, format_reward, validation_reward, improve_reward)
            seed: Random seed for reproducibility (overrides config.random_seed if provided)
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Seed parameter takes priority over config.random_seed
        if seed is not None:
            self.config.random_seed = seed
            print(f"[SingleTaskEvolvingGym] Override config random_seed with seed={seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        with open(initial_program_path, 'r') as f:
            self.initial_program_code = f.read()
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code) or "python"

        # Initialize components
        self.prompt_sampler = PromptSampler(self.config.prompt)

        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed
        self.database = ProgramDatabase(self.config.database)

        # Pass reinit_ratio to database
        self.database.reinit_ratio = database_reinit_ratio

        # Set reinitialization step control
        self.database.smallest_restart_step = smallest_restart_step
        self.database.largest_restart_step = largest_restart_step if largest_restart_step is not None else float('inf')
        self.add_historical_programs = add_historical_programs

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            database=self.database,
        )

        # Concurrency control
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self._evaluation_semaphore = asyncio.Semaphore(max_concurrent_evaluations)

        # Lazy output penalty configuration
        self.lazy_output_penalty_level = lazy_output_penalty_level

        # Reward processing configuration
        self.reward_process_type = reward_process_type

        # Prompt debug data
        self.log_prompts = log_prompts
        self._program_prompts: Optional[Dict[str, dict]] = {} if log_prompts else None
        self._prompts_lock = asyncio.Lock()  # Protect concurrent writes to _program_prompts

        output_str = (
            f"Initial program loaded from {initial_program_path}, "
            f"language={self.config.language}, "
            f"max_concurrent_evaluations={max_concurrent_evaluations}, "
            f"log_prompts={log_prompts}, "
            f"lazy_output_penalty_level={lazy_output_penalty_level}, "
            f"random_seed={self.config.random_seed}"
        )
        # logger.info(output_str)
        print(output_str)

        # Initial evaluation state
        self._initial_program_id = str(uuid.uuid4())
        self._initialized = False 

        # Optional recorder
        self._recorder = None

        # Initialize score_transform_config for consistent metrics creation
        self.score_transform_config = create_score_transform_config_from_dict(self.config.to_dict())
        print(f"Initialized score_transform_config: {self.score_transform_config.__dict__ if self.score_transform_config else None}")

    # ---------------------------
    # Initialization paths (initial evaluation runs only once)
    # ---------------------------

    @timed_async_operation("Gym Initialize")
    async def initialize(self) -> None:
        """
        Async initialization: perform one-time evaluation of initial program and add to database.
        """
        if self._initialized:
            return

        logger.info("Evaluating initial program during initialization (async)")
        initial_metrics = await self._evaluate_initial_program_async()

        initial_program = Program(
            id=self._initial_program_id,
            code=self.initial_program_code,
            language=self.config.language,
            metrics=initial_metrics,
            parent_id=None,
            generation=0,
            iteration_found=0
        )
        self.database.add(initial_program, iteration=0)
        # Create initial snapshot for lazy penalty level 2
        self.database.init_old_code_hashes()
        print(
            f"[InitialProgram] Added to database with ID: {self._initial_program_id}, metrics: {initial_metrics}"
        )
        # assert False, "debug stop here"
        self._initialized = True

    def initialize_sync(self) -> None:
        """
        Synchronous initialization: one-time initial evaluation entry point for non-async environments.
        - If current thread already has a running loop, raises RuntimeError to avoid nested event loops.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            asyncio.run(self.initialize())
        else:
            # When event loop already exists, synchronous wait causes nesting/deadlock, explicitly error
            raise RuntimeError(
                "initialize_sync() cannot be called when an event loop is running. "
                "Use `await gym.initialize()` instead."
            )

    async def _evaluate_initial_program_async(self) -> Dict:
        """Async helper for initial program evaluation"""
        try:
            # Evaluation timeout (seconds), default 1000
            timeout_s = getattr(self.config.evaluator, "timeout_s", None) or getattr(self.config.evaluator, "timeout", None) or 1000
            return await asyncio.wait_for(
                self.evaluator.evaluate_program(self.initial_program_code, self._initial_program_id),
                timeout=timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning("Initial program evaluation timed out")
            return {}
        except Exception as e:
            logger.warning(f"Initial program evaluation failed: {e}")
            return {}

    # ---------------------------
    # Prompt recording/association (concurrency-safe)
    # ---------------------------

    async def _associate_prompt_with_program(self, program_id: str, parent_id: str, response: str):
        """Associate temporary prompt data with real program_id (concurrency-safe)"""
        if not self.log_prompts or self._program_prompts is None:
            return

        try:
            async with self._prompts_lock:
                matching_temp_keys = [
                    k for k in self._program_prompts.keys()
                    if k.startswith(f"temp_{parent_id}_")
                ]

                if matching_temp_keys:
                    temp_key = max(
                        matching_temp_keys,
                        key=lambda k: self._program_prompts[k]["timestamp"]
                    )
                    prompt_data = self._program_prompts[temp_key].copy()
                    prompt_data["llm_response"] = response
                    prompt_data["program_id"] = program_id
                    # Real program_id also records timestamp for easy cleanup
                    prompt_data.setdefault("timestamp", time.time())

                    self._program_prompts[program_id] = prompt_data
                    logger.debug(
                        f"Associated prompt data with program {program_id[:8]} from parent {parent_id[:8]}"
                    )
                else:
                    logger.debug(f"No prompt data found for parent {parent_id[:8]}")

                self._cleanup_prompt_cache_locked()

        except Exception as e:
            logger.warning(f"Failed to associate prompt with program {program_id[:8]}: {e}")

    async def _log_prompt_to_database(self, program_id: str, parent_id: str):
        """Persist prompt data to database checkpoint (concurrency-safe)"""
        if not self.log_prompts or self._program_prompts is None:
            return

        try:
            async with self._prompts_lock:
                prompt_data = self._program_prompts.get(program_id, {})
                if prompt_data and 'prompt' in prompt_data:
                    prompt = prompt_data['prompt']
                    response = prompt_data.get('llm_response', '')

                    database_prompt = {
                        'system': prompt.get('system', ''),
                        'user': prompt.get('user', ''),
                        'parent_id': parent_id,
                        'evolution_round': prompt_data.get('evolution_round', 0),
                        'island': prompt_data.get('island', 0),
                        'timestamp': prompt_data.get('timestamp', 0)
                    }

                    self.database.log_prompt(
                        program_id=program_id,
                        template_key='evolution_prompt',
                        prompt=database_prompt,
                        responses=[response] if response else []
                    )

                    logger.debug(f"Logged prompt to database for program {program_id[:8]}")
                else:
                    logger.debug(f"No prompt data to log for program {program_id[:8]}")

        except Exception as e:
            logger.warning(f"Failed to log prompt to database for {program_id[:8]}: {e}")

    def _cleanup_prompt_cache_locked(self):
        """
        Only call when self._prompts_lock is already held.
        - Clean up expired temporary keys
        - Trim by timestamp when total exceeds limit
        """
        if not self.log_prompts or self._program_prompts is None:
            return

        now = time.time()
        # 1) Clean up expired temporary keys
        old_temp_keys = [
            k for k, v in self._program_prompts.items()
            if k.startswith("temp_") and (now - v.get("timestamp", now)) > self._TEMP_PROMPT_TTL_SEC
        ]
        for k in old_temp_keys:
            self._program_prompts.pop(k, None)

        # 2) Total capacity trimming
        if len(self._program_prompts) > self._PROMPT_CACHE_MAX:
            # Sort by timestamp from old to new, delete first N items
            items = [
                (k, v.get("timestamp", 0.0)) for k, v in self._program_prompts.items()
            ]
            items.sort(key=lambda x: x[1])  # Oldest first
            to_delete = len(self._program_prompts) - self._PROMPT_CACHE_MAX
            for i in range(to_delete):
                self._program_prompts.pop(items[i][0], None)

            logger.debug(f"Trimmed prompt cache by {to_delete} items")

    def get_program_prompts(self, program_id: str) -> dict:
        """Get prompt debug data for a program (snapshot)"""
        if not self.log_prompts or self._program_prompts is None:
            return {}
        # No forced locking here, just read snapshot
        return self._program_prompts.get(program_id, {}).copy()

    # ---------------------------
    # Sampling/scoring (logic consistent with iteration.py)
    # ---------------------------


    # @timed_operation("Problem Generation")
    def problem_generator(self) -> Tuple[Dict[str, str], Program]:
        """
        Generate problem - reference iteration.py:52-92

        Returns:
            (prompt_dict, parent_program)
        """
        # Island sampling
        parent, inspirations = self.database.sample(self.config.prompt.num_inspiration_programs)
        parent_artifacts = self.database.get_artifacts(parent.id)
        parent_island = parent.metadata.get("island", self.database.current_island)
        island_top_programs = self.database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = self.database.get_top_programs(3, island_idx=parent_island)

        def enhance_programs_with_artifacts(programs, include_artifacts):
            # [TODO]: our current code still don't include artifacts in prompt for efficiency (artifacts are not stored). Can try to add it later.
            if not include_artifacts:
                return [p.to_dict() for p in programs]
            enhanced = []
            for p in programs:
                program_dict = p.to_dict()
                artifacts = self.database.get_artifacts(p.id)
                if artifacts:
                    program_dict['artifacts'] = artifacts
                enhanced.append(program_dict)
            return enhanced

        include_top_artifacts = (
            self.config.prompt.include_artifacts_for_all_programs
            or self.config.prompt.include_top_program_artifacts
        )
        include_inspiration_artifacts = (
            self.config.prompt.include_artifacts_for_all_programs
            or self.config.prompt.include_inspiration_artifacts
        )

        prompt = self.prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=enhance_programs_with_artifacts(island_previous_programs, include_top_artifacts),
            top_programs=enhance_programs_with_artifacts(island_top_programs, include_top_artifacts),
            inspirations=enhance_programs_with_artifacts(inspirations, include_inspiration_artifacts),
            language=self.config.language,
            evolution_round=len(self.database.programs),
            diff_based_evolution=self.config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
        )

        # Store debug prompt (temporary key)
        if self.log_prompts and self._program_prompts is not None:
            prompt_data = {
                "parent_id": parent.id,
                "timestamp": time.time(),
                "prompt": prompt,
                "parent_program": parent.code,
                "parent_metrics": parent.metrics,
                "evolution_round": len(self.database.programs),
                "island": parent_island,
            }
            temp_key = f"temp_{parent.id}_{int(time.time() * 1000000)}"
            # Outside async context, avoid await; brief insert here, lock-free write risk is small (exclusive thread)
            self._program_prompts[temp_key] = prompt_data

            # Cleanup (lock-free fast path, real ordered cleanup happens when writing with lock)
            self._cleanup_prompt_cache_locked()

        return prompt, parent

    # @timed_async_operation("Response Scoring")
    async def response_scorer(self, response: str, parent_program: Program) -> Optional[Result]:
        """
        Score response with concurrency control - reference iteration.py:94-165

        Args:
            response: LLM response containing code modifications
            parent_program: Parent program object

        Returns:
            Optional[Result] with evaluation results
        """
        async with self._evaluation_semaphore:
            result = Result()
            iteration_start = time.time()

            try:
                result.parent = parent_program

                # Parse response (diff / full-rewrite)
                logger.debug(f"Response length: {len(response)}")
                logger.debug(f"First 500 chars of response: {response[:500]}")

                if self.config.diff_based_evolution:
                    logger.debug("Using diff-based evolution")
                    diff_blocks = extract_diffs(response)
                    logger.debug(f"Extracted {len(diff_blocks) if diff_blocks else 0} diff blocks")

                    # Check for NO_DIFF_BLOCKS_ERROR (most severe)
                    if not diff_blocks:
                        logger.warning(ErrorMessages.get_message(ErrorCodes.NO_DIFF_BLOCKS_ERROR))
                        result.child_metrics = create_evaluation_metrics(
                            combined_score=ErrorCodes.NO_DIFF_BLOCKS_ERROR,
                            validity=0.0,
                            score_transform_config=self.score_transform_config
                        )
                        result.artifacts = {
                            "error": ErrorMessages.get_message(ErrorCodes.NO_DIFF_BLOCKS_ERROR),
                            "error_type": "no_diff_blocks"
                        }

                        # Create child program for error case
                        child_program = Program(
                            id=str(uuid.uuid4()),
                            code="",  # Empty code for no diff blocks case
                            language=self.config.language,
                            parent_id=parent_program.id,
                            generation=parent_program.generation + 1,
                            metrics=result.child_metrics,
                            iteration_found=len(self.database.programs),
                            metadata={
                                "changes": ErrorMessages.get_message(ErrorCodes.NO_DIFF_BLOCKS_ERROR),
                                "parent_metrics": parent_program.metrics
                            }
                        )
                        result.child_program = child_program
                        return result

                    # Apply diffs
                    child_code = apply_diff(parent_program.code, response)
                    changes_summary = format_diff_summary(diff_blocks)
                    logger.info(f"Applied {len(diff_blocks)} diffs, resulting in {len(child_code)} chars")

                    # Check for NO_VALID_CHANGE (second most severe)
                    if check_code_identical(child_code, parent_program.code, self.config.language):
                        logger.warning(ErrorMessages.get_message(ErrorCodes.NO_VALID_CHANGE))
                        result.child_metrics = create_evaluation_metrics(
                            combined_score=ErrorCodes.NO_VALID_CHANGE,
                            validity=0.0,
                            score_transform_config=self.score_transform_config
                        )
                        result.artifacts = {
                            "error": ErrorMessages.get_message(ErrorCodes.NO_VALID_CHANGE),
                            "error_type": "no_valid_change"
                        }

                        child_program = Program(
                            id=str(uuid.uuid4()),
                            code=child_code,
                            language=self.config.language,
                            parent_id=parent_program.id,
                            generation=parent_program.generation + 1,
                            metrics=result.child_metrics,
                            iteration_found=len(self.database.programs),
                            metadata={
                                "changes": ErrorMessages.get_message(ErrorCodes.NO_VALID_CHANGE),
                                "parent_metrics": parent_program.metrics
                            }
                        )
                        result.child_program = child_program
                        return result
                else:
                    assert False, "Full rewrite mode is not supported now!"

                # Length limit
                if len(child_code) > self.config.max_code_length:
                    logger.warning(
                        f"Generated code exceeds maximum length ({len(child_code)} > {self.config.max_code_length})"
                    )
                    print(f"[SingleTaskGym] Generated code exceeds maximum length ({len(child_code)} > {self.config.max_code_length})")
                    return None

                # Build child program object (before evaluation, for lazy detection)
                child_id = str(uuid.uuid4())
                child_program = Program(
                    id=child_id,
                    code=child_code,
                    language=self.config.language,
                    parent_id=parent_program.id,
                    generation=parent_program.generation + 1,
                    metrics={},  # Empty for now, lazy detection doesn't need evaluation metrics
                    iteration_found=len(self.database.programs),
                    metadata={
                        "changes": changes_summary,
                        "parent_metrics": parent_program.metrics,
                    },
                )

                # Additional lazy output detection (before evaluation to save computation)
                if self.lazy_output_penalty_level >= 2:
                    assert False, "Lazy output penalty level 2 is not supported in debug, comment this line to enable."
                    lazy_penalty = self.database.check_lazy_output(
                        child_program, parent_program, self.lazy_output_penalty_level
                    )

                    if lazy_penalty is not None:
                        # Lazy output, return penalty score directly without evaluation
                        result.child_metrics = {"combined_score": lazy_penalty}
                        result.child_program = child_program
                        result.child_program.metrics = result.child_metrics

                        logger.info(f"Lazy output penalty applied: {lazy_penalty}")
                        return result

                # Not lazy output, continue with normal evaluation
                timeout_s = getattr(self.config.evaluator, "timeout_s", None) or getattr(self.config.evaluator, "timeout", None) or 1000
                try:
                    result.child_metrics = await asyncio.wait_for(
                        self.evaluator.evaluate_program(child_code, child_id),
                        timeout=timeout_s
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Evaluation timed out for child {child_id[:8]}")
                    result.child_metrics = {
                        "combined_score": ErrorCodes.EXECUTION_ERROR,
                        "timeout": True,
                        "error": 0.0
                    }
                    result.child_program = child_program
                    return result

                # Ensure combined_score exists (handle evaluator internal timeout/error cases)
                if "combined_score" not in result.child_metrics:
                    logger.warning(f"Missing combined_score in metrics for child {child_id[:8]}, using EXECUTION_ERROR")
                    result.child_metrics["combined_score"] = ErrorCodes.EXECUTION_ERROR

                # Artifacts
                artifacts = self.evaluator.get_pending_artifacts(child_id)

                # Update child program object with evaluation results
                child_program.metrics = result.child_metrics
                result.child_program = child_program

                # Associate/persist prompt (concurrency-safe)
                if self.log_prompts:
                    await self._associate_prompt_with_program(child_id, parent_program.id, response)
                    await self._log_prompt_to_database(child_id, parent_program.id)

                # Result fields
                result.llm_response = response
                result.artifacts = artifacts
                result.iteration_time = time.time() - iteration_start

                pending_envs = getattr(self.evaluator, "_pending_runtime_environments", None)
                if isinstance(pending_envs, dict) and child_id in pending_envs:
                    result.runtime_environment_path = pending_envs[child_id]

                logger.info(f"Successfully evaluated child program {child_id}, metrics: {result.child_metrics}")
                return result

            except asyncio.CancelledError:
                logger.info("response_scorer cancelled")
                raise
            except Exception as e:
                logger.exception(f"Error in gym iteration: {e}")
                print(f"[SingleTaskGym] Error in gym iteration: {e}")
                return None

    async def check_and_reinit_database(self, current_step: int = None, verbose: bool = True,
                                       add_historical_good_programs: int = 0) -> bool:
        """Check and execute database reinitialization"""

        async def reinit_func():
            # Reset initialization flag to allow reinitialization
            self._initialized = False
            await self.initialize()  # Load initial program

            # Load historical best programs
            print(f"add_historical_good_programs={add_historical_good_programs}")
            if add_historical_good_programs > 0:
                await self._reload_historical_programs(add_historical_good_programs, verbose)

        result = await self.database.database_reinit_async(
            initialize_func=reinit_func,
            initial_program_path=self.initial_program_path,
            current_step=current_step,
            verbose=verbose
        )

        # Only update step record after successful reinitialization
        if result and current_step is not None:
            self.database.last_reinit_step = current_step

        return result

    async def _reload_historical_programs(self, count: int, verbose: bool = True):
        """
        Reload historical best programs

        Args:
            count: Number of historical programs to load
            verbose: Whether to output detailed information
        """
        if not self.database.historical_best_programs:
            if verbose:
                print("[REINIT] No historical programs to reload")
            return

        # Sort by score and take top count
        sorted_historical = sorted(
            self.database.historical_best_programs,
            key=lambda x: x["score"],
            reverse=True
        )[:count]

        successful_reloads = 0

        for i, record in enumerate(sorted_historical):
            try:
                # Use saved metrics directly without re-evaluation
                historical_program = Program(
                    id=f"historical_{record['program_id'][:8]}_{i}",
                    code=record["code"],
                    language=record["language"],
                    metrics=record["metrics"],  # Use saved metrics directly
                    parent_id=None,
                    generation=0,
                    iteration_found=0
                )

                self.database.add(historical_program, iteration=0)
                successful_reloads += 1

                if verbose:
                    current_score = record["metrics"].get('combined_score', record["score"])
                    print(f"[REINIT] Reloaded historical program {i+1}/{count}: "
                          f"score={current_score:.6f}, ID={record['program_id'][:8]}")

            except Exception as e:
                if verbose:
                    print(f"[REINIT] Failed to reload historical program {i}: {e}")

        if verbose:
            print(f"[REINIT] Successfully reloaded {successful_reloads}/{len(sorted_historical)} historical programs")

    # === Recording and Visualization (Optional) ===

    def enable_recording(self, output_dir: Optional[str] = "./gym_output") -> None:
        """
        Enable recording and visualization functionality

        Args:
            output_dir: Output directory for recordings (defaults to gym_output)
        """
        from openevolve.evolving_gym.gym_recorder import GymRecorder
        self._recorder = GymRecorder(self, output_dir)
        logger.info("Recording enabled")

    def record_progress(self, training_step: int, **kwargs) -> None:
        """
        Record current gym state (requires initialize() + enable_recording() first)

        Args:
            training_step: Real training step number (required)
            **kwargs: Additional options passed to recorder (save_checkpoint, save_historical_records)
        """
        if self._recorder is None:
            logger.warning("Recording not enabled. Call enable_recording() first.")
            return

        self._recorder.record_step(training_step, **kwargs)

    @property
    def recording_enabled(self) -> bool:
        """Check if recording is enabled"""
        return self._recorder is not None
