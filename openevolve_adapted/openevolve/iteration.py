import asyncio
import os
import uuid
import logging
import time
from dataclasses import dataclass

from openevolve.database import Program, ProgramDatabase
from openevolve.config import Config
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None
    runtime_environment_path: str = None  # For syncing worker runtime environments


async def run_iteration_with_shared_db(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
):
    """
    Run a single iteration using shared memory database

    This is optimized for use with persistent worker processes.
    """
    logger = logging.getLogger(__name__)

    try:
        # Sample parent and inspirations from database
        parent, inspirations = database.sample(config.prompt.num_inspiration_programs)

        # Get artifacts for the parent program if available
        parent_artifacts = database.get_artifacts(parent.id)

        # Get island-specific top programs for prompt context (maintain island isolation)
        parent_island = parent.metadata.get("island", database.current_island)
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(3, island_idx=parent_island)

        # Enhance programs with artifacts if requested in config
        def enhance_programs_with_artifacts(programs, include_artifacts):
            if not include_artifacts:
                return [p.to_dict() for p in programs]
            
            enhanced = []
            for p in programs:
                program_dict = p.to_dict()
                artifacts = database.get_artifacts(p.id)
                if artifacts:
                    program_dict['artifacts'] = artifacts
                enhanced.append(program_dict)
            return enhanced

        # Determine artifact inclusion settings
        include_top_artifacts = config.prompt.include_artifacts_for_all_programs or config.prompt.include_top_program_artifacts
        include_inspiration_artifacts = config.prompt.include_artifacts_for_all_programs or config.prompt.include_inspiration_artifacts
        
        # Build prompt
        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=enhance_programs_with_artifacts(island_previous_programs, include_top_artifacts),
            top_programs=enhance_programs_with_artifacts(island_top_programs, include_top_artifacts),
            inspirations=enhance_programs_with_artifacts(inspirations, include_inspiration_artifacts),
            language=config.language,
            evolution_round=iteration,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
        )

        result = Result(parent=parent)
        iteration_start = time.time()

        # Generate code modification
        llm_response = await llm_ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )

        # Parse the response
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response)

            if not diff_blocks:
                logger.warning(f"Iteration {iteration+1}: No valid diffs found in response")
                return None

            # Apply the diffs
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            # Parse full rewrite
            new_code = parse_full_rewrite(llm_response, config.language)

            if not new_code:
                logger.warning(f"Iteration {iteration+1}: No valid code found in response")
                return None

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > config.max_code_length:
            logger.warning(
                f"Iteration {iteration+1}: Generated code exceeds maximum length "
                f"({len(child_code)} > {config.max_code_length})"
            )
            return None

        # Evaluate the child program
        child_id = str(uuid.uuid4())
        result.child_metrics = await evaluator.evaluate_program(child_code, child_id)

        # Handle artifacts if they exist
        artifacts = evaluator.get_pending_artifacts(child_id)

        # Create a child program
        result.child_program = Program(
            id=child_id,
            code=child_code,
            language=config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=result.child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
            },
        )

        result.prompt = prompt
        result.llm_response = llm_response
        result.artifacts = artifacts
        result.iteration_time = time.time() - iteration_start
        result.iteration = iteration

        # Get runtime environment path from worker evaluator for syncing
        if hasattr(evaluator, '_pending_runtime_environments') and child_id in evaluator._pending_runtime_environments:
            result.runtime_environment_path = evaluator._pending_runtime_environments[child_id]
        
        return result

    except Exception as e:
        logger.exception(f"Error in iteration {iteration}: {e}")
        return None
