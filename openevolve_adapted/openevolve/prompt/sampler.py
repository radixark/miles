"""
Prompt sampling for OpenEvolve
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution"""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(config.template_dir)

        # Note: Do not reset random seed here
        # The global random seed should be set by the caller (database or controller)
        # Calling random.seed() without arguments would reset to system time, breaking reproducibility

        # Store custom template mappings
        self.system_template_override = None
        self.user_template_override = None

        # Only log once to reduce duplication
        if not hasattr(logger, "_prompt_sampler_logged"):
            logger.info("Initialized prompt sampler")
            logger._prompt_sampler_logged = True

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        Set custom templates to use for this sampler

        Args:
            system_template: Template name for system message
            user_template: Template name for user message
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],  # Add inspirations parameter
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM

        Args:
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs (best by fitness)
            inspirations: List of inspiration programs (diverse/creative examples)
            language: Programming language
            evolution_round: Current evolution round
            diff_based_evolution: Whether to use diff-based evolution (True) or full rewrites (False)
            template_key: Optional override for template key
            program_artifacts: Optional artifacts from program evaluation
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on evolution mode (with overrides)
        if template_key:
            # Use explicitly provided template key
            user_template_key = template_key
        elif self.user_template_override:
            # Use the override set with set_templates
            user_template_key = self.user_template_override
        elif self.config.use_alphaevolve_style:
            # Use AlphaEvolve-style templates
            user_template_key = "alphaevolve_diff_user"
        else:
            # Default behavior: diff-based vs full rewrite
            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"

        # Get the template
        user_template = self.template_manager.get_template(user_template_key)

        # Use system template override if set
        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
        elif self.config.use_alphaevolve_style and not self.config.use_system_prompt:
            # For AlphaEvolve style, system message can be moved to user prompt
            # system_message = "You are an expert software developer and mathematician."
            system_message = "You are an expert software developer and mathematician. working on optimization problems. Your task is to iteratively improve the provided codebase."
        else:
            # Sample system message (either from list or use single message)
            system_message = self._sample_system_message()
            # If system_message is a template name rather than content, get the template
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        # Format metrics
        metrics_str = self._format_metrics(program_metrics)

        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )

        # Format evolution history
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language
        )

        # Format artifacts section if enabled and available
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Format the final user message
        if self.config.use_alphaevolve_style:
            # For AlphaEvolve style, include context in user message if needed
            context = ""
            if not self.config.use_system_prompt:
                # Move system message to user prompt
                sampled_system = self._sample_system_message()
                if sampled_system in self.template_manager.templates:
                    sampled_system = self.template_manager.get_template(sampled_system)
                # Use the sampled system message as context, no duplication
                context = sampled_system if sampled_system != "system_message" else "Act as an expert software developer. Your task is to iteratively improve the provided codebase."
            
            # Format current program using unified formatter
            current_program_data = {
                "metrics": program_metrics,
                "code": current_program,
                "artifacts": program_artifacts if self.config.include_artifacts else None
            }
            current_program_content = self._format_unified_program(
                current_program_data, 
                language, 
                "---------------- Current Program ----------------"
            )
            
            # Get artifact examples content based on configuration
            artifact_examples = ""
            if self.config.include_artifacts and self.config.artifact_examples_type == 1:
                artifact_examples = self.template_manager.get_template("alphaevolve_artifact_examples")
            elif self.config.include_artifacts and self.config.artifact_examples_type != 0:
                # Future artifact example types can be added here
                # For now, raise NotImplementedError for unsupported types
                raise NotImplementedError(f"Artifact examples type {self.config.artifact_examples_type} is not implemented yet")
            
            user_message = user_template.format(
                context=context,
                evolution_history=evolution_history,
                current_program_content=current_program_content,
                artifact_examples=artifact_examples,
                **kwargs,
            )
        else:
            user_message = user_template.format(
                metrics=metrics_str,
                improvement_areas=improvement_areas,
                evolution_history=evolution_history,
                current_program=current_program,
                language=language,
                artifacts=artifacts_section,
                **kwargs,
            )

        return {
            "system": system_message,
            "user": user_message,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics consistently for all programs"""
        formatted_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"{name}: {value:.3f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"{name}: {value}")
            else:
                formatted_parts.append(f"{name}: {value}")
        return "; ".join(formatted_parts)

    def _format_unified_program(self, program_data: Dict[str, Any], language: str, separator: str) -> str:
        """Unified program formatter - all programs use this for consistent formatting"""
        # Format metrics consistently
        metrics_str = self._format_metrics(program_data.get("metrics", {}))
        
        # Format artifacts if present
        artifacts_str = ""
        if "artifacts" in program_data and program_data["artifacts"]:
            artifacts_str = "\n" + self._render_artifacts(program_data["artifacts"])
        
        return self.template_manager.get_template("unified_program").format(
            separator=separator,
            metrics=metrics_str,
            language=language,
            program_code=program_data.get("code", ""),
            artifacts=artifacts_str
        )

    def _format_program_section(self, programs: List[Dict[str, Any]], language: str, section_type: str, max_programs: int) -> str:
        """Format a section of programs (prior/diverse/inspiration) with consistent formatting"""
        if not programs:
            return ""
            
        program_list = []
        selected_programs = programs[:min(max_programs, len(programs))]
        
        for i, program in enumerate(selected_programs):
            separator = f"---------------- {section_type} Program {i+1} ----------------"
            if i > 0:
                separator = f"\n{separator}"
            
            program_content = self._format_unified_program(program, language, separator)
            program_list.append(program_content)
        
        return "\n\n".join(program_list)

    def _format_previous_programs_section(self, previous_programs: List[Dict[str, Any]], language: str) -> str:
        """Format previous programs section (evolution history)"""
        if not previous_programs:
            return ""
        
        # Show most recent 3 attempts, most recent first
        selected_previous = previous_programs[-min(3, len(previous_programs)):]
        reversed_previous = list(reversed(selected_previous))
        
        return self._format_program_section(reversed_previous, language, "Previous", len(reversed_previous))

    def _format_top_programs_section(self, top_programs: List[Dict[str, Any]], language: str) -> str:
        """Format top performing programs section"""
        if not top_programs:
            return ""
        
        return self._format_program_section(top_programs, language, "Top", self.config.num_top_programs)

    def _format_diverse_programs_section(self, top_programs: List[Dict[str, Any]], language: str) -> str:
        """Format diverse programs section"""
        if (
            self.config.num_diverse_programs <= 0
            or len(top_programs) <= self.config.num_top_programs
        ):
            return ""
        
        # Skip the top programs we already included
        remaining_programs = top_programs[self.config.num_top_programs:]
        num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
        
        if num_diverse > 0:
            # Use random sampling to get diverse programs
            diverse_programs = random.sample(remaining_programs, num_diverse)
            return self._format_program_section(diverse_programs, language, "Diverse", num_diverse)
        
        return ""

    def _format_inspirations_section_v2(self, inspirations: List[Dict[str, Any]], language: str) -> str:
        """Format inspiration programs section (new modular version)"""
        if not inspirations or self.config.num_inspiration_programs <= 0:
            return ""
        
        return self._format_program_section(inspirations, language, "Inspiration", self.config.num_inspiration_programs)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
    ) -> str:
        """Identify potential areas for improvement"""
        # This method could be expanded to include more sophisticated analysis
        # For now, we'll use a simple approach

        improvement_areas = []

        # Check program length
        if len(current_program) > 500:
            improvement_areas.append(
                "Consider simplifying the code to improve readability and maintainability"
            )

        # Check for performance patterns in previous attempts
        if len(previous_programs) >= 2:
            recent_attempts = previous_programs[-2:]
            metrics_improved = []
            metrics_regressed = []

            for metric, value in metrics.items():
                # Only compare numeric metrics
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue

                improved = True
                regressed = True

                for attempt in recent_attempts:
                    attempt_value = attempt["metrics"].get(metric, 0)
                    # Only compare if both values are numeric
                    if isinstance(value, (int, float)) and isinstance(attempt_value, (int, float)):
                        if attempt_value <= value:
                            regressed = False
                        if attempt_value >= value:
                            improved = False
                    else:
                        # If either value is non-numeric, skip comparison
                        improved = False
                        regressed = False

                if improved and metric not in metrics_improved:
                    metrics_improved.append(metric)
                if regressed and metric not in metrics_regressed:
                    metrics_regressed.append(metric)

            if metrics_improved:
                improvement_areas.append(
                    f"Metrics showing improvement: {', '.join(metrics_improved)}. "
                    "Consider continuing with similar changes."
                )

            if metrics_regressed:
                improvement_areas.append(
                    f"Metrics showing regression: {', '.join(metrics_regressed)}. "
                    "Consider reverting or revising recent changes in these areas."
                )

        # If we don't have specific improvements to suggest
        if not improvement_areas:
            improvement_areas.append(
                "Focus on optimizing the code for better performance on the target metrics"
            )

        return "\n".join([f"- {area}" for area in improvement_areas])

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        if self.config.use_alphaevolve_style:
            return self._format_alphaevolve_evolution_history(
                previous_programs, top_programs, inspirations, language
            )
        
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format each section using modular functions
        previous_attempts_str = self._format_previous_programs_section(previous_programs, language)
        top_programs_str = self._format_top_programs_section(top_programs, language)
        
        # Format diverse programs with section header
        diverse_programs_content = self._format_diverse_programs_section(top_programs, language)
        diverse_programs_str = ""
        if diverse_programs_content:
            diverse_programs_str = "\n\n - Diverse Programs\n\n" + diverse_programs_content

        # Combine top and diverse programs
        combined_programs_str = top_programs_str + diverse_programs_str

        # Format inspirations section using modular function
        inspirations_section_str = self._format_inspirations_section_v2(inspirations, language)

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip() if previous_attempts_str else "",
            top_programs=combined_programs_str.strip() if combined_programs_str else "",
            inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self, inspirations: List[Dict[str, Any]], language: str
    ) -> str:
        """
        Format the inspirations section for the prompt

        Args:
            inspirations: List of inspiration programs
            language: Programming language

        Returns:
            Formatted inspirations section string
        """
        if not inspirations:
            return ""

        # Get templates
        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")

        inspiration_programs_str = ""

        # Use unified program formatting for inspirations
        inspiration_programs_content = self._format_program_section(
            inspirations, language, "Inspiration", len(inspirations)
        )
        inspiration_programs_str = inspiration_programs_content

        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )

    def _determine_program_type(self, program: Dict[str, Any]) -> str:
        """
        Determine the type/category of an inspiration program

        Args:
            program: Program dictionary

        Returns:
            String describing the program type
        """
        metadata = program.get("metadata", {})
        score = safe_numeric_average(program.get("metrics", {}))

        # Check metadata for explicit type markers
        if metadata.get("diverse", False):
            return "Diverse"
        if metadata.get("migrant", False):
            return "Migrant"
        if metadata.get("random", False):
            return "Random"

        # Classify based on score ranges
        if score >= 0.8:
            return "High-Performer"
        elif score >= 0.6:
            return "Alternative"
        elif score >= 0.4:
            return "Experimental"
        else:
            return "Exploratory"

    def _extract_unique_features(self, program: Dict[str, Any]) -> str:
        """
        Extract unique features of an inspiration program

        Args:
            program: Program dictionary

        Returns:
            String describing unique aspects of the program
        """
        features = []

        # Extract from metadata if available
        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if isinstance(changes, str) and len(changes) < 100:
                features.append(f"Modification: {changes}")

        # Analyze metrics for standout characteristics
        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    features.append(f"Excellent {metric_name} ({value:.3f})")
                elif value <= 0.3:
                    features.append(f"Alternative {metric_name} approach")

        # Code-based features (simple heuristics)
        code = program.get("code", "")
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append("Object-oriented approach")
            if "numpy" in code_lower or "np." in code_lower:
                features.append("NumPy-based implementation")
            if "for" in code_lower and "while" in code_lower:
                features.append("Mixed iteration strategies")
            if len(code.split("\n")) < 10:
                features.append("Concise implementation")
            elif len(code.split("\n")) > 50:
                features.append("Comprehensive implementation")

        # Default if no specific features found
        if not features:
            program_type = self._determine_program_type(program)
            features.append(f"{program_type} approach to the problem")

        # Use num_top_programs as limit for features (similar to how we limit programs)
        feature_limit = self.config.num_top_programs
        return ", ".join(features[:feature_limit])

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template

        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result

    def _sample_system_message(self) -> str:
        """Sample a system message from the configured list if enabled"""
        if not self.config.use_system_message_sampling or not self.config.system_message_list:
            return self.config.system_message
        
        # Extract messages and weights
        messages = []
        weights = []
        
        for item in self.config.system_message_list:
            messages.append(item.get("message", ""))
            weights.append(item.get("weight", 1.0))
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight <= 0:
            # Fallback to uniform sampling if all weights are zero
            return random.choice(messages)
        
        probabilities = [w / total_weight for w in weights]
        
        # Weighted random selection
        return random.choices(messages, weights=probabilities)[0]

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """
        Render artifacts for prompt inclusion

        Args:
            artifacts: Dictionary of artifact name to content

        Returns:
            Formatted string for prompt inclusion (empty string if no artifacts)
        """
        if not artifacts:
            return ""

        sections = []
        
        # Priority order for artifacts - successful execution output first
        priority_keys = [
            "last_execution_output",      # Successful runs with full output
            "execution_summary",          # Summary of all runs
            "performance_analysis",       # Performance metrics
            "failed_execution_output",    # Failed run outputs
            "failure_details",            # Details about failures
            "compilation_errors"          # Compilation error details
        ]
        
        # Add priority artifacts first
        processed_keys = set()
        for priority_key in priority_keys:
            if priority_key in artifacts:
                content = self._safe_decode_artifact(artifacts[priority_key])
                
                # For successful execution output, allow more space (up to 5000 chars)
                max_bytes = self.config.max_artifact_bytes
                if priority_key == "last_execution_output":
                    max_bytes = min(5000, self.config.max_artifact_bytes * 2)
                
                # Truncate if too long
                if len(content) > max_bytes:
                    content = content[:max_bytes] + "\n... (truncated)"

                sections.append(f"[{priority_key}]\n```\n{content}\n```")
                processed_keys.add(priority_key)
        
        # Add any remaining artifacts
        for key, value in artifacts.items():
            if key not in processed_keys:
                content = self._safe_decode_artifact(value)
                # Truncate if too long
                if len(content) > self.config.max_artifact_bytes:
                    content = content[: self.config.max_artifact_bytes] + "\n... (truncated)"

                sections.append(f"[{key}]\n```\n{content}\n```")

        if sections:
            return "\n" + "\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """
        Safely decode an artifact value to string

        Args:
            value: Artifact value (string or bytes)

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Apply security filter if enabled
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _format_alphaevolve_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format evolution history in AlphaEvolve style"""
        sections = []
        
        # Format prior programs using unified section formatter
        if top_programs:
            prior_programs_content = self._format_program_section(
                top_programs, language, "Prior", self.config.num_top_programs
            )
            if prior_programs_content:
                sections.append(
                    self.template_manager.get_template("alphaevolve_prior_programs").format(
                        prior_programs_list=prior_programs_content
                    )
                )
        
        # Format diverse programs using unified section formatter
        if (self.config.num_diverse_programs > 0 and len(top_programs) > self.config.num_top_programs):
            remaining_programs = top_programs[self.config.num_top_programs :]
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            
            if num_diverse > 0:
                diverse_programs = random.sample(remaining_programs, num_diverse)
                diverse_programs_content = self._format_program_section(
                    diverse_programs, language, "Diverse", num_diverse
                )
                if diverse_programs_content:
                    sections.append(
                        self.template_manager.get_template("alphaevolve_diverse_programs").format(
                            diverse_programs_list=diverse_programs_content
                        )
                    )
        
        # Format inspiration programs using unified section formatter
        if inspirations and self.config.num_inspiration_programs > 0:
            inspiration_programs_content = self._format_program_section(
                inspirations, language, "Inspiration", self.config.num_inspiration_programs
            )
            if inspiration_programs_content:
                sections.append(
                    self.template_manager.get_template("alphaevolve_inspirations").format(
                        inspiration_programs_list=inspiration_programs_content
                    )
                )
        
        return "\n\n".join(sections)
    
    def _apply_security_filter(self, text: str) -> str:
        """
        Apply security filtering to artifact text

        Args:
            text: Input text

        Returns:
            Filtered text with potential secrets/sensitive info removed
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Basic patterns for common secrets (can be expanded)
        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),  # Long alphanumeric tokens
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),  # OpenAI-style API keys
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),  # Password assignments
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),  # Token assignments
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
