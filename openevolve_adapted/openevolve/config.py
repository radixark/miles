"""
Configuration handling for OpenEvolve
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float = None
    top_p: float = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(
        default_factory=lambda: [
            LLMModelConfig(name="gpt-4o-mini", weight=0.8),
            LLMModelConfig(name="gpt-4o", weight=0.2),
        ]
    )

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if (self.primary_model or self.primary_model_weight) and len(self.models) < 1:
            # Ensure we have a primary model
            self.models.append(LLMModelConfig())
        if self.primary_model:
            self.models[0].name = self.primary_model
        if self.primary_model_weight:
            self.models[0].weight = self.primary_model_weight

        if (self.secondary_model or self.secondary_model_weight) and len(self.models) < 2:
            # Ensure we have a second model
            self.models.append(LLMModelConfig())
        if self.secondary_model:
            self.models[1].name = self.secondary_model
        if self.secondary_model_weight:
            self.models[1].weight = self.secondary_model_weight

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models or len(self.evaluator_models) < 1:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"
    
    # Multi-prompt support - list of system messages with sampling weights
    system_message_list: Optional[List[Dict[str, Union[str, float]]]] = None
    use_system_message_sampling: bool = False

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    num_inspiration_programs: int = 5

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True
    # Artifact examples type: 0 = none, 1 = alphaevolve_artifact_examples, others reserved for future use
    artifact_examples_type: int = 1
    # Stdout collection control
    max_stdout_chars: int = 4000  # Maximum characters of stdout to include per program
    include_top_program_artifacts: bool = False  # Include artifacts for top programs
    include_inspiration_artifacts: bool = False  # Include artifacts for inspiration programs
    include_artifacts_for_all_programs: bool = False  # If true, enable artifacts for all program types (overrides individual flags)
    
    # AlphaEvolve-style prompt configuration
    use_alphaevolve_style: bool = False
    use_system_prompt: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Bulk I/O mode for efficient disk operations (single file vs. many small files)
    use_bulk_io: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Quality filtering to prevent low-quality programs when database is full
    quality_filter_enabled: bool = False #True  # Only add programs with score > worst_score when DB is full

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program

    # File copying for evaluation
    copy_folders: List[str] = field(default_factory=list)  # Folders to copy to temp directory
    copy_files: List[str] = field(default_factory=list)    # Individual files to copy to temp directory
    
    # Runtime environment collection
    collect_runtime_environments: bool = True  # Whether to collect runtime environment from temp directories
    runtime_environment_patterns: List[str] = field(default_factory=lambda: ["files/*", "checkpoints/*", "results/*", "output/*"])  # Patterns to collect
    preserve_temp_directories: bool = False  # Whether to preserve temporary directories for debugging


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None
    
    # Output directory settings
    output_postfix: Optional[str] = None  # Postfix to add to output directory name
    
    # Checkpoint preservation settings
    codes_interval_ratio: int = 5  # Keep full checkpoint data every N * checkpoint_interval iterations
    
    # Real-time visualization settings
    enable_realtime_visualizations: bool = True  # Generate visualizations at each checkpoint

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000
    
    # Variable substitution for templates
    template_variables: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    def substitute_template_variables(self, text: str) -> str:
        """Substitute template variables in text using {variable_name} format
        
        Supports both flat variables {var_name} and nested {dict.key} syntax
        """
        if not self.template_variables or not text:
            return text
        
        result = text
        
        # First handle flat variables
        for var_name, var_value in self.template_variables.items():
            placeholder = "{" + var_name + "}"
            
            # Handle different types of variables
            if isinstance(var_value, (dict, list)):
                # Handle nested dict access like {core_parameters.n_circles}
                if isinstance(var_value, dict):
                    for nested_key, nested_value in var_value.items():
                        nested_placeholder = "{" + var_name + "." + nested_key + "}"
                        result = result.replace(nested_placeholder, str(nested_value))
                # Skip other complex structures
                continue
            else:
                # Simple values (string, int, float) 
                result = result.replace(placeholder, str(var_value))
        
        return result

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        # Set environment variable for modular_utils to use
        os.environ['OPENEVOLVE_CONFIG_PATH'] = str(Path(path).absolute())
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Pre-process variables to add derived values for prompt substitution
        if "variables" in config_dict:
            variables = config_dict["variables"]
            core_parameters = variables.get('core_parameters', {})
            
            # Add target_value for prompt substitution if exists in core_parameters
            target_value = core_parameters.get('target_value') or variables.get('TARGET_VALUE')
            if target_value is not None:
                variables['target_value'] = target_value
            
        
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Handle template variables first (separate from other top-level fields)
        if "variables" in config_dict:
            config.template_variables = config_dict["variables"]

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator", "variables"] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])

        # Apply template variable substitution to system messages
        if config.template_variables:
            config.prompt.system_message = config.substitute_template_variables(config.prompt.system_message)
            config.prompt.evaluator_system_message = config.substitute_template_variables(config.prompt.evaluator_system_message)
            
            # Also apply to system message list if it exists
            if config.prompt.system_message_list:
                for item in config.prompt.system_message_list:
                    if "message" in item:
                        item["message"] = config.substitute_template_variables(item["message"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            "output_postfix": self.output_postfix,
            "codes_interval_ratio": self.codes_interval_ratio,
            "template_variables": self.template_variables,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "system_message_list": self.prompt.system_message_list,
                "use_system_message_sampling": self.prompt.use_system_message_sampling,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "num_inspiration_programs": self.prompt.num_inspiration_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                # Note: meta-prompting features not implemented
                # "use_meta_prompting": self.prompt.use_meta_prompting,
                # "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                # Note: diversity_metric fixed to "edit_distance"
                # "diversity_metric": self.database.diversity_metric,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "log_prompts": self.database.log_prompts,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                # Note: resource limits not implemented
                # "memory_limit_mb": self.evaluator.memory_limit_mb,
                # "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                # Note: distributed evaluation not implemented
                # "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
                "copy_folders": self.evaluator.copy_folders,
                "copy_files": self.evaluator.copy_files,
            },
            # Evolution settings
            "diff_based_evolution": self.diff_based_evolution,
            "max_code_length": self.max_code_length,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
