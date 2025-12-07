"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# All templates will be loaded from files in prompts/default/


# Templates will be loaded from files
DEFAULT_TEMPLATES = {}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = {}

        # Load from default template directory first
        default_template_dir = Path(__file__).parent.parent / "prompts" / "default"
        if default_template_dir.exists():
            self._load_templates_from_dir(str(default_template_dir))
        else:
            raise FileNotFoundError(f"Default template directory not found: {default_template_dir}")

        # Load templates from user-provided directory if specified (will override defaults)
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
