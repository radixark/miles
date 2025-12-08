import os
import json
import tempfile
from transformers import AutoConfig

_patched = False


def apply_deepseek_v32_patch(restore_model_type=False):
    global _patched
    
    if _patched:
        return
    
    _original_from_pretrained = AutoConfig.from_pretrained
    
    def _patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if isinstance(pretrained_model_name_or_path, str) and os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config_json = json.load(f)
                    
                    if config_json.get("model_type") == "deepseek_v32":
                        config_json["model_type"] = "deepseek_v3"
                        if "architectures" in config_json:
                            config_json["architectures"] = ["DeepseekV3ForCausalLM"]
                        
                        tmp_path = os.path.join(tempfile.gettempdir(), "_tmp_config_folder")
                        os.makedirs(tmp_path, exist_ok=True)
                        unique_path = os.path.join(tmp_path, f"deepseek_v32_{os.getpid()}.json")
                        
                        with open(unique_path, "w") as f:
                            json.dump(config_json, f)
                        
                        config = _original_from_pretrained(unique_path, *args, **kwargs)
                        
                        if restore_model_type:
                            object.__setattr__(config, "model_type", "deepseek_v32")
                        
                        return config
                except Exception:
                    pass
        
        return _original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    
    AutoConfig.from_pretrained = _patched_from_pretrained
    _patched = True

