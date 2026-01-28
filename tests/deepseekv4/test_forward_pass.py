#!/usr/bin/env python3
"""
Test script for running forward/backward pass with Megatron or Reference implementation.
Uses typer for CLI (inspired by send_one_compare.py).

Usage:
    # Run Megatron forward pass
    python test_forward_pass.py megatron \
        --hf-checkpoint /data/weights/hello2026_5layer \
        --ref-load /data/weights/hello2026_5layer_torch_dist

    # Run Megatron backward pass
    python test_forward_pass.py megatron-backward \
        --hf-checkpoint /data/weights/hello2026_5layer \
        --ref-load /data/weights/hello2026_5layer_torch_dist

    # Run Reference implementation
    python test_forward_pass.py reference \
        --ckpt-path /data/weights/hello2026_native \
        --config-path /path/to/config_285B.json
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()

# Default paths - can be overridden via environment variables
MEGATRON_PATH = os.environ.get("MEGATRON_PATH", "/host_home/primary_synced/megatron-sunrise")
SCRIPT_DIR = Path(__file__).parent

# Reference implementation config path
REF_CONFIG_PATH = SCRIPT_DIR / "config_285B.json"


def make_input(
    tokenizer,
    input_seq_len: int,
    prompt_mode: str,
    prompt_text: Optional[str] = None,
    prompt_file: Optional[str] = None,
    apply_chat_template: bool = False,
    postfix_question: bool = False,
    postfix_dummy_len: int = 0,
) -> str:
    """Generate prompt text based on prompt mode (inspired by send_one_compare.py).
    
    Returns the prompt text (not tokenized).
    """
    if prompt_mode == "math":
        # Generate arithmetic sequences like "1+1=2, 1+2=3, ..."
        parts = []
        i = 1
        while True:
            parts.append(f"1+{i}={1+i}")
            text = ", ".join(parts)
            if apply_chat_template:
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True
                )
            else:
                input_ids = tokenizer.encode(text)
            if len(input_ids) >= input_seq_len:
                break
            i += 1
        return text

    if prompt_mode == "story":
        if prompt_file is None:
            raise ValueError("--prompt-file is required for 'story' mode")
        
        with open(prompt_file, "r") as f:
            text = f.read()

        # Truncate to approximate length
        input_ids = tokenizer.encode(text)
        if len(input_ids) > input_seq_len:
            input_ids = input_ids[:input_seq_len]
            text = tokenizer.decode(input_ids, skip_special_tokens=False)

        if postfix_question:
            text += "\n\nNow please tell me, who is the main character in the story above?"

        if postfix_dummy_len > 0:
            text += (
                "\n\nBelow, I will make some dummy text. Please ignore all text below. "
                + "1 " * postfix_dummy_len
            )

        return text

    if prompt_mode == "text":
        if prompt_text is None:
            raise ValueError("--prompt-text is required for 'text' mode")
        
        text = prompt_text
        
        if postfix_question:
            text += "\n\nPlease answer the question above."
            
        if postfix_dummy_len > 0:
            text += (
                "\n\nBelow, I will make some dummy text. Please ignore all text below. "
                + "1 " * postfix_dummy_len
            )

        return text

    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def get_model_script_path(model_type: str) -> Path:
    """Get the path to the model shell script."""
    # Go up from tests/deepseekv4 to miles-sunrise, then to scripts/models
    repo_base_dir = SCRIPT_DIR.parent.parent
    script_path = repo_base_dir / "scripts" / "models" / f"{model_type}.sh"
    if not script_path.exists():
        raise ValueError(f"Model script not found: {script_path}")
    return script_path


def get_model_args_from_shell(model_type: str) -> str:
    """Source the model shell script and return MODEL_ARGS as a string.
    
    Similar to command_utils.py: source "{repo_base_dir}/scripts/models/{model_type}.sh"
    """
    script_path = get_model_script_path(model_type)
    
    # Use bash to source the script and print MODEL_ARGS
    # This handles array expansion properly
    cmd = f'source "{script_path}" && echo "${{MODEL_ARGS[@]}}"'
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to source model script: {result.stderr}")
    
    return result.stdout.strip()


@app.command()
def megatron(
    hf_checkpoint: str = typer.Option(..., "--hf-checkpoint", help="Path to HuggingFace checkpoint"),
    ref_load: Optional[str] = typer.Option(None, "--ref-load", help="Path to Megatron checkpoint"),
    input_seq_len: int = typer.Option(128, "--input-seq-len", help="Input sequence length"),
    tp_size: int = typer.Option(1, "--tp-size", help="Tensor parallel size"),
    prompt_mode: str = typer.Option("text", "--prompt-mode", help="Prompt mode: math, story, text"),
    prompt_text: Optional[str] = typer.Option("The capital of France is ", "--prompt-text", help="Custom prompt text (for text mode)"),
    prompt_file: Optional[str] = typer.Option(None, "--prompt-file", help="Path to prompt file (for story mode)"),
    apply_chat_template: bool = typer.Option(False, "--apply-chat-template", help="Apply chat template"),
    postfix_question: bool = typer.Option(False, "--postfix-question", help="Add postfix question"),
    postfix_dummy_len: int = typer.Option(0, "--postfix-dummy-len", help="Add dummy text of this length"),
    model_type: str = typer.Option("deepseek-v4-285B-5layer", "--model-type", help="Model type"),
    top_k: int = typer.Option(5, "--top-k", help="Top-k predictions to show"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size"),
):
    """Run Megatron model forward pass."""
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Running Megatron Forward Pass Test")
    print("=" * 60)
    print(f"HF Checkpoint:    {hf_checkpoint}")
    print(f"Ref Load:         {ref_load or '(random weights)'}")
    print(f"Model Type:       {model_type}")
    print(f"TP Size:          {tp_size}")
    print(f"Seq Length:       {input_seq_len}")
    print(f"Prompt Mode:      {prompt_mode}")
    print(f"Chat Template:    {apply_chat_template}")
    print("=" * 60)

    # Load tokenizer to generate prompt
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)

    # Generate prompt
    if prompt_mode == "text" and prompt_text:
        prompt = prompt_text
        if postfix_question:
            prompt += "\n\nPlease answer the question above."
        if postfix_dummy_len > 0:
            prompt += "\n\nBelow, I will make some dummy text. Please ignore all text below. " + "1 " * postfix_dummy_len
    else:
        prompt = make_input(
            tokenizer=tokenizer,
            input_seq_len=input_seq_len,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
            prompt_file=prompt_file,
            apply_chat_template=False,  # Apply in worker script
            postfix_question=postfix_question,
            postfix_dummy_len=postfix_dummy_len,
        )

    print(f"Generated prompt ({len(prompt)} chars): {prompt[:100]}...")

    # Get model args by sourcing the shell script (like command_utils.py)
    model_script_path = get_model_script_path(model_type)
    print(f"Model script:     {model_script_path}")

    # Save prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file_path = f.name

    try:
        # Build the full command as a shell script
        # This sources the model script to get MODEL_ARGS, similar to command_utils.py
        script_path = SCRIPT_DIR / "run_megatron.py"
        
        extra_args = []
        if ref_load:
            extra_args.append(f'--ref-load "{ref_load}"')
        if apply_chat_template:
            extra_args.append("--apply-chat-template")
        extra_args_str = " ".join(extra_args)

        # Build shell command that sources model script and runs torchrun
        shell_cmd = f'''
source "{model_script_path}" && \\
PYTHONPATH="{MEGATRON_PATH}" \\
{sys.executable} -m torch.distributed.run \\
    --nproc-per-node={tp_size} \\
    "{script_path}" \\
    "${{MODEL_ARGS[@]}}" \\
    --prompt-file "{prompt_file_path}" \\
    --hf-checkpoint "{hf_checkpoint}" \\
    --seq-length {input_seq_len} \\
    --micro-batch-size {batch_size} \\
    --hidden-dropout 0 \\
    --attention-dropout 0 \\
    --top-k {top_k} \\
    --tp-size {tp_size} \\
    {extra_args_str}
'''
        print(f"\nRunning shell command...")
        result = subprocess.run(["bash", "-c", shell_cmd])

        if result.returncode != 0:
            raise RuntimeError(f"Megatron run failed with code {result.returncode}")

    finally:
        # Clean up temp file
        os.unlink(prompt_file_path)

    print("\nTest completed!")


@app.command("megatron-backward")
def megatron_backward(
    hf_checkpoint: str = typer.Option(..., "--hf-checkpoint", help="Path to HuggingFace checkpoint"),
    ref_load: Optional[str] = typer.Option(None, "--ref-load", help="Path to Megatron checkpoint"),
    input_seq_len: int = typer.Option(128, "--input-seq-len", help="Input sequence length"),
    tp_size: int = typer.Option(1, "--tp-size", help="Tensor parallel size"),
    prompt_mode: str = typer.Option("text", "--prompt-mode", help="Prompt mode: math, story, text"),
    prompt_text: Optional[str] = typer.Option("The capital of France is ", "--prompt-text", help="Custom prompt text (for text mode)"),
    prompt_file: Optional[str] = typer.Option(None, "--prompt-file", help="Path to prompt file (for story mode)"),
    apply_chat_template: bool = typer.Option(False, "--apply-chat-template", help="Apply chat template"),
    postfix_question: bool = typer.Option(False, "--postfix-question", help="Add postfix question"),
    postfix_dummy_len: int = typer.Option(0, "--postfix-dummy-len", help="Add dummy text of this length"),
    model_type: str = typer.Option("deepseek-v4-285B-5layer", "--model-type", help="Model type"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size"),
    routing_replay_dump_path: Optional[str] = typer.Option(None, "--routing-replay-dump-path", help="Path to save routing topk indices (for record mode)"),
    routing_replay_load_path: Optional[str] = typer.Option(None, "--routing-replay-load-path", help="Path to load routing topk indices (for replay mode)"),
):
    """Run Megatron model backward pass test with dummy loss."""
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Running Megatron Backward Pass Test")
    print("=" * 60)
    print(f"HF Checkpoint:    {hf_checkpoint}")
    print(f"Ref Load:         {ref_load or '(random weights)'}")
    print(f"Model Type:       {model_type}")
    print(f"TP Size:          {tp_size}")
    print(f"Seq Length:       {input_seq_len}")
    print(f"Prompt Mode:      {prompt_mode}")
    print(f"Chat Template:    {apply_chat_template}")
    if routing_replay_dump_path:
        print(f"Routing Replay:   DUMP to {routing_replay_dump_path}")
    elif routing_replay_load_path:
        print(f"Routing Replay:   LOAD from {routing_replay_load_path}")
    print("=" * 60)

    # Load tokenizer to generate prompt
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)

    # Generate prompt
    if prompt_mode == "text" and prompt_text:
        prompt = prompt_text
        if postfix_question:
            prompt += "\n\nPlease answer the question above."
        if postfix_dummy_len > 0:
            prompt += "\n\nBelow, I will make some dummy text. Please ignore all text below. " + "1 " * postfix_dummy_len
    else:
        prompt = make_input(
            tokenizer=tokenizer,
            input_seq_len=input_seq_len,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
            prompt_file=prompt_file,
            apply_chat_template=False,
            postfix_question=postfix_question,
            postfix_dummy_len=postfix_dummy_len,
        )

    print(f"Generated prompt ({len(prompt)} chars): {prompt[:100]}...")

    # Get model args by sourcing the shell script
    model_script_path = get_model_script_path(model_type)
    print(f"Model script:     {model_script_path}")

    # Save prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file_path = f.name

    try:
        script_path = SCRIPT_DIR / "run_megatron.py"

        extra_args = []
        if ref_load:
            extra_args.append(f'--ref-load "{ref_load}"')
        if apply_chat_template:
            extra_args.append("--apply-chat-template")
        extra_args_str = " ".join(extra_args)

        routing_replay_env = ""
        if routing_replay_dump_path:
            routing_replay_env = f'''ENABLE_ROUTING_REPLAY=1 \\
ROUTING_REPLAY_STAGE=record \\
ROUTING_REPLAY_DUMP_PATH="{routing_replay_dump_path}" \\
'''
        elif routing_replay_load_path:
            routing_replay_env = f'''ENABLE_ROUTING_REPLAY=1 \\
ROUTING_REPLAY_STAGE=replay_from_file \\
ROUTING_REPLAY_LOAD_PATH="{routing_replay_load_path}" \\
'''

        # Build shell command with backward pass flags
        shell_cmd = f'''
source "{model_script_path}" && \\
PYTHONPATH="{MEGATRON_PATH}" \\
SGLANG_DUMPER_DUMP_GRAD=1 \\
{routing_replay_env}{sys.executable} -m torch.distributed.run \\
    --nproc-per-node={tp_size} \\
    "{script_path}" \\
    "${{MODEL_ARGS[@]}}" \\
    --prompt-file "{prompt_file_path}" \\
    --hf-checkpoint "{hf_checkpoint}" \\
    --seq-length {input_seq_len} \\
    --micro-batch-size {batch_size} \\
    --hidden-dropout 0 \\
    --attention-dropout 0 \\
    --tp-size {tp_size} \\
    --run-backward \\
    --no-gradient-accumulation-fusion \\
    {extra_args_str}
'''
        print(f"\nRunning shell command...")
        result = subprocess.run(["bash", "-c", shell_cmd])

        if result.returncode != 0:
            raise RuntimeError(f"Megatron backward run failed with code {result.returncode}")

    finally:
        # Clean up temp file
        os.unlink(prompt_file_path)

    print("\nBackward pass test completed!")


@app.command()
def reference(
    ckpt_path: str = typer.Option(..., "--ckpt-path", help="Path to reference checkpoint (HF format with safetensors)"),
    config_path: Optional[str] = typer.Option(None, "--config-path", help="Path to model config JSON"),
    prompt_text: str = typer.Option("The capital of France is ", "--prompt-text", help="Prompt text"),
    input_seq_len: int = typer.Option(0, "--input-seq-len", help="Target input sequence length (0 = use prompt as-is)"),
    prompt_mode: str = typer.Option("text", "--prompt-mode", help="Prompt mode: math, story, text"),
    prompt_file: Optional[str] = typer.Option(None, "--prompt-file", help="Path to prompt file (for story mode)"),
    max_new_tokens: int = typer.Option(1, "--max-new-tokens", help="Max new tokens to generate"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
    tp_size: int = typer.Option(1, "--tp-size", help="Tensor parallel size"),
    top_k: int = typer.Option(5, "--top-k", help="Top-k predictions to show"),
    forward_only: bool = typer.Option(True, "--forward-only/--generate", help="Forward pass only (print logprobs) or generate tokens"),
    prefill_mode: bool = typer.Option(True, "--prefill-mode/--incremental-mode", help="Prefill mode (all tokens at once) or incremental mode"),
):
    """Run Reference implementation forward pass (for comparison with Megatron)."""
    from transformers import AutoTokenizer
    
    mode_str = "Forward Pass (logprobs)" if forward_only else "Generation"
    print("=" * 60)
    print(f"Running Reference Implementation - {mode_str}")
    print("=" * 60)
    print(f"Checkpoint:       {ckpt_path}")
    print(f"Config:           {config_path or REF_CONFIG_PATH}")
    print(f"Input Seq Len:    {input_seq_len}")
    print(f"Prompt Mode:      {prompt_mode}")
    print(f"Mode:             {mode_str}")
    if not forward_only:
        print(f"Max New Tokens:   {max_new_tokens}")
        print(f"Temperature:      {temperature}")
    print(f"Top-K:            {top_k}")
    print(f"TP Size:          {tp_size}")
    print("=" * 60)

    # Use default config if not provided
    if config_path is None:
        config_path = str(REF_CONFIG_PATH)

    # Generate prompt using make_input if input_seq_len > 0
    if input_seq_len > 0:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        prompt = make_input(
            tokenizer=tokenizer,
            input_seq_len=input_seq_len,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
            prompt_file=prompt_file,
            apply_chat_template=False,
        )
        print(f"Generated prompt ({len(prompt)} chars): {prompt[:100]}...")
    else:
        prompt = prompt_text

    # Use local run_reference.py in the same directory
    run_reference_script = SCRIPT_DIR / "run_reference.py"
    
    if not run_reference_script.exists():
        raise FileNotFoundError(f"run_reference.py not found at {run_reference_script}")

    env = os.environ.copy()
    env["NCCL_SOCKET_IFNAME"] = "lo"

    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nproc-per-node={tp_size}",
        str(run_reference_script),
        "--prompt", prompt,
        "--ckpt-path", ckpt_path,
        "--config-path", config_path,
        "--top-k", str(top_k),
        "--tp-size", str(tp_size),
    ]
    
    if forward_only:
        cmd.append("--forward-only")
        if prefill_mode:
            cmd.append("--prefill-mode")
    else:
        cmd.extend([
            "--max-new-tokens", str(max_new_tokens),
            "--temperature", str(temperature),
        ])

    print(f"\nRunning command: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Reference run failed with code {result.returncode}")

    print("\nReference run completed!")


@app.command()
def show_model_args(
    model_type: str = typer.Option("deepseek-v4-285B-5layer", "--model-type", help="Model type"),
):
    """Show model args for a given model type (by sourcing the shell script)."""
    model_args_str = get_model_args_from_shell(model_type)
    print(f"MODEL_ARGS from {model_type}.sh:")
    print(model_args_str)


if __name__ == "__main__":
    app()
