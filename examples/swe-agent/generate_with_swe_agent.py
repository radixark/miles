# Miles integration for SWE-Agent
# Minimal version: call Gym /run endpoint and return trajectory

from miles.utils.types import Sample
from miles.utils.http_utils import post
from miles.rollout.sglang_rollout import GenerateState


def build_tokens_and_mask_from_messages(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int], str, int]:

    if not messages or len(messages) < 2:
        return [], [], "", 0
    
    all_tokens = []
    loss_mask = []
    response_text = ""
    prompt_length = 0
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        
        msg_tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        all_tokens.extend(msg_tokens)
        
        if i < 2:
            # Prompt
            prompt_length += len(msg_tokens)
        else:
            # Response
            response_text += content
            if msg["role"] == "assistant":
                loss_mask.extend([1] * len(msg_tokens))
            else:
                loss_mask.extend([0] * len(msg_tokens))
    
    response_length = len(all_tokens) - prompt_length
    
    return all_tokens, loss_mask, response_text, response_length


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Custom generation function for SWE-Agent integration.
    Calls Gym /run endpoint with external sglang_url.
    """
    # Prepare request for Gym /run endpoint
    request = {
        "responses_create_params": {
            "temperature": sampling_params.get("temperature", 0.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "input": [],
        },
        **sample.metadata,
        "sglang_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1",
    }

    print("&&&& Before Gym /run endpoint &&&&")
    print(request)

    import os
    gym_url = os.getenv("SWE_AGENT_GYM_URL", "http://localhost:11000")
    response = await post(f"{gym_url}/run", request)
    
    # Get messages from response
    messages = response.get("messages", [])

    print("&&&& After Gym /run endpoint &&&&")
    print(f"messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        print(f"Message {i}: role={msg.get('role')}, content_len={len(content)}")
        if i < 3 or i >= len(messages) - 2:  # 打印前3条和后2条
            print(f"  content_preview: {content[:200]}")
    print(f"Total messages: {len(messages)}")
    
    # Build tokens and loss_mask from messages
    state = GenerateState(args)
    tokens, loss_mask, response_text, response_length = build_tokens_and_mask_from_messages(
        messages=messages,
        tokenizer=state.tokenizer,
    )
    
    sample.rollout_log_probs = None # TODO
    sample.tokens = tokens
    sample.loss_mask = loss_mask
    sample.response = response_text
    sample.response_length = response_length
    sample.status = Sample.Status.COMPLETED

    print(f"=" * 60)
    print(f"📊 Tokenization Debug Info:")
    print(f"  all_tokens length: {len(tokens)}")
    print(f"  loss_mask length: {len(loss_mask)}")
    print(f"  response_length (computed): {response_length}")
    print(f"  prompt_length (computed): {len(tokens) - response_length}")
    
    # ⚠️ 关键检查
    if len(loss_mask) != response_length:
        print(f"❌ CRITICAL BUG: loss_mask length ({len(loss_mask)}) != response_length ({response_length})")
        print(f"   This WILL cause NaN in log_probs calculation!")
        print(f"   Difference: {response_length - len(loss_mask)} tokens")
    else:
        print(f"✅ loss_mask length matches response_length")
    
    if len(loss_mask) > 0:
        avg_loss_mask = sum(loss_mask) / len(loss_mask)
        print(f"  Average loss mask: {avg_loss_mask:.4f}")
        print(f"  Total 1s in loss_mask: {sum(loss_mask)}")
    else:
        print(f"⚠️ WARNING: loss_mask is empty!")
    print(f"=" * 60)
    
    # Store metadata for reward
    sample.metadata["reward"] = response.get("reward", 0.0)
    sample.metadata["eval_report"] = response.get("metadata", {})
    sample.metadata["messages"] = messages # for test
    
    return sample


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward function - already computed in generate()"""
    reward = sample.metadata.get("reward", 0.0)
    return reward

