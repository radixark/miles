def add_rlve_arguments(parser):
    """Add RLVE (Reinforcement Learning with Verifiable Environments) arguments."""
    parser.add_argument(
        "--rlve",
        action="store_true",
        default=False,
        help="Enable RLVE training mode with procedurally generated verifiable environments.",
    )
    parser.add_argument(
        "--environment-list",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of verifiable environments to train on. "
            "Accepts multiple string values (e.g. --environment-list Sorting Fibonacci). "
            "Required when --rlve is True."
        ),
    )
    parser.add_argument(
        "--initial-difficulty",
        type=int,
        default=0,
        help="Initial difficulty (upper bound) for each environment.",
    )
    parser.add_argument(
        "--difficulty-sliding-window-size",
        type=int,
        default=4,
        help="Size of the sliding window for problem difficulty sampling.",
    )
    parser.add_argument(
        "--min-metric-to-increase-difficulty",
        type=float,
        default=0.9,
        help="When accuracy exceeds this value, difficulty will be increased.",
    )
    parser.add_argument(
        "--min-prompts-before-difficulty-check",
        type=int,
        default=8,
        help="Minimum number of prompts before performing a difficulty check.",
    )
    parser.add_argument(
        "--answer-marker-type",
        type=str,
        default=r"\boxed{}",
        choices=[r"\boxed{}", r"<answer></answer>"],
        help="The type of answer marker to use for extracting answers.",
    )
    parser.add_argument(
        "--custom-prompt-preprocessor",
        type=str,
        default=None,
        choices=["TinyZero", "ChatTemplate_NoSystemPrompt"],
        help="Choose a custom prompt preprocessor for RLVE.",
    )
    return parser
