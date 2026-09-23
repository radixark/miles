"""Build the Tinker gateway with recorded-session components and HTTP routes."""

from fastapi import FastAPI

from miles.tinker.core.prompt_renderer import PromptRenderer
from miles.tinker.core.service import TinkerService
from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.server.app import build_app
from miles.tinker.server.session_routes import setup_session_routes
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.chat_template_utils.message_matcher_hub import strict_message_matches
from miles.utils.processing_utils import load_tokenizer


def build_session_app(service: TinkerService, *, args) -> tuple[FastAPI, TrajectoryCollector]:
    """Build the SDK app plus a collector and session routes; the caller owns background tasks."""
    app = build_app(service)
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path)
    tito_tokenizer = None
    if args.tinker_tito_model is not None:
        tito_tokenizer = get_tito_tokenizer(
            tokenizer, args.tinker_tito_model, chat_template_kwargs=args.apply_chat_template_kwargs
        )
    renderer = PromptRenderer(
        tokenizer,
        args.apply_chat_template_kwargs,
        tito_tokenizer=tito_tokenizer,
        message_matcher=strict_message_matches,
    )
    collector = TrajectoryCollector(
        service,
        renderer,
        session_ttl_s=args.tinker_session_ttl_s,
        strict_truncation=args.tinker_session_strict_truncation,
    )
    setup_session_routes(app, collector, max_body_bytes=args.tinker_session_max_body_bytes)
    return app, collector
