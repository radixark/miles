"""Exercise the chess recipe's real Stockfish load envelope."""

import asyncio
import json
import subprocess
import time
from dataclasses import asdict, dataclass

import chess
import chess.engine
from tap import Tap

import chess_agent


class Arguments(Tap):
    """Load-test two real Stockfish processes per simulated chess game."""

    stockfish_path: str = "/usr/games/stockfish"
    stockfish_timeout_seconds: float = 20.0
    num_games: int = 64
    max_concurrent_games: int = 16
    move_time_seconds: float = 0.2


@dataclass(frozen=True)
class ProbeResult:
    num_games: int
    max_concurrent_games: int
    max_live_engines: int
    failures: int
    remaining_stockfish_processes: int
    elapsed_seconds: float


def _stockfish_process_count() -> int:
    result = subprocess.run(
        ["pgrep", "-cx", "stockfish"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 1:
        return 0
    if result.returncode != 0:
        raise RuntimeError(f"pgrep failed with exit code {result.returncode}")
    return int(result.stdout.strip())


async def run_probe(arguments: Arguments) -> ProbeResult:
    """Start, exercise, and clean up two engines for every simulated game."""

    if arguments.stockfish_timeout_seconds <= 0:
        raise ValueError("stockfish_timeout_seconds must be positive")
    if arguments.num_games < 1:
        raise ValueError("num_games must be at least 1")
    if arguments.max_concurrent_games < 1:
        raise ValueError("max_concurrent_games must be at least 1")
    if arguments.move_time_seconds <= 0:
        raise ValueError("move_time_seconds must be positive")
    initial_processes = _stockfish_process_count()
    if initial_processes:
        raise RuntimeError(f"refusing to run while {initial_processes} Stockfish processes already exist")

    live_engines = 0
    max_live_engines = 0
    state_lock = asyncio.Lock()
    started_at = time.monotonic()
    limiter = chess_agent._game_limiter(arguments.max_concurrent_games)

    async def update_live(delta: int) -> None:
        nonlocal live_engines, max_live_engines
        async with state_lock:
            live_engines += delta
            max_live_engines = max(max_live_engines, live_engines)

    async def one_game() -> None:
        opponent: chess.engine.SimpleEngine | None = None
        reviewer: chess.engine.SimpleEngine | None = None
        async with limiter:
            try:
                opponent = await asyncio.to_thread(
                    chess.engine.SimpleEngine.popen_uci,
                    arguments.stockfish_path,
                    timeout=arguments.stockfish_timeout_seconds,
                )
                await update_live(1)
                reviewer = await asyncio.to_thread(
                    chess.engine.SimpleEngine.popen_uci,
                    arguments.stockfish_path,
                    timeout=arguments.stockfish_timeout_seconds,
                )
                await update_live(1)
                await asyncio.to_thread(
                    opponent.configure,
                    {
                        "UCI_LimitStrength": True,
                        "UCI_Elo": 1320,
                        "Threads": 1,
                        "Hash": 64,
                    },
                )
                await asyncio.to_thread(
                    reviewer.configure,
                    {"Threads": 1, "Hash": 64},
                )
                board = chess.Board()
                played = await asyncio.to_thread(
                    opponent.play,
                    board,
                    chess.engine.Limit(time=arguments.move_time_seconds),
                )
                if played.move is None:
                    raise RuntimeError("Stockfish returned no move")
                board.push(played.move)
                await asyncio.to_thread(
                    reviewer.analyse,
                    board,
                    chess.engine.Limit(time=arguments.move_time_seconds),
                )
            finally:
                engines = [engine for engine in (reviewer, opponent) if engine is not None]
                cleanup_results = await asyncio.gather(
                    *(asyncio.to_thread(engine.quit) for engine in engines),
                    return_exceptions=True,
                )
                await update_live(-len(engines))
                cleanup_failures = [result for result in cleanup_results if isinstance(result, BaseException)]
                if cleanup_failures:
                    raise RuntimeError(f"{len(cleanup_failures)} Stockfish processes failed to quit cleanly") from cleanup_failures[0]

    outcomes = await asyncio.gather(
        *(one_game() for _ in range(arguments.num_games)),
        return_exceptions=True,
    )
    failures = sum(isinstance(outcome, BaseException) for outcome in outcomes)
    elapsed_seconds = time.monotonic() - started_at
    remaining_processes = _stockfish_process_count()
    return ProbeResult(
        num_games=arguments.num_games,
        max_concurrent_games=arguments.max_concurrent_games,
        max_live_engines=max_live_engines,
        failures=failures,
        remaining_stockfish_processes=remaining_processes,
        elapsed_seconds=round(elapsed_seconds, 3),
    )


def main() -> None:
    arguments = Arguments().parse_args()
    result = asyncio.run(run_probe(arguments))
    print(json.dumps(asdict(result), sort_keys=True))
    if result.failures or result.remaining_stockfish_processes:
        raise SystemExit(1)
    expected_limit = 2 * arguments.max_concurrent_games
    if result.max_live_engines > expected_limit:
        raise RuntimeError(f"observed {result.max_live_engines} engines; expected at most {expected_limit}")


if __name__ == "__main__":
    main()
