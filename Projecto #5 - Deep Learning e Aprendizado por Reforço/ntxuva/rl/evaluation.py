from __future__ import annotations

from dataclasses import dataclass

from ntxuva.game.game import NtxuvaGame


@dataclass
class EvaluationResult:
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_moves: float
    avg_captured_for: float
    avg_captured_against: float


def play_match(agent_a, agent_b, max_moves: int = 400) -> tuple[int | None, NtxuvaGame]:
    game = NtxuvaGame()
    agents = [agent_a, agent_b]
    while not game.terminal and game.move_count < max_moves:
        agent = agents[game.current_player]
        move = agent.select_move(game)
        game.apply_move(move)
    if not game.terminal:
        # Empate operacional por limite de plies.
        game.terminal = True
        game.draw = True
        game.winner = None
    return game.winner, game


def evaluate(agent, opponent_factory, games: int = 100, max_moves: int = 400) -> EvaluationResult:
    wins = losses = draws = total_moves = 0
    cap_for = cap_against = 0

    for i in range(games):
        # Alterna a posição inicial para reduzir viés do primeiro jogador.
        if i % 2 == 0:
            a0, a1 = agent, opponent_factory()
            agent_player = 0
        else:
            a0, a1 = opponent_factory(), agent
            agent_player = 1
        winner, game = play_match(a0, a1, max_moves=max_moves)
        total_moves += game.move_count
        cap_for += int(game.captured[agent_player])
        cap_against += int(game.captured[1 - agent_player])
        if winner is None:
            draws += 1
        elif winner == agent_player:
            wins += 1
        else:
            losses += 1

    return EvaluationResult(
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / games if games else 0.0,
        avg_moves=total_moves / games if games else 0.0,
        avg_captured_for=cap_for / games if games else 0.0,
        avg_captured_against=cap_against / games if games else 0.0,
    )
