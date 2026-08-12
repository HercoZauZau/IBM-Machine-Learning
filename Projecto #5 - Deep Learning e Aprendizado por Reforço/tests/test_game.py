import numpy as np

from ntxuva.game.action_codec import ActionCodec
from ntxuva.game.game import NtxuvaGame
from ntxuva.game.move import Move


def test_initial_state():
    game = NtxuvaGame()
    assert game.board.shape == (4, 8)
    assert np.all(game.board == 2)
    assert int(game.board.sum()) == 64
    assert game.captured.tolist() == [0, 0]
    assert game.current_player == 0
    assert not game.terminal


def test_seed_conservation_after_move():
    game = NtxuvaGame()
    before = int(game.board.sum() + game.captured.sum())
    move = game.legal_moves()[0]
    game.apply_move(move)
    after = int(game.board.sum() + game.captured.sum())
    assert before == after == 64


def test_all_initial_pits_with_two_are_candidate_origins():
    game = NtxuvaGame()
    origins = {m.origin for m in game.legal_moves()}
    assert origins == set(range(16))


def test_singleton_rule_only_when_no_gula_and_next_empty():
    game = NtxuvaGame()
    game.board[:] = 0
    path = game.sow_path(0)
    r0, c0 = path[0]
    r1, c1 = path[1]
    game.board[r0, c0] = 1
    game.board[r1, c1] = 0
    moves = game.legal_moves(0)
    assert {m.origin for m in moves} == {0}

    for rr, cc in path:
        game.board[rr, cc] = 1
    assert game.legal_moves(0) == []


def test_gula_blocks_singletons():
    game = NtxuvaGame()
    game.board[:] = 0
    path = game.sow_path(0)
    r0, c0 = path[0]
    r1, c1 = path[1]
    r2, c2 = path[2]
    game.board[r0, c0] = 1
    game.board[r1, c1] = 0
    game.board[r2, c2] = 2
    origins = {m.origin for m in game.legal_moves(0)}
    assert origins == {2}


def test_capture_is_generated_with_extra_choices():
    game = NtxuvaGame()
    game.board[:] = 0
    # Constrói uma jogada do jogador 0 que termina vazia na fila interna.
    # Escolhe uma origem cujo próximo destino seja inner row.
    path = game.sow_path(0)
    inner, _ = game.rows_for_player(0)
    origin = None
    landing = None
    for i, (r, c) in enumerate(path):
        nr, nc = path[(i + 1) % len(path)]
        if nr == inner:
            origin = i
            landing = (nr, nc)
            break
    assert origin is not None and landing is not None
    r, c = path[origin]
    game.board[r, c] = 1
    # Casa de destino vazia; oponente interno oposto com sementes.
    opp_inner, opp_outer = game.rows_for_player(1)
    game.board[opp_inner, landing[1]] = 3
    game.board[opp_outer, landing[1]] = 2
    # Duas casas adicionais para captura.
    extras_globals = []
    for rr, cc in game.sow_path(1):
        if cc != landing[1] and len(extras_globals) < 2:
            game.board[rr, cc] = 4
            extras_globals.append((rr, cc))

    total_before = int(game.board.sum() + game.captured.sum())
    moves = [m for m in game.legal_moves(0) if m.origin == origin]
    assert moves
    # Com exactamente duas casas extra, deve existir uma única combinação.
    assert len(moves) == 1
    result = game.apply_move(moves[0])
    assert result.captured == 3 + 2 + 4 + 4
    assert int(game.board.sum() + game.captured.sum()) == total_before


def test_action_codec_round_trip():
    codec = ActionCodec(16)
    for move in [Move(0, ()), Move(3, (5,)), Move(15, (2, 14))]:
        assert codec.decode(codec.encode(move)) == Move(move.origin, tuple(sorted(move.extra_captures)))


def test_state_size_and_mask():
    game = NtxuvaGame()
    state = game.encode_state()
    mask = game.action_mask()
    assert state.shape == (34,)
    assert mask.shape == (2192,)
    assert mask.sum() == len(game.legal_moves())
