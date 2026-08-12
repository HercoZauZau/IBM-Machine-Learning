from ntxuva.agents.minimax_agent import MinimaxAgent
from ntxuva.agents.random_agent import RandomAgent
from ntxuva.game.game import NtxuvaGame


def test_random_agent_returns_legal_move():
    game = NtxuvaGame()
    agent = RandomAgent(seed=1)
    move = agent.select_move(game)
    assert move in game.legal_moves()


def test_minimax_agent_returns_legal_move():
    game = NtxuvaGame()
    agent = MinimaxAgent(depth=1, max_children=12)
    move = agent.select_move(game)
    assert move in game.legal_moves()


def test_random_game_terminates_and_conserves_seeds():
    game = NtxuvaGame()
    agents = [RandomAgent(10), RandomAgent(20)]
    while not game.terminal and game.move_count < 200:
        game.apply_move(agents[game.current_player].select_move(game))
    assert game.terminal
    assert int(game.board.sum() + game.captured.sum()) == 64
    assert game.move_count < 200
