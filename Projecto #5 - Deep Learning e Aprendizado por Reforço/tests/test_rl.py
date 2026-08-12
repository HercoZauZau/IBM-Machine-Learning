import torch

from ntxuva.agents.dqn_agent import DQNAgent
from ntxuva.game.game import NtxuvaGame
from ntxuva.rl.network import QNetwork


def test_network_output_size():
    game = NtxuvaGame()
    net = QNetwork(game.state_size, game.codec.action_size, hidden_size=32)
    x = torch.zeros((2, game.state_size))
    y = net(x)
    assert y.shape == (2, game.codec.action_size)


def test_untrained_dqn_respects_action_mask():
    game = NtxuvaGame()
    agent = DQNAgent(game.state_size, game.codec.action_size, hidden_size=32, device="cpu")
    action = agent.select_action(game)
    assert action in game.legal_action_ids()
