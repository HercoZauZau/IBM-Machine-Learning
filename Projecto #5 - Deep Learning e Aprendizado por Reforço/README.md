# Agente Inteligente para Ntxuva

Sistema completo em Python para jogar **Ntxuva** contra agentes de Inteligência Artificial, com foco num agente de **Aprendizado por Reforço (Deep Q-Network — DQN)** treinado por *self-play*.

<img src='../assets/capa v.png'>

---

<br/>

O projecto inclui:

- motor independente do jogo e das regras;
- interface gráfica em **Tkinter**;
- agente aleatório para baseline;
- agente **MiniMax com poda alpha-beta**, inspirado no trabalho de Ali, Gimo e Saide (2020);
- agente tabular Q-Learning para fins didácticos;
- agente DQN em PyTorch;
- treino DQN por self-play;
- mascaramento de acções inválidas;
- avaliação contra Random e MiniMax;
- testes automatizados;
- documentação das regras, arquitectura, RL e decisões de implementação.

## 1. Referência principal

A principal referência académica do projecto é:

> Ali, F. D. M. A., Gimo, E., & Saide, S. M. (2020). *A MiniMax Agent for Playing Ntxuva Game – The Mozambican Variant of Mancala*. 2020 International Conference on Artificial Intelligence, Big Data, Computing and Data Communication Systems (icABCD). DOI: 10.1109/icABCD49160.2020.9183848.

O artigo é utilizado como referência para o contexto computacional do Ntxuva e para o baseline MiniMax. A formalização operacional das regras foi ainda confrontada com descrições públicas do Ntxuva.

## 2. Estrutura

```text
ntxuva_ai/
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── ntxuva/
│   ├── game/
│   │   ├── game.py
│   │   ├── move.py
│   │   └── action_codec.py
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── minimax_agent.py
│   │   ├── qlearning_agent.py
│   │   └── dqn_agent.py
│   ├── rl/
│   │   ├── network.py
│   │   ├── replay_buffer.py
│   │   ├── trainer.py
│   │   └── evaluation.py
│   └── gui/
│       └── app.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── play_cli.py
│   └── plot_training.py
├── results/
└── tests/
```

## 3. Requisitos

- Python 3.10 ou superior;
- Tkinter;
- PyTorch;
- NumPy;
- Matplotlib (apenas para gráficos de treino);
- pytest.

Em Ubuntu/Debian, se Tkinter não estiver instalado:

```bash
sudo apt install python3-tk
```

Criar ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Jogar

```bash
python main.py
```

A aplicação abre uma janela Tkinter. O jogador humano ocupa as duas filas inferiores e pode escolher entre:

- DQN;
- MiniMax;
- Random.

Se não existir ainda um checkpoint DQN em `models/ntxuva_dqn_best.pt`, a opção DQN usa MiniMax como fallback para que a interface continue jogável.

Este pacote inclui um **checkpoint DQN de demonstração**. Numa avaliação curta, obteve 67,5% de vitórias em 40 partidas contra Random, mas 0/12 contra MiniMax depth=1. O objectivo do checkpoint é tornar a aplicação imediatamente testável; para resultados finais deve-se treinar e avaliar o agente em experiências maiores. Ver `models/MODEL_CARD.md`.

## 5. Treinar o agente DQN

Treino padrão:

```bash
python scripts/train.py --episodes 10000
```

Exemplo curto para validar o pipeline:

```bash
python scripts/train.py --episodes 500
```

O treino guarda:

```text
results/training/
├── training_history.csv
└── models/
    ├── ntxuva_dqn_best.pt
    ├── ntxuva_dqn_final.pt
    └── ntxuva_dqn_epXXXX.pt
```

O script também copia o melhor/final checkpoint para:

```text
models/ntxuva_dqn_best.pt
```

## 6. Avaliar

Contra agente aleatório:

```bash
python scripts/evaluate.py --games 100 --opponent random
```

Contra MiniMax:

```bash
python scripts/evaluate.py --games 50 --opponent minimax --depth 2
```

Métricas apresentadas:

- vitórias;
- derrotas;
- empates;
- win rate;
- duração média da partida;
- capturas médias.

## 7. Visualizar o treino

```bash
python scripts/plot_training.py results/training/training_history.csv
```

Os gráficos são guardados em `results/training/plots/`.

## 8. Testes

```bash
pytest -q
```

Os testes verificam, entre outros pontos:

- configuração inicial;
- conservação do total de sementes;
- geração de jogadas válidas;
- regra gula/tchonga;
- relay sowing;
- captura;
- codec de acções;
- interface comum dos agentes;
- execução de uma partida completa.

## 9. Modelo de Aprendizado por Reforço

O estado é representado da perspectiva do jogador da vez:

```text
16 casas próprias
+ 16 casas adversárias
+ sementes capturadas pelo jogador
+ sementes capturadas pelo adversário
= 34 entradas
```

A rede DQN utiliza duas camadas escondidas com ReLU e produz um Q-value para cada acção codificada.

### Espaço de acções

Uma jogada pode incluir a origem e, se houver captura, até duas casas adicionais do adversário. Para 16 casas por jogador:

```text
16 × [1 + 16 + C(16,2)] = 2192 acções possíveis
```

A maior parte é inválida num dado estado; por isso o sistema aplica **action masking**.

### Reward

Por omissão:

```text
vitória  = +1
empate   =  0
derrota  = -1
jogada normal = 0
```

O treino suporta opcionalmente recompensa pequena por captura através de `--capture-reward`, mas o padrão utiliza recompensa terminal esparsa.

### Self-play

A mesma rede representa os dois jogadores. Como o próximo estado é visto da perspectiva do adversário, a actualização usa a forma zero-sum:

```text
target = reward - gamma × max Q(next_state)
```

Isto é detalhado em `docs/APRENDIZADO_REFORCO.md`.

## 10. Limitações

O Ntxuva possui variantes regionais. Este projecto fixa uma **variante computacional explícita**, para que os resultados sejam reproduzíveis. 


O DQN incluído no código é uma arquitectura de investigação/ensino; a força final depende do número de episódios, hiperparâmetros e qualidade da avaliação contra adversários externos.

## 11. Próximos desenvolvimentos

- Double DQN;
- Dueling DQN;
- Prioritized Experience Replay;
- curriculum learning: Random → MiniMax → self-play;
- comparação DQN vs PPO/MCTS;
- análise de vantagem do primeiro jogador;
- aprendizagem das opções de captura com arquitectura hierárquica;
- validação etnográfica das regras com jogadores moçambicanos de diferentes regiões.

