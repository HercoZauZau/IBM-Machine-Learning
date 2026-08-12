from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from ntxuva.agents.dqn_agent import DQNAgent
from ntxuva.agents.minimax_agent import MinimaxAgent
from ntxuva.agents.random_agent import RandomAgent
from ntxuva.game.game import NtxuvaGame
from ntxuva.game.move import Move


class NtxuvaApp(tk.Tk):
    BOARD_BG = "#8b5a2b"
    PIT_BG = "#d7a86e"
    PIT_ACTIVE = "#f7d08a"
    PIT_SELECT = "#9bd3ae"
    TEXT = "#211a14"

    def __init__(self, model_path: str | Path | None = None):
        super().__init__()
        self.title("Ntxuva AI — Aprendizado por Reforço")
        self.resizable(False, False)
        self.game = NtxuvaGame(columns=8)
        self.human_player = 0
        self.model_path = Path(model_path) if model_path else None
        self.pending_origin: int | None = None
        self.pending_actions: list[Move] = []
        self.selected_extras: list[int] = []
        self.agent = self._make_agent("DQN")

        self._build_ui()
        self._draw_board()
        self._update_status("Sua vez. Seleccione uma casa.")

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="NTXUVA AI", font=("TkDefaultFont", 18, "bold")).pack(side="left")

        ttk.Label(header, text="Adversário:").pack(side="left", padx=(25, 5))
        self.agent_var = tk.StringVar(value="DQN")
        combo = ttk.Combobox(
            header,
            textvariable=self.agent_var,
            state="readonly",
            width=12,
            values=("DQN", "MiniMax", "Random"),
        )
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._on_agent_changed)

        ttk.Button(header, text="Novo jogo", command=self.new_game).pack(side="right", padx=4)
        ttk.Button(header, text="Regras", command=self.show_rules).pack(side="right", padx=4)

        info = ttk.Frame(self, padding=(12, 0, 12, 8))
        info.pack(fill="x")
        self.score_label = ttk.Label(info, text="")
        self.score_label.pack(side="left")
        self.turn_label = ttk.Label(info, text="")
        self.turn_label.pack(side="right")

        self.canvas = tk.Canvas(self, width=900, height=430, bg=self.BOARD_BG, highlightthickness=0)
        self.canvas.pack(padx=12, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.status = ttk.Label(self, text="", anchor="center", padding=10, font=("TkDefaultFont", 11, "bold"))
        self.status.pack(fill="x")

        foot = ttk.Label(
            self,
            text="Jogador: filas inferiores • Agente: filas superiores • clique numa casa válida para jogar",
            anchor="center",
            padding=(8, 0, 8, 10),
        )
        foot.pack(fill="x")

    def _make_agent(self, name: str):
        if name == "Random":
            return RandomAgent()
        if name == "MiniMax":
            return MinimaxAgent(depth=2, max_children=28)
        # DQN
        model = self.model_path
        if model and model.exists():
            return DQNAgent(self.game.state_size, self.game.codec.action_size, model_path=model)
        # Sem checkpoint treinado, MiniMax é fallback seguro; a interface continua funcional.
        return MinimaxAgent(depth=2, max_children=28)

    def _on_agent_changed(self, _event=None) -> None:
        requested = self.agent_var.get()
        self.agent = self._make_agent(requested)
        if requested == "DQN" and not (self.model_path and self.model_path.exists()):
            messagebox.showinfo(
                "Modelo DQN",
                "Não foi encontrado um checkpoint DQN treinado.\n"
                "A interface usará MiniMax até treinar/copiar um modelo para models/ntxuva_dqn_best.pt.",
            )
        self.new_game()

    def new_game(self) -> None:
        self.game.reset()
        self.pending_origin = None
        self.pending_actions = []
        self.selected_extras = []
        self._draw_board()
        self._update_status("Sua vez. Seleccione uma casa.")

    def _pit_geometry(self, row: int, col: int) -> tuple[float, float, float]:
        margin_x = 70
        step_x = 105
        y_positions = [70, 165, 265, 360]
        x = margin_x + col * step_x
        y = y_positions[row]
        radius = 34
        return x, y, radius

    def _draw_board(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(450, 22, text="AGENTE", fill="white", font=("TkDefaultFont", 12, "bold"))
        self.canvas.create_text(450, 408, text="JOGADOR", fill="white", font=("TkDefaultFont", 12, "bold"))

        legal_origins = {m.origin for m in self.game.legal_moves()} if self.game.current_player == self.human_player and not self.game.terminal else set()
        eligible_extra_globals: set[tuple[int, int]] = set()
        if self.pending_actions:
            for move in self.pending_actions:
                for local in move.extra_captures:
                    if local not in self.selected_extras:
                        eligible_extra_globals.add(self.game.local_to_global(1 - self.human_player, local))

        for row in range(4):
            for col in range(self.game.columns):
                x, y, r = self._pit_geometry(row, col)
                fill = self.PIT_BG
                if self.game.current_player == self.human_player and row in self.game.rows_for_player(self.human_player):
                    try:
                        local = self.game.global_to_local(self.human_player, row, col)
                        if local in legal_origins:
                            fill = self.PIT_ACTIVE
                    except ValueError:
                        pass
                if (row, col) in eligible_extra_globals:
                    fill = self.PIT_SELECT
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill, outline="#4b2e18", width=3)
                n = int(self.game.board[row, col])
                self.canvas.create_text(x, y, text=str(n), fill=self.TEXT, font=("TkDefaultFont", 13, "bold"))

        self.score_label.config(text=f"Capturas — Jogador: {int(self.game.captured[0])} | Agente: {int(self.game.captured[1])}")
        turn = "Jogador" if self.game.current_player == 0 else "Agente"
        self.turn_label.config(text=f"Jogada {self.game.move_count + 1} • Vez: {turn}")

    def _hit_test(self, event) -> tuple[int, int] | None:
        for row in range(4):
            for col in range(self.game.columns):
                x, y, r = self._pit_geometry(row, col)
                if (event.x - x) ** 2 + (event.y - y) ** 2 <= r ** 2:
                    return row, col
        return None

    def _on_canvas_click(self, event) -> None:
        if self.game.terminal or self.game.current_player != self.human_player:
            return
        hit = self._hit_test(event)
        if hit is None:
            return
        row, col = hit

        if self.pending_actions:
            self._handle_capture_choice(row, col)
            return

        if row not in self.game.rows_for_player(self.human_player):
            return
        try:
            origin = self.game.global_to_local(self.human_player, row, col)
        except ValueError:
            return
        actions = [m for m in self.game.legal_moves() if m.origin == origin]
        if not actions:
            self._update_status("Essa casa não pode iniciar uma jogada.")
            return
        if len(actions) == 1:
            self._play_human_move(actions[0])
            return

        self.pending_origin = origin
        self.pending_actions = actions
        self.selected_extras = []
        self._update_status("Captura activada: seleccione duas casas adicionais do adversário.")
        self._draw_board()

    def _handle_capture_choice(self, row: int, col: int) -> None:
        if row not in self.game.rows_for_player(1 - self.human_player):
            return
        try:
            local = self.game.global_to_local(1 - self.human_player, row, col)
        except ValueError:
            return
        possible = {x for m in self.pending_actions for x in m.extra_captures}
        if local not in possible or local in self.selected_extras:
            return
        self.selected_extras.append(local)

        matches = [m for m in self.pending_actions if tuple(sorted(m.extra_captures)) == tuple(sorted(self.selected_extras))]
        if matches:
            move = matches[0]
            self.pending_actions = []
            self.pending_origin = None
            self.selected_extras = []
            self._play_human_move(move)
            return

        # Ainda falta uma segunda casa.
        self._update_status("Seleccione a segunda casa adicional a capturar.")
        self._draw_board()

    def _play_human_move(self, move: Move) -> None:
        try:
            result = self.game.apply_move(move)
        except ValueError as exc:
            self._update_status(str(exc))
            return
        self._draw_board()
        if self._check_game_over():
            return
        msg = f"Capturou {result.captured} sementes. " if result.captured else ""
        self._update_status(msg + "O agente está a jogar...")
        self.after(250, self._agent_turn)

    def _agent_turn(self) -> None:
        if self.game.terminal or self.game.current_player == self.human_player:
            return
        try:
            move = self.agent.select_move(self.game)
            result = self.game.apply_move(move)
        except Exception as exc:  # mantém a GUI recuperável
            messagebox.showerror("Erro do agente", str(exc))
            return
        self._draw_board()
        if self._check_game_over():
            return
        msg = f"O agente capturou {result.captured} sementes. " if result.captured else ""
        self._update_status(msg + "Sua vez.")

    def _check_game_over(self) -> bool:
        if not self.game.terminal:
            return False
        if self.game.winner is None:
            msg = "Empate."
        elif self.game.winner == self.human_player:
            msg = "Vitória!"
        else:
            msg = "O agente venceu."
        self._update_status(msg)
        messagebox.showinfo(
            "Fim da partida",
            f"{msg}\nCapturas: Jogador {int(self.game.captured[0])} × {int(self.game.captured[1])} Agente",
        )
        return True

    def _update_status(self, text: str) -> None:
        self.status.config(text=text)

    def show_rules(self) -> None:
        top = tk.Toplevel(self)
        top.title("Regras — Ntxuva")
        top.geometry("720x600")
        text = tk.Text(top, wrap="word", padx=18, pady=18)
        text.pack(fill="both", expand=True)
        rules = """REGRAS DA VARIANTE COMPUTACIONAL

1. O tabuleiro possui quatro filas de oito casas. Cada jogador controla duas filas.
2. Cada casa começa com duas sementes.
3. As jogadas seguem o sentido anti-horário nas duas filas do próprio jogador.
4. Enquanto houver uma casa com duas ou mais sementes (gula), a jogada deve começar numa dessas casas.
5. Se não houver gula, uma casa com uma única semente (tchonga) só pode ser usada quando a casa seguinte está vazia.
6. Retiram-se todas as sementes da casa escolhida e distribuem-se uma a uma.
7. Se a última semente cair numa casa que já continha sementes, recolhe-se o conteúdo dessa casa e a distribuição continua.
8. A jogada termina quando a última semente cai numa casa que estava vazia.
9. Se essa casa vazia estiver na fila interna e a casa interna directamente oposta do adversário possuir sementes, ocorre captura.
10. São capturadas as sementes das casas opostas da mesma coluna e, quando existem, sementes de duas outras casas do adversário escolhidas pelo jogador.
11. A partida termina quando o jogador da vez não dispõe de jogada válida. Vence quem capturou mais sementes.

A documentação em docs/REGRAS.md descreve as fontes, decisões de implementação e limitações desta formalização."""
        text.insert("1.0", rules)
        text.config(state="disabled")


def run_gui(model_path: str | Path | None = None) -> None:
    app = NtxuvaApp(model_path=model_path)
    app.mainloop()
