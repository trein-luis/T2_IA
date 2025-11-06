import random
import numpy as np

# Ajuste estes imports para o que você realmente tem no projeto:
from NeuralNetwork import NeuralNetwork  # NÃO use alias nn para evitar sombra
# Ex.: from MiniMax import MinimaxPlayer, TicTacToe
from MiniMax import MinimaxPlayer
from Game import TicTacToe
 # ajuste se os nomes forem outros


class GeneticAlgorithm:
    """
    Algoritmo Genético para evolução dos pesos da rede neural.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): parâmetros do AG:
              - population_size (int)
              - tournament_size (int)
              - mutation_rate (float)
              - elitism_rate (float, 0..1)
              - convergence_threshold (int)
              - medium_mode_generations (int)
        """
        self.config = config
        self.population: list[np.ndarray] = []
        self.generation = 0
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.convergence_counter = 0

        # Opcional: RNG dedicado (melhor p/ reproducibilidade)
        self.rng = np.random.default_rng()

        # Cache opcional de fitness da geração atual
        self._last_fitness_scores: list[float] | None = None

    def _weights_dim(self) -> int:
        """
        Determina o número total de pesos da rede.
        É esperado que NeuralNetwork exponha 'total_weights' sem precisar de pesos.
        """
        net = NeuralNetwork()
        if not hasattr(net, "total_weights"):
            raise AttributeError(
                "NeuralNetwork precisa expor 'total_weights' para inicializar população."
            )
        return int(net.total_weights)

    def initialize_population(self):
        """Inicializa a população com indivíduos aleatórios (np.ndarray)."""
        dim = self._weights_dim()
        self.population = [
            self.rng.normal(loc=0.0, scale=0.5, size=dim).astype(np.float32)
            for _ in range(self.config["population_size"])
        ]
        self._last_fitness_scores = None  # limpa cache

    def evaluate_fitness(self, weights: np.ndarray, generation: int) -> float:
        """
        Avalia o fitness de um indivíduo jogando contra o Minimax.

        Penaliza jogadas inválidas. Retorna a média em N partidas.
        """
        net = NeuralNetwork(weights)

        difficulty = "medium" if generation < self.config["medium_mode_generations"] else "hard"
        minimax = MinimaxPlayer(difficulty)

        fitness = 0.0
        num_games = 10  # pode virar config se quiser

        for _ in range(num_games):
            game = TicTacToe()
            invalid_moves = 0

            while not game.is_game_over():
                if game.current_player == "X":  # IA sempre X
                    move = net.predict(game.board)
                    # Garante que seja um índice inteiro aceitável
                    if isinstance(move, (list, np.ndarray)):
                        # se a rede retorna distribuição/valores, escolha argmax
                        move = int(np.argmax(move))
                    else:
                        move = int(move)

                    if not game.is_valid_move(move):
                        invalid_moves += 1
                        valid = [i for i, c in enumerate(game.board) if c == ""]
                        move = random.choice(valid) if valid else 0

                    game.make_move(move)
                else:
                    move = minimax.get_move(game.board, "O")
                    game.make_move(move)

            # resultado do ponto de vista de X
            result = game.get_result("X")
            if result == "win":
                fitness += 10
            elif result == "draw":
                fitness += 0
            elif result == "loss":
                fitness -= 10

            # penalização de jogadas inválidas
            fitness -= invalid_moves * 5

        return float(fitness / num_games)

    def _safe_tournament_size(self) -> int:
        return max(2, min(self.config["tournament_size"], len(self.population)))

    def tournament_selection(self, fitness_scores: list[float]) -> np.ndarray:
        """Seleção por torneio (retorna CÓPIA do indivíduo)."""
        k = self._safe_tournament_size()
        indices = random.sample(range(len(fitness_scores)), k)
        winner_idx = max(indices, key=lambda i: fitness_scores[i])
        # retorna cópia para não vazar referência do vetor no crossover/mutate
        return self.population[winner_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crossover aritmético para valores reais (alfa ~ U[0,1])."""
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent2 + (1 - (1 - alpha)) * parent1  # equivalente ao acima
        return child1.astype(np.float32), child2.astype(np.float32)

    def mutate(self, weights: np.ndarray) -> np.ndarray:
        """Mutação gaussiana elemento a elemento, prob. = mutation_rate."""
        mutated = weights.copy()
        rate = float(self.config["mutation_rate"])
        # máscara booleana dos genes a mutar
        mask = self.rng.random(mutated.shape) < rate
        mutated[mask] += self.rng.normal(0.0, 0.3, size=mask.sum())
        return mutated.astype(np.float32)

    def evolve_generation(self) -> dict:
        """
        Executa uma geração do AG e retorna estatísticas.
        """
        # 1) Avalia fitness
        fitness_scores = [self.evaluate_fitness(w, self.generation) for w in self.population]
        self._last_fitness_scores = fitness_scores  # cache

        # 2) Estatísticas
        best_idx = int(np.argmax(fitness_scores))
        best_fitness = float(fitness_scores[best_idx])
        avg_fitness = float(np.mean(fitness_scores))
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # 3) Convergência
        thr = int(self.config["convergence_threshold"])
        if len(self.best_fitness_history) > thr:
            recent = self.best_fitness_history[-thr:]
            if max(recent) - min(recent) < 0.1:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0

        # 4) Elitismo
        pop_size = int(self.config["population_size"])
        elite_size = max(1, int(round(pop_size * float(self.config["elitism_rate"]))))
        elite_indices = list(np.argsort(fitness_scores)[-elite_size:])
        elite = [self.population[i].copy() for i in elite_indices]

        # 5) Nova população
        new_population: list[np.ndarray] = elite.copy()
        while len(new_population) < pop_size:
            p1 = self.tournament_selection(fitness_scores)
            p2 = self.tournament_selection(fitness_scores)
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        # atualiza população e geração
        self.population = new_population
        self.generation += 1

        # melhor indivíduo da geração AVALIADA (não da nova)
        best_weights = self.population[0]  # placeholder
        # Para retornar os pesos do melhor avaliado, use o vetor salvo de elite:
        # O melhor está no fim de elite_indices
        best_weights = elite[-1] if elite else self.population[0]

        return {
            "generation": int(self.generation),
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_weights": best_weights.tolist(),
        }

    def get_best_network(self) -> NeuralNetwork:
        """
        Retorna a melhor rede neural da população atual.
        Usa o cache de fitness da última geração se existir; caso contrário, reavalia.
        """
        if self._last_fitness_scores is None or len(self._last_fitness_scores) != len(self.population):
            fitness_scores = [self.evaluate_fitness(w, self.generation) for w in self.population]
        else:
            fitness_scores = self._last_fitness_scores

        best_idx = int(np.argmax(fitness_scores))
        return NeuralNetwork(self.population[best_idx])
