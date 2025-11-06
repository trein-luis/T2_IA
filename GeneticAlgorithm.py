import NeuralNetwork as nn
import MiniMax as mm

class GeneticAlgorithm:
    """
    Algoritmo Genético para evolução dos pesos da rede neural
    """
    
    def __init__(self, config):
        """
        Args:
            config: dicionário com parâmetros do AG
        """
        self.config = config
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.convergence_counter = 0
    
    def initialize_population(self):
        """Inicializa a população com indivíduos aleatórios"""
        nn = NeuralNetwork()
        self.population = []
        
        for _ in range(self.config['population_size']):
            weights = np.random.randn(nn.total_weights) * 0.5
            self.population.append(weights)
    
    def evaluate_fitness(self, weights, generation):
        """
        Avalia o fitness de um indivíduo
        
        Args:
            weights: pesos da rede neural
            generation: geração atual
        
        Returns:
            float: fitness do indivíduo
        """
        nn = NeuralNetwork(weights)
        
        # Determina modo do Minimax baseado na geração
        if generation < self.config['medium_mode_generations']:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
        
        minimax = MinimaxPlayer(difficulty)
        
        fitness = 0
        num_games = 10  # Joga 10 partidas por avaliação
        
        for _ in range(num_games):
            game = TicTacToe()
            invalid_moves = 0
            
            while not game.is_game_over():
                if game.current_player == 'X':  # IA sempre joga como X
                    move = nn.predict(game.board)
                    
                    if not game.is_valid_move(move):
                        invalid_moves += 1
                        # Escolhe movimento aleatório válido
                        valid = [i for i, c in enumerate(game.board) if c == '']
                        move = random.choice(valid) if valid else 0
                    
                    game.make_move(move)
                else:  # Minimax joga como O
                    move = minimax.get_move(game.board, 'O')
                    game.make_move(move)
            
            # Calcula fitness baseado no resultado
            result = game.get_result('X')
            if result == 'win':
                fitness += 10
            elif result == 'draw':
                fitness += 0
            elif result == 'loss':
                fitness -= 10
            
            # Penaliza jogadas inválidas
            fitness -= invalid_moves * 5
        
        return fitness / num_games  # Fitness médio
    
    def tournament_selection(self, fitness_scores):
        """Seleção por torneio"""
        tournament_indices = random.sample(
            range(len(fitness_scores)), 
            self.config['tournament_size']
        )
        tournament = [(i, fitness_scores[i]) for i in tournament_indices]
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return self.population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """
        Crossover aritmético para valores reais
        
        Args:
            parent1, parent2: arrays de pesos
        
        Returns:
            tuple: (child1, child2)
        """
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def mutate(self, weights):
        """
        Mutação gaussiana
        
        Args:
            weights: array de pesos
        
        Returns:
            array: pesos mutados
        """
        mutated = weights.copy()
        for i in range(len(mutated)):
            if random.random() < self.config['mutation_rate']:
                mutated[i] += np.random.randn() * 0.3
        return mutated
    
    def evolve_generation(self):
        """
        Executa uma geração do AG
        
        Returns:
            dict: estatísticas da geração
        """
        # Avaliar fitness de toda população
        fitness_scores = [
            self.evaluate_fitness(w, self.generation) 
            for w in self.population
        ]
        
        # Estatísticas
        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Verificar convergência
        if len(self.best_fitness_history) > self.config['convergence_threshold']:
            recent = self.best_fitness_history[-self.config['convergence_threshold']:]
            if max(recent) - min(recent) < 0.1:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0
        
        # Elitismo - mantém os melhores
        elite_size = int(
            self.config['population_size'] * self.config['elitism_rate']
        )
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite = [self.population[i] for i in elite_indices]
        
        # Nova população
        new_population = elite.copy()
        
        # Gerar novos indivíduos
        while len(new_population) < self.config['population_size']:
            parent1 = self.tournament_selection(fitness_scores)
            parent2 = self.tournament_selection(fitness_scores)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.config['population_size']:
                new_population.append(child2)
        
        self.population = new_population
        self.generation += 1
        
        return {
            'generation': self.generation,
            'best_fitness': float(best_fitness),
            'avg_fitness': float(avg_fitness),
            'best_weights': self.population[elite_indices[-1]].tolist()
        }
    
    def get_best_network(self):
        """Retorna a melhor rede neural da população"""
        fitness_scores = [
            self.evaluate_fitness(w, self.generation) 
            for w in self.population
        ]
        best_idx = np.argmax(fitness_scores)
        return NeuralNetwork(self.population[best_idx])

