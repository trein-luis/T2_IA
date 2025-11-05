import numpy as np

class NeuralNetwork:
    """
    Rede Neural MLP com 2 camadas
    Arquitetura: 9 (input) -> 18 (hidden) -> 9 (output)
    """
    
    def __init__(self, weights=None):
        self.input_size = 9
        self.hidden_size = 18
        self.output_size = 9
        
        # Total de pesos: (9*18 + 18) + (18*9 + 9) = 351
        self.total_weights = (
            self.input_size * self.hidden_size + self.hidden_size +
            self.hidden_size * self.output_size + self.output_size
        )
        
        if weights is not None and len(weights) == self.total_weights:
            self.weights = np.array(weights)
        else:
            # Inicialização aleatória dos pesos
            self.weights = np.random.randn(self.total_weights) * 0.5
    
    def sigmoid(self, x):
        """Função de ativação sigmoide"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, input_data):
        """
        Propagação forward da rede neural
        
        Args:
            input_data: array de 9 elementos (tabuleiro)
        
        Returns:
            output: array de 9 elementos (probabilidades)
        """
        idx = 0
        
        # Camada de entrada para camada oculta
        w1_size = self.input_size * self.hidden_size
        w1 = self.weights[idx:idx + w1_size].reshape(
            self.hidden_size, self.input_size
        )
        idx += w1_size
        
        b1 = self.weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        hidden = self.sigmoid(np.dot(w1, input_data) + b1)
        
        # Camada oculta para camada de saída
        w2_size = self.hidden_size * self.output_size
        w2 = self.weights[idx:idx + w2_size].reshape(
            self.output_size, self.hidden_size
        )
        idx += w2_size
        
        b2 = self.weights[idx:idx + self.output_size]
        
        output = self.sigmoid(np.dot(w2, hidden) + b2)
        
        return output
    
    def predict(self, board):
        """
        Prediz a melhor jogada para o tabuleiro dado
        
        Args:
            board: lista de 9 elementos ('X', 'O' ou '')
        
        Returns:
            int: índice da melhor jogada (0-8) ou -1 se não houver
        """
        # Converte tabuleiro: X=1, O=-1, vazio=0
        input_data = np.array([
            1 if cell == 'X' else -1 if cell == 'O' else 0 
            for cell in board
        ])
        
        output = self.forward(input_data)
        
        # Encontrar melhor jogada válida
        valid_moves = [i for i, cell in enumerate(board) if cell == '']
        
        if not valid_moves:
            return -1
        
        # Seleciona a jogada com maior score entre as válidas
        best_move = max(valid_moves, key=lambda i: output[i])
        return best_move
    
    def export_weights(self):
        """Exporta os pesos da rede"""
        return self.weights.tolist()







