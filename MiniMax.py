import random
import NeuralNetwork as nn
import MiniMax as mm

class MinimaxPlayer:
    """
    Jogador Minimax com diferentes níveis de dificuldade
    """
    
    def __init__(self, difficulty='hard'):
        """
        Args:
            difficulty: 'medium' ou 'hard'
        """
        self.difficulty = difficulty
    
    def get_move(self, board, player):
        """
        Obtém a melhor jogada
        
        Args:
            board: lista com o estado do tabuleiro
            player: 'X' ou 'O'
        
        Returns:
            int: índice da melhor jogada
        """
        if self.difficulty == 'medium' and random.random() < 0.5:
            # 50% jogadas aleatórias no modo médio
            valid_moves = [i for i, cell in enumerate(board) if cell == '']
            return random.choice(valid_moves) if valid_moves else -1
        
        return self.minimax_move(board, player)
    
    def minimax_move(self, board, player):
        """Calcula a melhor jogada usando minimax"""
        valid_moves = [i for i, cell in enumerate(board) if cell == '']
        if not valid_moves:
            return -1
        
        best_score = float('-inf')
        best_move = valid_moves[0]
        
        for move in valid_moves:
            board_copy = board.copy()
            board_copy[move] = player
            score = self.minimax(board_copy, False, player)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def minimax(self, board, is_maximizing, player):
        """Algoritmo minimax recursivo"""
        opponent = 'O' if player == 'X' else 'X'
        winner = self._check_winner_static(board)
        
        if winner == player:
            return 10
        elif winner == opponent:
            return -10
        elif '' not in board:
            return 0
        
        if is_maximizing:
            best_score = float('-inf')
            for i in range(9):
                if board[i] == '':
                    board[i] = player
                    score = self.minimax(board, False, player)
                    board[i] = ''
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] == '':
                    board[i] = opponent
                    score = self.minimax(board, True, player)
                    board[i] = ''
                    best_score = min(score, best_score)
            return best_score
    
    def _check_winner_static(self, board):
        """Verifica vencedor sem modificar o tabuleiro"""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for line in lines:
            if (board[line[0]] == board[line[1]] == board[line[2]] 
                and board[line[0]] != ''):
                return board[line[0]]
        return None