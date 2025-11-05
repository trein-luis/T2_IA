class TicTacToe:
    """
    Implementação do jogo da velha
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reinicia o jogo"""
        self.board = [''] * 9
        self.current_player = 'X'
        return self.board.copy()
    
    def is_valid_move(self, position):
        """Verifica se a jogada é válida"""
        return 0 <= position < 9 and self.board[position] == ''
    
    def make_move(self, position):
        """
        Realiza uma jogada
        
        Args:
            position: índice da posição (0-8)
        
        Returns:
            bool: True se a jogada foi válida, False caso contrário
        """
        if not self.is_valid_move(position):
            return False
        
        self.board[position] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True
    
    def check_winner(self):
        """
        Verifica se há um vencedor
        
        Returns:
            str: 'X', 'O' ou None
        """
        # Linhas, colunas e diagonais
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # linhas
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # colunas
            [0, 4, 8], [2, 4, 6]              # diagonais
        ]
        
        for line in lines:
            if (self.board[line[0]] == self.board[line[1]] == 
                self.board[line[2]] and self.board[line[0]] != ''):
                return self.board[line[0]]
        
        return None
    
    def is_game_over(self):
        """Verifica se o jogo terminou"""
        return self.check_winner() is not None or '' not in self.board
    
    def get_result(self, player):
        """
        Obtém o resultado do jogo para um jogador
        
        Args:
            player: 'X' ou 'O'
        
        Returns:
            str: 'win', 'loss', 'draw' ou None
        """
        winner = self.check_winner()
        if winner == player:
            return 'win'
        elif winner is not None:
            return 'loss'
        elif '' not in self.board:
            return 'draw'
        return None
    
    def get_board(self):
        """Retorna uma cópia do tabuleiro"""
        return self.board.copy()