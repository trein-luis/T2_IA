# ==================== App.py (VERSÃO CORRIGIDA) ====================
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
import json
import secrets

# Importa os módulos criados anteriormente
from NeuralNetwork import NeuralNetwork
from Game import TicTacToe
from MiniMax import MinimaxPlayer
from GeneticAlgorithm import GeneticAlgorithm
from Config import CONFIG

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# Armazena instâncias de treinamento por sessão
training_sessions = {}

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Inicia um novo jogo"""
    game = TicTacToe()
    
    # Armazena o jogo na sessão
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(8)
    
    return jsonify({
        'board': game.get_board(),
        'current_player': game.current_player,
        'game_over': False
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Processa uma jogada do usuário"""
    data = request.json
    board = data.get('board')
    position = data.get('position')
    
    game = TicTacToe()
    game.board = board
    game.current_player = data.get('current_player', 'X')
    
    if game.make_move(position):
        return jsonify({
            'success': True,
            'board': game.get_board(),
            'current_player': game.current_player,
            'winner': game.check_winner(),
            'game_over': game.is_game_over()
        })
    
    return jsonify({'success': False}), 400

@app.route('/api/minimax_move', methods=['POST'])
def minimax_move():
    """Calcula jogada do Minimax"""
    data = request.json
    board = data.get('board')
    player = data.get('player', 'O')
    difficulty = data.get('difficulty', 'hard')
    
    minimax = MinimaxPlayer(difficulty)
    move = minimax.get_move(board, player)
    
    game = TicTacToe()
    game.board = board
    game.current_player = player
    
    if move >= 0 and game.make_move(move):
        return jsonify({
            'move': move,
            'board': game.get_board(),
            'current_player': game.current_player,
            'winner': game.check_winner(),
            'game_over': game.is_game_over()
        })
    
    return jsonify({'error': 'No valid move'}), 400

@app.route('/api/nn_move', methods=['POST'])
def nn_move():
    """Calcula jogada da Rede Neural"""
    data = request.json
    board = data.get('board')
    weights = data.get('weights')
    
    if not weights:
        return jsonify({'error': 'No trained network'}), 400
    
    nn = NeuralNetwork(weights)
    move = nn.predict(board)
    
    game = TicTacToe()
    game.board = board
    game.current_player = data.get('current_player', 'O')
    
    if move >= 0 and game.make_move(move):
        return jsonify({
            'move': move,
            'board': game.get_board(),
            'current_player': game.current_player,
            'winner': game.check_winner(),
            'game_over': game.is_game_over()
        })
    
    return jsonify({'error': 'No valid move'}), 400

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Inicia o treinamento"""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(8)
    
    session_id = session['session_id']
    
    ga = GeneticAlgorithm(CONFIG)
    ga.initialize_population()
    
    training_sessions[session_id] = {
        'ga': ga,
        'is_training': True
    }
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'config': CONFIG
    })

@app.route('/api/train_generation', methods=['POST'])
def train_generation():
    """Executa uma geração do treinamento"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in training_sessions:
        return jsonify({'error': 'No training session'}), 400
    
    # Verifica se o treinamento foi pausado/parado
    if not training_sessions[session_id]['is_training']:
        ga = training_sessions[session_id]['ga']
        best_nn = ga.get_best_network()
        
        return jsonify({
            'completed': True,
            'stopped': True,
            'generation': ga.generation,
            'best_fitness': float(ga.best_fitness_history[-1]) if ga.best_fitness_history else 0,
            'avg_fitness': float(ga.avg_fitness_history[-1]) if ga.avg_fitness_history else 0,
            'weights': best_nn.export_weights(),
            'history': {
                'best': ga.best_fitness_history,
                'avg': ga.avg_fitness_history
            }
        })
    
    ga = training_sessions[session_id]['ga']
    
    if ga.generation >= CONFIG['max_generations'] or ga.convergence_counter >= 10:
        training_sessions[session_id]['is_training'] = False
        best_nn = ga.get_best_network()
        
        return jsonify({
            'completed': True,
            'stopped': False,
            'generation': ga.generation,
            'best_fitness': float(ga.best_fitness_history[-1]) if ga.best_fitness_history else 0,
            'avg_fitness': float(ga.avg_fitness_history[-1]) if ga.avg_fitness_history else 0,
            'weights': best_nn.export_weights(),
            'history': {
                'best': ga.best_fitness_history,
                'avg': ga.avg_fitness_history
            }
        })
    
    stats = ga.evolve_generation()
    
    return jsonify({
        'completed': False,
        'generation': stats['generation'],
        'best_fitness': stats['best_fitness'],
        'avg_fitness': stats['avg_fitness'],
        'history': {
            'best': ga.best_fitness_history,
            'avg': ga.avg_fitness_history
        }
    })

@app.route('/api/get_training_status', methods=['GET'])
def get_training_status():
    """Obtém o status do treinamento"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in training_sessions:
        return jsonify({'is_training': False})
    
    ga = training_sessions[session_id]['ga']
    
    return jsonify({
        'is_training': training_sessions[session_id]['is_training'],
        'generation': ga.generation,
        'best_fitness': float(ga.best_fitness_history[-1]) if ga.best_fitness_history else 0,
        'avg_fitness': float(ga.avg_fitness_history[-1]) if ga.avg_fitness_history else 0
    })

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """Para o treinamento"""
    session_id = session.get('session_id')
    
    if session_id and session_id in training_sessions:
        training_sessions[session_id]['is_training'] = False
        ga = training_sessions[session_id]['ga']
        
        best_nn = ga.get_best_network()
        
        return jsonify({
            'success': True,
            'weights': best_nn.export_weights(),
            'generation': ga.generation
        })
    
    return jsonify({'error': 'No training session'}), 400

if __name__ == '__main__':
    print("🚀 Servidor iniciando...")
    print("📡 Acesse: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)