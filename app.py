from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
from ai_engine import CFRAgent, RandomAgent, Card, SafeResult
import utils
import github_utils
import time
import json
from threading import Thread, Event
import logging
from functools import wraps

# Настройка логирования
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Глобальные экземпляры AI
cfr_agent = None
random_agent = RandomAgent()

def validate_game_state(game_state):
    """Валидация состояния игры"""
    if not isinstance(game_state, dict):
        return False, "Invalid game state type"
        
    required_fields = ['board', 'selected_cards', 'discarded_cards', 'ai_settings']
    for field in required_fields:
        if field not in game_state:
            return False, f"Missing required field: {field}"
            
    # Проверка структуры доски
    if not isinstance(game_state['board'], dict):
        return False, "Invalid board structure"
    
    for line in ['top', 'middle', 'bottom']:
        if line not in game_state['board']:
            return False, f"Missing board line: {line}"
        if not isinstance(game_state['board'][line], list):
            return False, f"Invalid {line} line type"
            
    # Проверка карт
    if not isinstance(game_state['selected_cards'], list):
        return False, "Invalid selected_cards type"
    if not isinstance(game_state['discarded_cards'], list):
        return False, "Invalid discarded_cards type"
        
    # Проверка настроек AI
    if not isinstance(game_state['ai_settings'], dict):
        return False, "Invalid ai_settings type"
        
    return True, None

def execute_ai_move(game_state, ai_agent, timeout_seconds):
    """Безопасное выполнение хода AI"""
    timeout_event = Event()
    result = SafeResult()
    
    ai_thread = Thread(target=ai_agent.get_move,
                      args=(game_state, timeout_event, result))
    
    try:
        ai_thread.start()
        ai_thread.join(timeout=timeout_seconds)
        
        if ai_thread.is_alive():
            timeout_event.set()
            ai_thread.join()
            return {'error': 'AI move timed out'}
            
        move = result.get_move()
        if move is None:
            return {'error': 'AI failed to make a move'}
            
        return move
    except Exception as e:
        logger.exception("Error executing AI move:")
        return {'error': str(e)}

def atomic_session_update(func):
    """Декоратор для атомарного обновления сессии"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("Session update error:")
            session.rollback()
            raise
    return wrapper

def initialize_ai_agent(ai_settings):
    """Инициализация AI агента с обработкой ошибок"""
    global cfr_agent
    logger.info(f"Инициализация AI агента с настройками: {ai_settings}")
    try:
        iterations = int(ai_settings.get('iterations', 10000))
        stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
    except ValueError as e:
        logger.error(f"Ошибка преобразования настроек AI: {e}")
        iterations = 10000
        stop_threshold = 0.001

    try:
        cfr_agent = CFRAgent(iterations=iterations, stop_threshold=stop_threshold)
        if os.environ.get("AI_PROGRESS_TOKEN"):
            if cfr_agent.load_ai_progress_from_github():
                logger.info("Прогресс AI успешно загружен")
            else:
                logger.warning("Не удалось загрузить прогресс AI")
        else:
            logger.info("AI_PROGRESS_TOKEN не установлен")
        return True
    except Exception as e:
        logger.exception("Ошибка инициализации AI агента:")
        return False

# Инициализация AI агента при запуске
initialize_ai_agent({})
def serialize_card(card):
    """Безопасная сериализация карты с проверкой"""
    try:
        return card.to_dict() if card else None
    except AttributeError as e:
        logger.error(f"Ошибка сериализации карты: {e}")
        return None

def serialize_move(move, next_slots):
    """Безопасная сериализация хода с валидацией"""
    logger.debug(f"Сериализация хода: {move}, next_slots: {next_slots}")
    try:
        if not isinstance(move, dict):
            raise ValueError("Move must be a dictionary")
            
        serialized = {
            key: [serialize_card(card) for card in cards] if isinstance(cards, list) 
                else serialize_card(cards)
            for key, cards in move.items()
        }
        serialized['next_available_slots'] = next_slots
        logger.debug(f"Сериализованный ход: {serialized}")
        return serialized
    except Exception as e:
        logger.error(f"Ошибка сериализации хода: {e}")
        return None

def get_next_available_slots(board):
    """Вычисление следующих доступных слотов с валидацией"""
    logger.debug("Вычисление следующих доступных слотов")
    try:
        if not isinstance(board, dict):
            raise ValueError("Board must be a dictionary")
            
        next_slots = {}
        max_slots = {'top': 3, 'middle': 5, 'bottom': 5}
        
        for line in ['top', 'middle', 'bottom']:
            if line not in board:
                raise ValueError(f"Missing line: {line}")
                
            line_cards = board.get(line, [])
            if not isinstance(line_cards, list):
                raise ValueError(f"Invalid line type for {line}")
                
            slot = 0
            while slot < len(line_cards) and slot < max_slots[line] and line_cards[slot] is not None:
                slot += 1
            next_slots[line] = min(slot, max_slots[line])
            
        logger.debug(f"Следующие доступные слоты: {next_slots}")
        return next_slots
    except Exception as e:
        logger.error(f"Ошибка вычисления слотов: {e}")
        return {'top': 0, 'middle': 0, 'bottom': 0}

@app.route('/')
def home():
    """Обработчик главной страницы"""
    logger.debug("Обработка запроса главной страницы")
    return render_template('index.html')

@app.route('/training')
def training():
    """Обработчик страницы тренировки с улучшенной обработкой состояния"""
    logger.debug("Обработка запроса страницы тренировки")

    try:
        if 'game_state' not in session:
            logger.info("Инициализация нового состояния игры в сессии")
            session['game_state'] = {
                'selected_cards': [],
                'board': {
                    'top': [None] * 3,
                    'middle': [None] * 5,
                    'bottom': [None] * 5
                },
                'discarded_cards': [],
                'ai_settings': {
                    'fantasyType': 'normal',
                    'fantasyMode': False,
                    'aiTime': '5',
                    'iterations': '10000',
                    'stopThreshold': '0.001',
                    'aiType': 'mccfr',
                    'placementMode': 'standard'
                }
            }
            logger.info(f"Инициализировано новое состояние игры: {session['game_state']}")
        else:
            logger.info("Загрузка существующего состояния игры")
            # Безопасное преобразование карт
            try:
                for key in ['selected_cards', 'discarded_cards']:
                    if key in session['game_state']:
                        session['game_state'][key] = [
                            Card.from_dict(card_dict) for card_dict in session['game_state'][key]
                            if isinstance(card_dict, dict)
                        ]

                for line in ['top', 'middle', 'bottom']:
                    if line in session['game_state']['board']:
                        session['game_state']['board'][line] = [
                            Card.from_dict(card_dict) if isinstance(card_dict, dict) else None
                            for card_dict in session['game_state']['board'][line]
                        ]
            except Exception as e:
                logger.error(f"Ошибка преобразования карт: {e}")
                # Сброс состояния при ошибке
                session.pop('game_state', None)
                return redirect(url_for('training'))

        # Проверка и реинициализация AI
        if (cfr_agent is None or
            session['game_state']['ai_settings'] != session.get('previous_ai_settings')):
            if initialize_ai_agent(session['game_state']['ai_settings']):
                session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()
                logger.info(f"Реинициализирован AI агент с настройками: {session['game_state']['ai_settings']}")
            else:
                logger.error("Ошибка реинициализации AI агента")
                return jsonify({'error': 'AI initialization failed'}), 500

        logger.info(f"Текущее состояние игры: {session['game_state']}")
        return render_template('training.html', game_state=session['game_state'])
        
    except Exception as e:
        logger.exception("Ошибка в обработчике training:")
        return jsonify({'error': str(e)}), 500
@app.route('/update_state', methods=['POST'])
@atomic_session_update
def update_state():
    """Обработчик обновления состояния с улучшенной валидацией"""
    logger.debug("Обработка запроса обновления состояния")
    
    if not request.is_json:
        logger.error("Ошибка: Запрос не в формате JSON")
        return jsonify({'error': 'Content type must be application/json'}), 400

    try:
        game_state = request.get_json()
        logger.debug(f"Получено обновление состояния игры: {game_state}")

        # Валидация входящего состояния
        is_valid, error = validate_game_state(game_state)
        if not is_valid:
            logger.error(f"Ошибка валидации состояния: {error}")
            return jsonify({'error': error}), 400

        # Инициализация состояния сессии при необходимости
        if 'game_state' not in session:
            session['game_state'] = {}
            logger.info("Инициализировано новое состояние сессии")

        logger.debug(f"Состояние сессии ДО обновления: {session['game_state']}")

        # Безопасное обновление доски
        try:
            if 'board' in game_state:
                current_board = session['game_state'].get('board', {
                    'top': [None] * 3,
                    'middle': [None] * 5,
                    'bottom': [None] * 5
                })

                for line in ['top', 'middle', 'bottom']:
                    if line in game_state['board']:
                        new_line = game_state['board'][line]
                        current_line = current_board[line]
                        for i, new_card in enumerate(new_line):
                            if i < len(current_line):
                                current_line[i] = Card.from_dict(new_card) if isinstance(new_card, dict) else None
                        current_board[line] = current_line

                session['game_state']['board'] = current_board
                logger.debug(f"Обновленная доска: {session['game_state']['board']}")

            # Безопасное обновление карт
            for key in ['selected_cards', 'discarded_cards']:
                if key in game_state:
                    session['game_state'][key] = [
                        Card.from_dict(card) if isinstance(card, dict) else None
                        for card in game_state[key]
                    ]

            # Обновление настроек AI
            if 'ai_settings' in game_state:
                old_settings = session['game_state'].get('ai_settings', {})
                new_settings = game_state['ai_settings']
                
                if new_settings != old_settings:
                    logger.info("Настройки AI изменились, реинициализация агента")
                    if initialize_ai_agent(new_settings):
                        session['game_state']['ai_settings'] = new_settings
                        session['previous_ai_settings'] = new_settings.copy()
                    else:
                        return jsonify({'error': 'Failed to initialize AI with new settings'}), 500

            logger.debug(f"Состояние сессии ПОСЛЕ обновления: {session['game_state']}")
            return jsonify({'status': 'success'})

        except Exception as e:
            logger.exception("Ошибка обновления состояния:")
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.exception("Ошибка в update_state:")
        return jsonify({'error': str(e)}), 500
@app.route('/ai_move', methods=['POST'])
def ai_move():
    """Обработчик хода AI с улучшенной обработкой ошибок и валидацией"""
    global cfr_agent
    global random_agent

    logger.debug("Обработка запроса хода AI")
    
    try:
        game_state_data = request.get_json()
        logger.debug(f"Получены данные состояния игры для хода AI: {game_state_data}")

        # Валидация входящего состояния
        is_valid, error = validate_game_state(game_state_data)
        if not is_valid:
            logger.error(f"Ошибка валидации состояния: {error}")
            return jsonify({'error': error}), 400

        num_cards = len(game_state_data.get('selected_cards', []))
        ai_settings = game_state_data.get('ai_settings', {})
        ai_type = ai_settings.get('aiType', 'mccfr')

        try:
            # Безопасное преобразование карт
            selected_cards = [Card.from_dict(card) for card in game_state_data.get('selected_cards', [])]
            discarded_cards = [Card.from_dict(card) for card in game_state_data.get('discarded_cards', [])]
            
            # Создание и валидация доски
            board = ai_engine.Board()
            board_data = game_state_data.get('board', {})
            
            for line in ['top', 'middle', 'bottom']:
                line_data = board_data.get(line, [])
                for card_data in line_data:
                    if card_data:
                        board.place_card(line, Card.from_dict(card_data))

            logger.debug(f"Обработанная доска: {board}")

            # Подсчет слотов и валидация
            occupied_slots = sum(1 for line in ['top', 'middle', 'bottom']
                               for card in board_data.get(line, [])
                               if card is not None)
            free_slots = 13 - occupied_slots

            logger.debug(f"Занято слотов: {occupied_slots}")
            logger.debug(f"Свободных слотов: {free_slots}")
            logger.debug(f"Выбрано карт: {num_cards}")

            # Создание состояния игры
            game_state = ai_engine.GameState(
                selected_cards=selected_cards,
                board=board,
                discarded_cards=discarded_cards,
                ai_settings=ai_settings,
                deck=ai_engine.Card.get_all_cards()
            )
            logger.debug(f"Создано состояние игры: {game_state}")

            # Проверка терминального состояния
            if game_state.is_terminal():
                logger.info("Игра в терминальном состоянии")
                payoff = game_state.get_payoff()
                royalties = game_state.calculate_royalties()
                total_royalty = sum(royalties.values())
                
                # Сохранение прогресса AI
                if cfr_agent and ai_type == 'mccfr':
                    try:
                        cfr_agent.save_ai_progress_to_github()
                        logger.info("Прогресс AI сохранен")
                    except Exception as e:
                        logger.error(f"Ошибка сохранения прогресса AI: {e}")

                return jsonify({
                    'message': 'Game over',
                    'payoff': payoff,
                    'royalties': royalties,
                    'total_royalty': total_royalty
                }), 200

            # Получение следующих доступных слотов
            next_available_slots = get_next_available_slots(game_state_data['board'])
            logger.debug(f"Следующие доступные слоты: {next_available_slots}")

            # Выполнение хода AI
            ai_agent = cfr_agent if ai_type == 'mccfr' else random_agent
            if ai_agent is None:
                raise ValueError(f"AI agent {ai_type} not initialized")

            move = execute_ai_move(
                game_state=game_state,
                ai_agent=ai_agent,
                timeout_seconds=int(ai_settings.get('aiTime', 5))
            )

            if isinstance(move, dict) and 'error' in move:
                logger.error(f"Ошибка выполнения хода AI: {move['error']}")
                return jsonify({'error': move['error']}), 500

            # Обновление состояния сессии
            if move:
                logger.info("Обновление состояния после хода AI")
                current_board = session['game_state']['board']

                for line in ['top', 'middle', 'bottom']:
                    if line in move:
                        new_cards = move[line]
                        current_line = current_board[line]
                        slot_index = next_available_slots[line]

                        for new_card in new_cards:
                            if new_card is not None and slot_index < len(current_line):
                                current_line[slot_index] = new_card
                                slot_index += 1

                session['game_state']['board'] = current_board
                session['game_state']['selected_cards'] = []

                if 'discarded' in move and move['discarded']:
                    if isinstance(move['discarded'], list):
                        session['game_state']['discarded_cards'].extend(move['discarded'])
                    else:
                        session['game_state']['discarded_cards'].append(move['discarded'])

            # Расчет роялти
            royalties = game_state.calculate_royalties()
            total_royalty = sum(royalties.values())

            # Сериализация и отправка ответа
            serialized_move = serialize_move(move, next_available_slots)
            if serialized_move is None:
                raise ValueError("Failed to serialize move")

            logger.debug(f"Отправка ответа: move={serialized_move}, royalties={royalties}")
            return jsonify({
                'move': serialized_move,
                'royalties': royalties,
                'total_royalty': total_royalty
            }), 200

        except Exception as e:
            logger.exception("Ошибка выполнения хода AI:")
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.exception("Критическая ошибка в ai_move:")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=10000)
