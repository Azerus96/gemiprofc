from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
from ai_engine import CFRAgent, RandomAgent, Card
import utils
import github_utils
import time
import json
from threading import Thread, Event
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Глобальные экземпляры AI
cfr_agent = None  # Инициализируем, но создаем, только если нужен MCCFR
random_agent = RandomAgent()

def initialize_ai_agent(ai_settings):
    global cfr_agent
    logger.info(f"Инициализация AI агента с настройками: {ai_settings}")
    try:
        iterations = int(ai_settings.get('iterations', 10000))  # Значение по умолчанию 10000
        stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
    except ValueError as e:
        logger.error(f"Неверные значения iterations или stopThreshold: {e}. Используются значения по умолчанию.")
        iterations = 10000
        stop_threshold = 0.001

    if ai_settings.get('aiType') == 'mccfr':  # Создаем CFRAgent только если нужен MCCFR
        cfr_agent = CFRAgent(iterations=iterations, stop_threshold=stop_threshold)
        logger.info(f"AI агент MCCFR инициализирован: {cfr_agent}")

        if os.environ.get("AI_PROGRESS_TOKEN"):
            try:
                # Загрузка с GitHub
                logger.info("Попытка загрузить прогресс AI с GitHub...")
                if github_utils.load_ai_progress_from_github():
                    data = utils.load_ai_progress('cfr_data.pkl') # Загружаем локально ПОСЛЕ GitHub
                    if data:
                        cfr_agent.nodes = data['nodes']
                        cfr_agent.iterations = data['iterations']
                        cfr_agent.stop_threshold = data.get('stop_threshold', 0.0001)
                        logger.info("Прогресс AI успешно загружен и применен к агенту.")
                    else:
                        logger.warning("Прогресс AI с GitHub загружен, но данные повреждены или пусты.")
                else:
                    logger.warning("Не удалось загрузить прогресс AI с GitHub.")

            except Exception as e:
                logger.error(f"Ошибка загрузки прогресса AI: {e}")
        else:
            logger.info("AI_PROGRESS_TOKEN не установлен. Загрузка прогресса отключена.")
    else:
        cfr_agent = None  # Если не MCCFR, то cfr_agent не нужен
        logger.info("Используется случайный AI агент.")


def serialize_card(card):
    return card.to_dict() if card else None

def serialize_move(move): # Убрал next_slots
    logger.debug(f"Сериализация хода: {move}")
    serialized = {
        key: [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards)
        for key, cards in move.items()
    }
    # next_slots больше не нужен, т.к. мы не передаем его в ответе
    logger.debug(f"Сериализованный ход: {serialized}")
    return serialized

# next_available_slots больше не нужна, т.к. мы не передаем ее в ответе
# def get_next_available_slots(board): ...

@app.route('/')
def home():
    logger.debug("Обработка запроса главной страницы")
    return render_template('index.html')

@app.route('/training')
def training():
    logger.debug("Обработка запроса страницы тренировки")

    # Инициализация состояния сессии
    session['game_state'] = {
        'selected_cards': [],
        'board': {
            'top': [None] * 3,
            'middle': [None] * 5,
            'bottom': [None] * 5
        },
        'discarded_cards': [],
        'ai_settings': {  # Устанавливаем правильные значения по умолчанию
            'fantasyType': 'normal',
            'fantasyMode': False,
            'aiTime': '60',
            'iterations': '100000',
            'stopThreshold': '0.0001',
            'aiType': 'mccfr',
            'placementMode': 'standard'
        }
    }
    logger.info(f"Инициализировано состояние игры: {session['game_state']}")

    # Проверка необходимости реинициализации AI
    if session.get('previous_ai_settings') != session['game_state']['ai_settings']:
        initialize_ai_agent(session['game_state']['ai_settings'])
        session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()
        logger.info(f"Реинициализирован AI агент с настройками: {session['game_state']['ai_settings']}")

    logger.info(f"Текущее состояние игры в сессии: {session['game_state']}")
    return render_template('training.html', game_state=session['game_state'])

@app.route('/update_state', methods=['POST'])
def update_state():
    logger.debug("Обработка запроса обновления состояния - START")
    if not request.is_json:
        logger.error("Ошибка: Запрос не в формате JSON")
        return jsonify({'error': 'Content type must be application/json'}), 400

    try:
        game_state = request.get_json()
        logger.debug(f"Получено обновление состояния игры: {game_state}")

        if not isinstance(game_state, dict):
            logger.error("Ошибка: Неверный формат состояния игры (не словарь)")
            return jsonify({'error': 'Invalid game state format'}), 400

        # Инициализация состояния игры в сессии, если его нет
        if 'game_state' not in session:
            session['game_state'] = {}  # Пустой словарь, если не было
            logger.info("Инициализировано новое состояние сессии при обновлении.")

        logger.debug(f"Состояние сессии ДО обновления: {session['game_state']}")

        # Обновление доски - сохраняем существующие карты
        if 'board' in game_state:
            current_board = session['game_state'].get('board', {  #  Используем .get()
                'top': [None] * 3,
                'middle': [None] * 5,
                'bottom': [None] * 5
            })

            # Обновляем только новые карты, сохраняя существующие
            for line in ['top', 'middle', 'bottom']:
                if line in game_state['board']:
                    new_line = game_state['board'][line]
                    current_line = current_board.get(line, []) #  Используем .get()
                    for i, new_card in enumerate(new_line):
                        if i < len(current_line):  #  Проверяем индекс
                            if new_card is not None:
                                # Важно: преобразуем словарь в объект Card
                                current_line[i] = Card.from_dict(new_card) if isinstance(new_card, dict) else None
                    current_board[line] = current_line

            session['game_state']['board'] = current_board
            logger.debug(f"Обновленная доска: {session['game_state']['board']}")

        # Обновление других ключей.  Преобразуем словари в объекты Card.
        for key in ['selected_cards', 'discarded_cards']:
            if key in game_state:
                session['game_state'][key] = [
                    Card.from_dict(card) if isinstance(card, dict) else None
                    for card in game_state[key]
                ]
                logger.debug(f"Обновлены {key} в сессии: {session['game_state'][key]}")

        # Добавляем карты, удаленные через "-", в discarded_cards
        if 'removed_cards' in game_state:
            removed_cards = [Card.from_dict(card) if isinstance(card, dict) else card for card in game_state['removed_cards']]
            session['game_state']['discarded_cards'].extend([card.to_dict() for card in removed_cards]) #  Сохраняем как словари
            logger.debug(f"removed_cards добавлены в discarded_cards сессии: {session['game_state']['discarded_cards']}")
            #  Удаляем удаленные карты из selected_cards
            session['game_state']['selected_cards'] = [
                card for card in session['game_state']['selected_cards']
                if card.to_dict() not in [removed_card.to_dict() for removed_card in removed_cards]
            ]

        if 'ai_settings' in game_state:
            session['game_state']['ai_settings'] = game_state['ai_settings']
            # Реинициализация AI агента при изменении настроек
            if game_state.get('ai_settings') != session.get('previous_ai_settings'):
                logger.info("Настройки AI изменились, реинициализация агента")
                initialize_ai_agent(game_state['ai_settings'])
                session['previous_ai_settings'] = game_state.get('ai_settings', {}).copy() #  Важно: копируем!

        logger.debug(f"Состояние сессии ПОСЛЕ обновления: {session['game_state']}")
        logger.debug("Обработка запроса обновления состояния - END")
        return jsonify({'status': 'success'})

    except Exception as e:
        logger.exception(f"Ошибка в update_state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ai_move', methods=['POST'])
def ai_move():
    global cfr_agent
    global random_agent

    logger.debug("Обработка запроса хода AI - START")
    game_state_data = request.get_json()
    logger.debug(f"Получены данные состояния игры для хода AI: {game_state_data}")

    if not isinstance(game_state_data, dict):
        logger.error("Ошибка: game_state_data не является словарем")
        return jsonify({'error': 'Invalid game state data format'}), 400

    num_cards = len(game_state_data.get('selected_cards', []))
    ai_settings = game_state_data.get('ai_settings', {})
    ai_type = ai_settings.get('aiType', 'mccfr')

    try:
        # Обработка и валидация данных (используем .get() с значениями по умолчанию)
        selected_cards = [Card.from_dict(card) for card in game_state_data.get('selected_cards', [])]
        discarded_cards = [Card.from_dict(card) for card in game_state_data.get('discarded_cards', [])]

        board_data = game_state_data.get('board', {})
        board = ai_engine.Board()
        for line in ['top', 'middle', 'bottom']:
            line_data = board_data.get(line, [])
            for card_data in line_data:
                if card_data:  #  Проверяем, что card_data не None
                    board.place_card(line, Card.from_dict(card_data))

        logger.debug(f"Обработанная доска: {board}")

        # Создание состояния игры
        game_state = ai_engine.GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards,
            ai_settings=ai_settings,
            deck=ai_engine.Card.get_all_cards() #  Передаем полную колоду
        )
        logger.debug(f"Создано состояние игры: {game_state}")

        # Проверка терминального состояния
        if game_state.is_terminal():
            logger.info("Игра в терминальном состоянии")
            payoff = game_state.get_payoff()
            royalties = game_state.calculate_royalties()
            total_royalty = sum(royalties.values())
            logger.info(f"Игра окончена. Выплата: {payoff}, Роялти: {royalties}, Всего: {total_royalty}")

            # Сохранение прогресса AI (для MCCFR)
            if cfr_agent and ai_settings.get('aiType') == 'mccfr':
                try:
                    cfr_agent.save_progress()
                    logger.info("Прогресс AI сохранен локально.")
                    if github_utils.save_ai_progress_to_github():
                        logger.info("Прогресс AI сохранен на GitHub.")
                    else:
                        logger.warning("Не удалось сохранить прогресс AI на GitHub.")
                except Exception as e:
                    logger.error(f"Ошибка сохранения прогресса AI: {e}")

            return jsonify({
                'message': 'Game over',
                'payoff': payoff,
                'royalties': royalties,
                'total_royalty': total_royalty
            }), 200

        #  Убрал вычисление next_available_slots, т.к. оно не нужно для запроса

    except (KeyError, TypeError, ValueError) as e:
        logger.exception("Исключение при настройке состояния игры:")
        return jsonify({'error': f"Error during game state setup: {e}"}), 500

    timeout_event = Event()
    result = {'move': None}

    # Выбор AI агента
    if ai_type == 'mccfr':
        if cfr_agent is None:  #  Проверяем, инициализирован ли агент
            logger.error("Ошибка: MCCFR агент не инициализирован")
            return jsonify({'error': 'MCCFR agent not initialized'}), 500
        ai_thread = Thread(target=cfr_agent.get_move, args=(game_state, timeout_event, result))
    elif ai_type == 'random':
        ai_thread = Thread(target=random_agent.get_move, args=(game_state, timeout_event, result))
    else:
        logger.error(f"Неизвестный тип AI агента: {ai_type}")
        return jsonify({'error': f'Unknown AI agent type: {ai_type}'}), 400

    ai_thread.start()
    ai_thread.join(timeout=int(ai_settings.get('aiTime', 5)))  #  Таймаут из настроек

    if ai_thread.is_alive():
        timeout_event.set()
        ai_thread.join()  #  Ожидаем завершения потока
        logger.warning("Время ожидания хода AI истекло")
        return jsonify({'error': 'AI move timed out'}), 504

    move = result.get('move')
    logger.debug(f"Получен ход AI: {move}")
    if move is None or 'error' in move:
        logger.error(f"Ошибка хода AI: {move.get('error', 'Unknown error')}")
        return jsonify({'error': move.get('error', 'Unknown error')}), 500

    #  Убрал обновление состояния сессии здесь.  Оно делается в update_state.

    # Расчет роялти (после применения хода)
    royalties = game_state.calculate_royalties()
    total_royalty = sum(royalties.values())

    logger.debug(f"Отправка хода AI: {move}, Роялти: {royalties}, Всего роялти: {total_royalty}")
    logger.debug("Обработка запроса хода AI - END")
    return jsonify({
        'move': serialize_move(move),  #  Убрал next_available_slots
        'royalties': royalties,
        'total_royalty': total_royalty
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=10000)
