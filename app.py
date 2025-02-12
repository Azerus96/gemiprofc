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
cfr_agent = None
random_agent = RandomAgent()

def initialize_ai_agent(ai_settings):
    global cfr_agent
    logger.info(f"Инициализация AI агента с настройками: {ai_settings}")
    try:
        iterations = int(ai_settings.get('iterations', 10000))  # Значение по умолчанию 10000
        stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
    except ValueError:
        logger.error("Неверные значения iterations или stopThreshold. Используются значения по умолчанию.")
        iterations = 10000  # Установим более высокое значение по умолчанию
        stop_threshold = 0.001

    cfr_agent = CFRAgent(iterations=iterations, stop_threshold=stop_threshold)

    if os.environ.get("AI_PROGRESS_TOKEN"):
        try:
            # Загрузка с GitHub
            if github_utils.load_ai_progress_from_github():
                data = utils.load_ai_progress('cfr_data.pkl') # Загружаем локально ПОСЛЕ GitHub
                if data:
                    cfr_agent.nodes = data['nodes']
                    cfr_agent.iterations = data['iterations']
                    cfr_agent.stop_threshold = data.get('stop_threshold', 0.0001)
                logger.info("Прогресс AI успешно загружен.")
            else:
                logger.warning("Не удалось загрузить прогресс AI с GitHub.")

        except Exception as e:
            logger.error(f"Ошибка загрузки прогресса AI: {e}")
    else:
        logger.info("AI_PROGRESS_TOKEN не установлен. Загрузка прогресса отключена.")

# Инициализация AI агента с настройками по умолчанию при запуске
initialize_ai_agent({})

def serialize_card(card):
    return card.to_dict() if card else None

def serialize_move(move, next_slots):
    logger.debug(f"Сериализация хода: {move}, next_slots: {next_slots}")
    serialized = {
        key: [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards)
        for key, cards in move.items()
    }
    serialized['next_available_slots'] = next_slots
    logger.debug(f"Сериализованный ход: {serialized}")
    return serialized

def get_next_available_slots(board):
    logger.debug("Вычисление следующих доступных слотов")
    next_slots = {}
    for line in ['top', 'middle', 'bottom']:
        line_cards = board.get(line, [])
        slot = 0
        # Исправлено: используем None для проверки занятости слота
        while slot < len(line_cards) and line_cards[slot] is not None:
            slot += 1
        next_slots[line] = slot
    logger.debug(f"Следующие доступные слоты: {next_slots}")
    return next_slots

@app.route('/')
def home():
    logger.debug("Обработка запроса главной страницы")
    return render_template('index.html')

@app.route('/training')
def training():
    logger.debug("Обработка запроса страницы тренировки")

    # Инициализация состояния, *ДАЖЕ ЕСЛИ* оно уже есть в сессии (для настроек по умолчанию)
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
    logger.info(f"Инициализировано/перезаписано состояние игры: {session['game_state']}")


    # Проверка необходимости реинициализации AI (оставляем как было)
    if (cfr_agent is None or
        session['game_state']['ai_settings'] != session.get('previous_ai_settings')):
        initialize_ai_agent(session['game_state']['ai_settings'])
        session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()
        logger.info(f"Реинициализирован AI агент с настройками: {session['game_state']['ai_settings']}")

    logger.info(f"Текущее состояние игры в сессии: {session['game_state']}")
    return render_template('training.html', game_state=session['game_state'])

@app.route('/update_state', methods=['POST'])
def update_state():
    logger.debug("Обработка запроса обновления состояния")
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
            current_board = session['game_state'].get('board', {
                'top': [None] * 3,
                'middle': [None] * 5,
                'bottom': [None] * 5
            })

            # Обновляем только новые карты, сохраняя существующие
            for line in ['top', 'middle', 'bottom']:
                if line in game_state['board']:
                    new_line = game_state['board'][line]
                    current_line = current_board[line]
                    for i, new_card in enumerate(new_line):
                        if i < len(current_line):
                            if new_card is not None:
                                # Важно: преобразуем словарь в объект Card
                                current_line[i] = Card.from_dict(new_card) if isinstance(new_card, dict) else None
                    current_board[line] = current_line

            session['game_state']['board'] = current_board
            logger.debug(f"Обновленная доска: {session['game_state']['board']}")

        # Обновление других ключей.  Преобразуем словари в объекты Card.
        for key in ['selected_cards', 'discarded_cards']:
            if key in game_state:
                if key == 'discarded_cards':
                    session['game_state'][key] = [
                        card for card in game_state[key]
                    ]
                else:
                    session['game_state'][key] = [
                        Card.from_dict(card) if isinstance(card, dict) else None
                        for card in game_state[key]
                    ]


        # Добавляем карты, удаленные через "-", в discarded_cards
        if 'removed_cards' in game_state:  # 'removed_cards' приходит из frontend
            removed_cards = [Card.from_dict(card) for card in game_state['removed_cards']]
            session['game_state']['discarded_cards'].extend([card.to_dict() for card in removed_cards]) #Сохраняем в виде словаря

        if 'ai_settings' in game_state:
            session['game_state']['ai_settings'] = game_state['ai_settings']

        # Реинициализация AI агента при изменении настроек
        if game_state.get('ai_settings') != session.get('previous_ai_settings'):
            logger.info("Настройки AI изменились, реинициализация агента")
            initialize_ai_agent(game_state['ai_settings'])
            session['previous_ai_settings'] = game_state.get('ai_settings', {}).copy()

        logger.debug(f"Состояние сессии ПОСЛЕ обновления: {session['game_state']}")
        return jsonify({'status': 'success'})

    except Exception as e:
        logger.exception(f"Ошибка в update_state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ai_move', methods=['POST'])
def ai_move():
    global cfr_agent
    global random_agent

    logger.debug("Обработка запроса хода AI")
    game_state_data = request.get_json()
    logger.debug(f"Получены данные состояния игры для хода AI: {game_state_data}")

    if not isinstance(game_state_data, dict):
        logger.error("Ошибка: game_state_data не является словарем")
        return jsonify({'error': 'Invalid game state data format'}), 400

    num_cards = len(game_state_data.get('selected_cards', []))
    ai_settings = game_state_data.get('ai_settings', {})
    ai_type = ai_settings.get('aiType', 'mccfr')

    try:
        # Обработка и валидация данных
        selected_cards_data = game_state_data.get('selected_cards', [])
        if not isinstance(selected_cards_data, list):
            logger.error("Ошибка: selected_cards не является списком")
            return jsonify({'error': 'Invalid selected_cards format'}), 400
        selected_cards = [Card.from_dict(card) for card in selected_cards_data]
        logger.debug(f"Обработанные selected_cards: {selected_cards}")

        discarded_cards_data = game_state_data.get('discarded_cards', [])
        if not isinstance(discarded_cards_data, list):
            logger.error("Ошибка: discarded_cards не является списком")
            return jsonify({'error': 'Invalid discarded_cards format'}), 400
        discarded_cards = [Card.from_dict(card) for card in discarded_cards_data]
        logger.debug(f"Обработанные discarded_cards: {discarded_cards}")

        board_data = game_state_data.get('board', {})
        if not isinstance(board_data, dict):
            logger.error("Ошибка: board не является словарем")
            return jsonify({'error': 'Invalid board format'}), 400

        # Создание объекта доски с сохранением существующих карт
        board = ai_engine.Board()
        for line in ['top', 'middle', 'bottom']:
            line_data = board_data.get(line, [])
            if not isinstance(line_data, list):
                logger.error(f"Ошибка: board[{line}] не является списком")
                return jsonify({'error': f'Invalid board[{line}] format'}), 400
            for card_data in line_data:
                if card_data:
                    board.place_card(line, Card.from_dict(card_data))
        logger.debug(f"Обработанная доска: {board}")

        # Подсчет свободных слотов (оставляем для логгирования, но не используем для ограничений)
        occupied_slots = sum(1 for line in ['top', 'middle', 'bottom']
                             for card in session['game_state']['board'].get(line, [])
                             if card is not None)
        free_slots = 13 - occupied_slots

        logger.debug(f"Всего занято слотов: {occupied_slots}")
        logger.debug(f"Свободных слотов: {free_slots}")
        logger.debug(f"Количество выбранных карт: {num_cards}")

        # Создание состояния игры
        game_state = ai_engine.GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards_data,  # Передаём как есть (список словарей)
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
            logger.info(f"Игра окончена. Выплата: {payoff}, Роялти: {royalties}, Всего: {total_royalty}")

            # Сохранение прогресса AI (для MCCFR)
            if cfr_agent and ai_settings.get('aiType') == 'mccfr':
                try:
                    cfr_agent.save_progress() # Сначала сохраняем локально
                    logger.info("Прогресс AI сохранен локально.")
                    if github_utils.save_ai_progress_to_github():  # Попытка сохранить на GitHub
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

        # Получение следующих доступных слотов
        next_available_slots = get_next_available_slots(session['game_state']['board'])
        logger.debug(f"Следующие доступные слоты ПЕРЕД ходом AI: {next_available_slots}")

    except (KeyError, TypeError, ValueError) as e:
        logger.exception("Исключение при настройке состояния игры:")
        return jsonify({'error': f"Error during game state setup: {e}"}), 500

    timeout_event = Event()
    result = {'move': None}

    # Выбор и выполнение хода AI
    try:
        if ai_type == 'mccfr':
            if cfr_agent is None:
                logger.error("Ошибка: MCCFR агент не инициализирован")
                return jsonify({'error': 'MCCFR agent not initialized'}), 500
            ai_thread = Thread(target=cfr_agent.get_move,
                             args=(game_state, timeout_event, result)) # убрали , num_cards
        else:  # ai_type == 'random'
            ai_thread = Thread(target=random_agent.get_move,
                             args=(game_state, timeout_event, result)) # убрали , num_cards

        ai_thread.start()
        ai_thread.join(timeout=int(ai_settings.get('aiTime', 5)))

        if ai_thread.is_alive():
            timeout_event.set()
            ai_thread.join()
            logger.warning("Время ожидания хода AI истекло")
            return jsonify({'error': 'AI move timed out'}), 504

        move = result.get('move')
        if move is None or 'error' in move:
            logger.error(f"Ошибка хода AI: {move.get('error', 'Unknown error')}")
            return jsonify({'error': move.get('error', 'Unknown error')}), 500

        logger.debug(f"Получен ход AI: {move}")

    except Exception as e:
        logger.exception("Исключение при выполнении хода AI:")
        return jsonify({'error': f"Error during AI move execution: {e}"}), 500

    # Сериализация и отправка ответа
    try:
        serialized_move = serialize_move(move, next_available_slots)
        logger.debug(f"Сериализованный ход: {serialized_move}")


        #  Обновляем состояние сессии *ДО* расчета роялти
        if move:
            logger.info("Обновление состояния игры в сессии (перед расчетом роялти)")
            current_board = session['game_state']['board']

            # Добавляем только новые карты, сохраняя существующие
            for line in ['top', 'middle', 'bottom']:
                if line in move:
                    new_line = move[line]  # Это список объектов Card (или пустой список)
                    current_line = current_board[line]  # Это тоже список
                    slot_index = next_available_slots[line] # Смотрим, куда ставить

                    for new_card in new_line: # Итерируем по *новым* картам
                        if new_card is not None:
                            if slot_index < len(current_line):
                                current_line[slot_index] = new_card  # Кладем карту
                                slot_index += 1 # Двигаем индекс
                            else:
                                logger.error(f"Ошибка: Недостаточно места в линии {line}!")

            session['game_state']['board'] = current_board

            # Обновляем selected_cards и discarded_cards
            session['game_state']['selected_cards'] = []
            if 'discarded' in move and move['discarded']:
                if isinstance(move['discarded'], list):
                     session['game_state']['discarded_cards'].extend(move['discarded'])
                else:
                    session['game_state']['discarded_cards'].append(move['discarded'])


            logger.debug(f"Обновленная доска в сессии (после хода): {session['game_state']['board']}")


        # Расчет роялти *ПОСЛЕ* обновления доски в сессии.
        royalties = game_state.calculate_royalties()
        logger.debug(f"Рассчитанные роялти: {royalties}")  # Лог расчета
        total_royalty = sum(royalties.values())

        logger.debug(f"Отправка хода AI: {serialized_move}, Роялти: {royalties}, Всего роялти: {total_royalty}")
        return jsonify({
            'move': serialized_move,
            'royalties': royalties,
            'total_royalty': total_royalty
        }), 200

    except Exception as e:
        logger.exception("Исключение при сериализации и отправке ответа:")
        return jsonify({'error': f"Error during move serialization: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=10000)
