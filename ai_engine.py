import random
import itertools
from collections import defaultdict, Counter
import github_utils
import utils
from threading import Event, Thread, Lock
import time
import math
import logging
from typing import List, Dict, Optional, Tuple, Set

# Настройка логирования
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafeResult:
    """Потокобезопасная обертка для результата AI с улучшенной обработкой ошибок."""
    def __init__(self):
        self._data = {'move': None}
        self._lock = Lock()

    def set_move(self, move):
        """Потокобезопасная установка хода."""
        with self._lock:
            self._data['move'] = move

    def get_move(self):
        """Потокобезопасное получение хода."""
        with self._lock:
            return self._data.get('move')

class Card:
    """Представление игральной карты с улучшенной валидацией."""
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['♥', '♦', '♣', '♠']

    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Must be one of: {self.RANKS}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Must be one of: {self.SUITS}")
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        if isinstance(other, dict):
            return self.rank == other.get('rank') and self.suit == other.get('suit')
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def to_dict(self) -> dict:
        """Преобразование карты в словарь."""
        return {'rank': self.rank, 'suit': self.suit}

    @staticmethod
    def from_dict(card_dict: dict) -> 'Card':
        """Создание карты из словаря с валидацией."""
        if not isinstance(card_dict, dict):
            raise ValueError("Input must be a dictionary")
        if 'rank' not in card_dict or 'suit' not in card_dict:
            raise ValueError("Dictionary must contain 'rank' and 'suit' keys")
        return Card(card_dict['rank'], card_dict['suit'])

    @staticmethod
    def get_all_cards() -> List['Card']:
        """Получение полной колоды карт."""
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

class Hand:
    """Представление руки игрока с потокобезопасностью."""
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards = cards if cards is not None else []
        self._lock = Lock()

    def add_card(self, card: Card):
        """Добавление карты с проверкой дубликатов."""
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        with self._lock:
            if card not in self.cards:  # Проверка на дубликаты
                self.cards.append(card)

    def remove_card(self, card: Card) -> bool:
        """Удаление карты с возвратом статуса операции."""
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        with self._lock:
            try:
                self.cards.remove(card)
                return True
            except ValueError:
                logger.error(f"Card {card} not found in hand: {self.cards}")
                return False

    def clear(self):
        """Очистка руки."""
        with self._lock:
            self.cards.clear()

    def get_cards(self) -> List[Card]:
        """Безопасное получение копии списка карт."""
        with self._lock:
            return self.cards.copy()

    def __repr__(self):
        return ', '.join(map(str, self.cards))

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

class Board:
    """Представление игровой доски с улучшенной обработкой карт."""
    def __init__(self):
        self.top: List[Optional[Card]] = []
        self.middle: List[Optional[Card]] = []
        self.bottom: List[Optional[Card]] = []
        self._lock = Lock()
        self._max_cards = {'top': 3, 'middle': 5, 'bottom': 5}

    def place_card(self, line: str, card: Card, position: Optional[int] = None) -> bool:
        """Размещение карты с проверкой валидности позиции."""
        if line not in ['top', 'middle', 'bottom']:
            raise ValueError(f"Invalid line: {line}")

        with self._lock:
            target_line = getattr(self, line)
            max_cards = self._max_cards[line]

            if position is not None:
                if position < 0 or position >= max_cards:
                    raise ValueError(f"Invalid position {position} for line {line}")
                if len(target_line) <= position:
                    target_line.extend([None] * (position - len(target_line) + 1))
                target_line[position] = card
            else:
                if len(target_line) >= max_cards:
                    raise ValueError(f"{line} line is full")
                target_line.append(card)
            return True

    def remove_card(self, line: str, position: int) -> Optional[Card]:
        """Удаление карты с возвратом удаленной карты."""
        if line not in ['top', 'middle', 'bottom']:
            raise ValueError(f"Invalid line: {line}")

        with self._lock:
            target_line = getattr(self, line)
            if 0 <= position < len(target_line):
                card = target_line[position]
                target_line[position] = None
                return card
            return None

    def is_full(self) -> bool:
        """Проверка заполненности доски."""
        with self._lock:
            return (len(self.top) == self._max_cards['top'] and
                    len(self.middle) == self._max_cards['middle'] and
                    len(self.bottom) == self._max_cards['bottom'] and
                    all(card is not None for card in self.top + self.middle + self.bottom))

    def clear(self):
        """Очистка доски."""
        with self._lock:
            self.top = []
            self.middle = []
            self.bottom = []

    def get_cards(self, line: str) -> List[Optional[Card]]:
        """Безопасное получение карт линии."""
        if line not in ['top', 'middle', 'bottom']:
            raise ValueError("Invalid line specified")
        with self._lock:
            return getattr(self, line).copy()

    def get_all_cards(self) -> List[Card]:
        """Получение всех карт на доске."""
        with self._lock:
            return [card for line in [self.top, self.middle, self.bottom]
                   for card in line if card is not None]

    def __repr__(self):
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

class GameState:
    """Представление состояния игры с улучшенной обработкой сброшенных карт."""
    def __init__(self, selected_cards: Optional[List[Card]] = None,
                 board: Optional[Board] = None,
                 discarded_cards: Optional[List[Card]] = None,
                 ai_settings: Optional[Dict] = None,
                 deck: Optional[List[Card]] = None):
        self.selected_cards = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board = board if board is not None else Board()
        self.discarded_cards = discarded_cards if discarded_cards is not None else []
        self.ai_settings = ai_settings if ai_settings is not None else {}
        self.current_player = 0
        self.deck = deck if deck is not None else self.create_deck()
        self.rank_map = {rank: i for i, rank in enumerate(Card.RANKS)}
        self.suit_map = {suit: i for i, suit in enumerate(Card.SUITS)}
        self._lock = Lock()
        self._used_cards: Set[Tuple[str, str]] = set()  # Для отслеживания использованных карт
        self._update_used_cards()

    def _update_used_cards(self):
        """Обновление множества использованных карт."""
        with self._lock:
            self._used_cards.clear()
            # Добавляем карты с доски
            for card in self.board.get_all_cards():
                self._used_cards.add((card.rank, card.suit))
            # Добавляем сброшенные карты
            for card in self.discarded_cards:
                self._used_cards.add((card.rank, card.suit))
            # Добавляем выбранные карты
            for card in self.selected_cards.get_cards():
                self._used_cards.add((card.rank, card.suit))

    def create_deck(self) -> List[Card]:
        """Создание новой колоды."""
        return Card.get_all_cards()

    def get_current_player(self) -> int:
        """Получение текущего игрока."""
        return self.current_player

    def is_terminal(self) -> bool:
        """Проверка завершения игры."""
        return self.board.is_full()

    def get_num_cards_to_draw(self) -> int:
        """Определение количества карт для взятия."""
        with self._lock:
            placed_cards = len(self.board.get_all_cards())
            if placed_cards == 5:
                return 3
            elif placed_cards in [7, 10]:
                return 3
            return 0

    def is_card_available(self, card: Card) -> bool:
        """Проверка доступности карты."""
        with self._lock:
            return (card.rank, card.suit) not in self._used_cards

    def get_available_cards(self) -> List[Card]:
        """Получение списка доступных карт с учетом сброшенных."""
        with self._lock:
            return [card for card in self.deck 
                   if (card.rank, card.suit) not in self._used_cards]

    def discard_card(self, card: Card) -> bool:
        """Сброс карты с обновлением состояния."""
        with self._lock:
            if card not in self.discarded_cards:
                self.discarded_cards.append(card)
                self._used_cards.add((card.rank, card.suit))
                return True
            return False

    def get_actions(self) -> List[Dict]:
        """Получение доступных действий с учетом сброшенных карт."""
        if self.is_terminal():
            return []

        try:
            num_cards = len(self.selected_cards)
            actions = []
            placement_mode = self.ai_settings.get('placementMode', 'standard')

            logger.debug(f"get_actions called - cards: {num_cards}, mode: {placement_mode}")

            if num_cards == 0:
                return []

            if placement_mode == "first_deal":
                return self._get_first_deal_actions()
            elif placement_mode == "standard":
                return self._get_standard_actions()
            elif placement_mode == "fantasy":
                return self._get_fantasy_actions()
            elif placement_mode == "free":
                return self._get_free_actions()
            else:
                logger.error(f"Unknown placement mode: {placement_mode}")
                return []

        except Exception as e:
            logger.exception(f"Error in get_actions: {e}")
            return []

    def _get_first_deal_actions(self) -> List[Dict]:
        """Обработка первой раздачи с учетом сброшенных карт."""
        actions = []
        try:
            available_cards = [card for card in self.selected_cards.cards 
                             if self.is_card_available(card)]
            
            for p in itertools.permutations(available_cards):
                if len(p) >= 5:  # Проверяем, достаточно ли карт
                    actions.append({
                        'top': list(p[:1]),
                        'middle': list(p[1:3]),
                        'bottom': list(p[3:5]),
                        'discarded': None
                    })
        except Exception as e:
            logger.error(f"Error in first deal actions: {e}")
        return actions

    def _get_standard_actions(self) -> List[Dict]:
        """Обработка стандартного хода с учетом сброшенных карт."""
        actions = []
        try:
            available_cards = [card for card in self.selected_cards.cards 
                             if self.is_card_available(card)]
            
            for discarded_index in range(len(available_cards)):
                remaining_cards = [card for i, card in enumerate(available_cards) 
                                 if i != discarded_index]
                
                for top_count in range(min(len(remaining_cards) + 1, 
                                         3 - len(self.board.top))):
                    for middle_count in range(min(len(remaining_cards) - top_count + 1,
                                                5 - len(self.board.middle))):
                        bottom_count = len(remaining_cards) - top_count - middle_count
                        if bottom_count <= (5 - len(self.board.bottom)):
                            action = {
                                'top': remaining_cards[:top_count],
                                'middle': remaining_cards[top_count:top_count + middle_count],
                                'bottom': remaining_cards[top_count + middle_count:],
                                'discarded': available_cards[discarded_index]
                            }
                            actions.append(action)
        except Exception as e:
            logger.error(f"Error in standard actions: {e}")
        return actions

    def _get_fantasy_actions(self) -> List[Dict]:
        """Обработка фэнтези-режима с учетом сброшенных карт."""
        try:
            if self.ai_settings.get('fantasyMode'):
                return self._get_fantasy_repeat_actions()
            else:
                return self._get_fantasy_entry_actions()
        except Exception as e:
            logger.error(f"Error in fantasy actions: {e}")
            return []

    def _get_fantasy_repeat_actions(self) -> List[Dict]:
        """Обработка повторных фэнтези-действий."""
        valid_repeats = []
        try:
            available_cards = [card for card in self.selected_cards.cards 
                             if self.is_card_available(card)]
            
            for p in itertools.permutations(available_cards):
                action = {
                    'top': list(p[:3]),
                    'middle': list(p[3:8]),
                    'bottom': list(p[8:13]),
                    'discarded': list(p[13:])
                }
                if self.is_valid_fantasy_repeat(action):
                    valid_repeats.append(action)
            
            if valid_repeats:
                return sorted(valid_repeats, 
                            key=lambda a: self.calculate_action_royalty(a), 
                            reverse=True)
            else:
                return [self._create_standard_fantasy_action(p) 
                       for p in itertools.permutations(available_cards)]
        except Exception as e:
            logger.error(f"Error in fantasy repeat actions: {e}")
            return []

    def _get_fantasy_entry_actions(self) -> List[Dict]:
        """Обработка входных фэнтези-действий."""
        try:
            available_cards = [card for card in self.selected_cards.cards 
                             if self.is_card_available(card)]
            
            valid_entries = []
            for p in itertools.permutations(available_cards):
                action = self._create_standard_fantasy_action(p)
                if self.is_valid_fantasy_entry(action):
                    valid_entries.append(action)
            
            return valid_entries if valid_entries else [
                self._create_standard_fantasy_action(p)
                for p in itertools.permutations(available_cards)
            ]
        except Exception as e:
            logger.error(f"Error in fantasy entry actions: {e}")
            return []

    def _create_standard_fantasy_action(self, cards: tuple) -> Dict:
        """Создание стандартного фэнтези-действия."""
        return {
            'top': list(cards[:3]),
            'middle': list(cards[3:8]),
            'bottom': list(cards[8:13]),
            'discarded': list(cards[13:])
        }

    def apply_action(self, action: Dict) -> 'GameState':
        """Применение действия с обработкой сброшенных карт."""
        try:
            new_board = Board()
            # Копируем существующие карты
            for line in ['top', 'middle', 'bottom']:
                current_cards = self.board.get_cards(line)
                for card in current_cards:
                    if card and self.is_card_available(card):
                        new_board.place_card(line, card)

            # Добавляем новые карты из действия
            for line in ['top', 'middle', 'bottom']:
                if line in action:
                    for card in action[line]:
                        if card and self.is_card_available(card):
                            new_board.place_card(line, card)

            # Обрабатываем сброшенные карты
            new_discarded_cards = self.discarded_cards.copy()
            if 'discarded' in action and action['discarded']:
                if isinstance(action['discarded'], list):
                    for card in action['discarded']:
                        if card not in new_discarded_cards:
                            new_discarded_cards.append(card)
                else:
                    if action['discarded'] not in new_discarded_cards:
                        new_discarded_cards.append(action['discarded'])

            return GameState(
                selected_cards=Hand(),
                board=new_board,
                discarded_cards=new_discarded_cards,
                ai_settings=self.ai_settings,
                deck=self.deck[:]
            )
        except Exception as e:
            logger.error(f"Error applying action: {e}")
            raise

    def get_information_set(self) -> str:
        """Получение информационного набора с учетом сброшенных карт."""
        try:
            def sort_cards(cards):
                return sorted(cards, key=lambda card: (
                    self.rank_map[card.rank], 
                    self.suit_map[card.suit]
                ))

            components = []
            # Добавляем информацию о доске
            for prefix, cards in [
                ('T', self.board.get_cards('top')),
                ('M', self.board.get_cards('middle')),
                ('B', self.board.get_cards('bottom'))
            ]:
                sorted_cards = sort_cards([c for c in cards if c is not None])
                cards_str = ','.join(str(card) for card in sorted_cards)
                components.append(f"{prefix}:{cards_str}")

            # Добавляем информацию о сброшенных картах
            discarded_str = ','.join(str(card) for card in sort_cards(self.discarded_cards))
            components.append(f"D:{discarded_str}")

            # Добавляем информацию о выбранных картах
            selected_str = ','.join(str(card) for card in sort_cards(self.selected_cards.cards))
            components.append(f"S:{selected_str}")

            return '|'.join(components)
        except Exception as e:
            logger.error(f"Error generating information set: {e}")
            return "ERROR"

    def get_payoff(self) -> float:
        """Получение выигрыша с учетом сброшенных карт."""
        if not self.is_terminal():
            raise ValueError("Game is not in terminal state")

        try:
            if self.is_dead_hand():
                return -self.calculate_royalties()
            return self.calculate_royalties()
        except Exception as e:
            logger.error(f"Error calculating payoff: {e}")
            return 0

    def is_dead_hand(self) -> bool:
        """Проверка на мертвую руку с улучшенной валидацией."""
        try:
            if not self.board.is_full():
                return False

            top_rank, _ = self.evaluate_hand(self.board.get_cards('top'))
            middle_rank, _ = self.evaluate_hand(self.board.get_cards('middle'))
            bottom_rank, _ = self.evaluate_hand(self.board.get_cards('bottom'))

            return top_rank > middle_rank or middle_rank > bottom_rank
        except Exception as e:
            logger.error(f"Error checking dead hand: {e}")
            return True  # В случае ошибки безопаснее считать руку мертвой

    def evaluate_hand(self, cards: List[Card]) -> Tuple[int, float]:
        """Оценка комбинации с улучшенной обработкой ошибок."""
        try:
            if not cards or not all(isinstance(card, Card) for card in cards):
                return 11, 0  # Invalid hand rank

            n = len(cards)
            if n == 5:
                return self._evaluate_five_card_hand(cards)
            elif n == 3:
                return self._evaluate_three_card_hand(cards)
            else:
                return 11, 0  # Invalid hand size
        except Exception as e:
            logger.error(f"Error evaluating hand: {e}")
            return 11, 0

    def _evaluate_five_card_hand(self, cards: List[Card]) -> Tuple[int, float]:
        """Оценка пятикарточной комбинации."""
        try:
            if self.is_royal_flush(cards):
                return 1, 25.0
            if self.is_straight_flush(cards):
                high_card = max(cards, key=lambda c: self.rank_map[c.rank])
                return 2, 15.0 + self.rank_map[high_card.rank] / 100

            if self.is_four_of_a_kind(cards):
                quad_rank = [card.rank for card in cards 
                           if sum(1 for c in cards if c.rank == card.rank) == 4][0]
                return 3, 10.0 + self.rank_map[quad_rank] / 100

            if self.is_full_house(cards):
                trip_rank = [card.rank for card in cards 
                           if sum(1 for c in cards if c.rank == card.rank) == 3][0]
                return 4, 6.0 + self.rank_map[trip_rank] / 100

            if self.is_flush(cards):
                score = 4.0 + sum(self.rank_map[card.rank] for card in cards) / 1000
                return 5, score

            if self.is_straight(cards):
                score = 2.0 + sum(self.rank_map[card.rank] for card in cards) / 1000
                return 6, score

            if self.is_three_of_a_kind(cards):
                trip_rank = [card.rank for card in cards 
                           if sum(1 for c in cards if c.rank == card.rank) == 3][0]
                return 7, 2.0 + self.rank_map[trip_rank] / 100

            if self.is_two_pair(cards):
                pairs = sorted([self.rank_map[card.rank] for card in cards 
                              if sum(1 for c in cards if c.rank == card.rank) == 2],
                             reverse=True)
                return 8, sum(pairs) / 1000

            if self.is_one_pair(cards):
                pair_rank = [card.rank for card in cards 
                           if sum(1 for c in cards if c.rank == card.rank) == 2][0]
                return 9, self.rank_map[pair_rank] / 1000

            # High Card
            score = sum(self.rank_map[card.rank] for card in cards) / 10000
            return 10, score

        except Exception as e:
            logger.error(f"Error evaluating five-card hand: {e}")
            return 11, 0

    def _evaluate_three_card_hand(self, cards: List[Card]) -> Tuple[int, float]:
        """Оценка трехкарточной комбинации."""
        try:
            if self.is_three_of_a_kind(cards):
                rank = cards[0].rank  # В сете все ранги одинаковые
                return 7, 10.0 + self.rank_map[rank]

            if self.is_one_pair(cards):
                pair_rank = [card.rank for card in cards 
                           if sum(1 for c in cards if c.rank == card.rank) == 2][0]
                return 8, self.get_pair_bonus(cards)

            # High Card
            return 9, self.get_high_card_bonus(cards)

        except Exception as e:
            logger.error(f"Error evaluating three-card hand: {e}")
            return 11, 0

    def calculate_royalties(self) -> Dict[str, int]:
        """Расчет роялти с учетом сброшенных карт."""
        try:
            if self.is_dead_hand():
                return {'top': 0, 'middle': 0, 'bottom': 0}

            royalties = {}
            for line in ['top', 'middle', 'bottom']:
                cards = self.board.get_cards(line)
                royalties[line] = self.get_line_royalties(line, cards)

            return royalties
        except Exception as e:
            logger.error(f"Error calculating royalties: {e}")
            return {'top': 0, 'middle': 0, 'bottom': 0}

    def get_line_royalties(self, line: str, cards: List[Card]) -> int:
        """Расчет роялти для линии."""
        try:
            if not cards:
                return 0

            rank, _ = self.evaluate_hand(cards)
            
            if line == 'top':
                if rank == 7:  # Three of a Kind
                    return 10 + self.rank_map[cards[0].rank]
                elif rank == 8:  # One Pair
                    return self.get_pair_bonus(cards)
                elif rank == 9:  # High Card
                    return self.get_high_card_bonus(cards)
                    
            elif line == 'middle':
                if rank <= 6:  # Royal Flush to Straight
                    return self.get_royalties_for_hand(rank) * 2
                    
            elif line == 'bottom':
                if rank <= 6:  # Royal Flush to Straight
                    return self.get_royalties_for_hand(rank)
                    
            return 0
        except Exception as e:
            logger.error(f"Error calculating line royalties: {e}")
            return 0

    def get_royalties_for_hand(self, hand_rank: int) -> int:
        """Получение базовых роялти для комбинации."""
        try:
            royalties = {
                1: 25,  # Royal Flush
                2: 15,  # Straight Flush
                3: 10,  # Four of a Kind
                4: 6,   # Full House
                5: 4,   # Flush
                6: 2    # Straight
            }
            return royalties.get(hand_rank, 0)
        except Exception as e:
            logger.error(f"Error getting hand royalties: {e}")
            return 0

    # Методы проверки комбинаций
    def is_royal_flush(self, cards: List[Card]) -> bool:
        """Проверка на роял-флеш."""
        try:
            if not self.is_flush(cards):
                return False
            ranks = sorted([self.rank_map[card.rank] for card in cards])
            return ranks == [8, 9, 10, 11, 12]  # 10, J, Q, K, A
        except Exception as e:
            logger.error(f"Error checking royal flush: {e}")
            return False

    def is_straight_flush(self, cards: List[Card]) -> bool:
        """Проверка на стрит-флеш."""
        try:
            return self.is_straight(cards) and self.is_flush(cards)
        except Exception as e:
            logger.error(f"Error checking straight flush: {e}")
            return False

    def is_four_of_a_kind(self, cards: List[Card]) -> bool:
        """Проверка на каре."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return 4 in rank_counts.values()
        except Exception as e:
            logger.error(f"Error checking four of a kind: {e}")
            return False

    def is_full_house(self, cards: List[Card]) -> bool:
        """Проверка на фулл-хаус."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return 3 in rank_counts.values() and 2 in rank_counts.values()
        except Exception as e:
            logger.error(f"Error checking full house: {e}")
            return False

    def is_flush(self, cards: List[Card]) -> bool:
        """Проверка на флеш."""
        try:
            return len(set(card.suit for card in cards)) == 1
        except Exception as e:
            logger.error(f"Error checking flush: {e}")
            return False

    def is_straight(self, cards: List[Card]) -> bool:
        """Проверка на стрит с учетом особых случаев."""
        try:
            ranks = sorted([self.rank_map[card.rank] for card in cards])
            
            # Проверка на колесо (A,2,3,4,5)
            if ranks == [0, 1, 2, 3, 12]:
                return True
                
            # Обычная проверка на стрит
            return all(ranks[i+1] - ranks[i] == 1 for i in range(len(ranks)-1))
        except Exception as e:
            logger.error(f"Error checking straight: {e}")
            return False

    def is_three_of_a_kind(self, cards: List[Card]) -> bool:
        """Проверка на тройку."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return 3 in rank_counts.values()
        except Exception as e:
            logger.error(f"Error checking three of a kind: {e}")
            return False

    def is_two_pair(self, cards: List[Card]) -> bool:
        """Проверка на две пары."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return list(rank_counts.values()).count(2) == 2
        except Exception as e:
            logger.error(f"Error checking two pair: {e}")
            return False

    def is_one_pair(self, cards: List[Card]) -> bool:
        """Проверка на пару."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return 2 in rank_counts.values()
        except Exception as e:
            logger.error(f"Error checking one pair: {e}")
            return False

    def get_pair_bonus(self, cards: List[Card]) -> int:
        """Расчет бонуса за пару."""
        try:
            if len(cards) != 3:
                return 0
            rank_counts = Counter(card.rank for card in cards)
            for rank, count in rank_counts.items():
                if count == 2:
                    rank_index = Card.RANKS.index(rank)
                    if rank_index >= Card.RANKS.index('6'):
                        return 1 + rank_index - Card.RANKS.index('6')
            return 0
        except Exception as e:
            logger.error(f"Error calculating pair bonus: {e}")
            return 0

    def get_high_card_bonus(self, cards: List[Card]) -> int:
        """Расчет бонуса за старшую карту."""
        try:
            if len(cards) != 3:
                return 0
            ranks = [card.rank for card in cards]
            if len(set(ranks)) == 3:  # Все карты разные
                highest_rank = max(ranks, key=lambda r: Card.RANKS.index(r))
                return 1 if highest_rank == 'A' else 0
            return 0
        except Exception as e:
            logger.error(f"Error calculating high card bonus: {e}")
            return 0

class CFRNode:
    """Узел для CFR с улучшенной потокобезопасностью."""
    def __init__(self, actions):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.actions = actions
        self._lock = Lock()

    def get_strategy(self, realization_weight: float) -> Dict[str, float]:
        """Получение текущей стратегии."""
        with self._lock:
            try:
                normalizing_sum = 0
                strategy = defaultdict(float)
                
                for action in self.actions:
                    strategy[action] = max(0, self.regret_sum[action])
                    normalizing_sum += strategy[action]

                for action in self.actions:
                    if normalizing_sum > 0:
                        strategy[action] /= normalizing_sum
                    else:
                        strategy[action] = 1.0 / len(self.actions)
                    
                    self.strategy_sum[action] += realization_weight * strategy[action]
                
                return dict(strategy)
            except Exception as e:
                logger.error(f"Error calculating strategy: {e}")
                return {action: 1.0 / len(self.actions) for action in self.actions}

    def get_average_strategy(self) -> Dict[str, float]:
        """Получение усредненной стратегии."""
        with self._lock:
            try:
                avg_strategy = defaultdict(float)
                normalizing_sum = sum(self.strategy_sum.values())
                
                if normalizing_sum > 0:
                    for action in self.actions:
                        avg_strategy[action] = self.strategy_sum[action] / normalizing_sum
                else:
                    for action in self.actions:
                        avg_strategy[action] = 1.0 / len(self.actions)
                
                return dict(avg_strategy)
            except Exception as e:
                logger.error(f"Error calculating average strategy: {e}")
                return {action: 1.0 / len(self.actions) for action in self.actions}

class CFRAgent:
    """Реализация CFR агента для игры."""
    def __init__(self, iterations: int = 1000, stop_threshold: float = 0.001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 100
        self._lock = Lock()

    def train(self, timeout_event: Event, result: Dict):
        """Обучение агента."""
        try:
            for i in range(self.iterations):
                if timeout_event.is_set():
                    logger.info(f"Training interrupted after {i} iterations")
                    break

                all_cards = Card.get_all_cards()
                random.shuffle(all_cards)
                game_state = GameState(deck=all_cards)
                game_state.selected_cards = Hand(all_cards[:5])

                self.cfr(game_state, 1, 1, timeout_event, result, i + 1)

                if (i + 1) % self.save_interval == 0:
                    self.save_ai_progress_to_github()
                    if self.check_convergence():
                        break

        except Exception as e:
            logger.exception(f"Error during training: {e}")

    def cfr(self, game_state: GameState, p0: float, p1: float, 
            timeout_event: Event, result: Dict, iteration: int) -> float:
        """Выполнение CFR алгоритма."""
        if timeout_event.is_set():
            return 0

        try:
            if game_state.is_terminal():
                return game_state.get_payoff()

            player = game_state.get_current_player()
            info_set = game_state.get_information_set()

            with self._lock:
                if info_set not in self.nodes:
                    actions = game_state.get_actions()
                    if not actions:
                        return 0
                    self.nodes[info_set] = CFRNode(actions)
                node = self.nodes[info_set]

            strategy = node.get_strategy(p0 if player == 0 else p1)
            util = defaultdict(float)
            node_util = 0

            for action in node.actions:
                if timeout_event.is_set():
                    return 0

                next_state = game_state.apply_action(action)
                if player == 0:
                    util[action] = -self.cfr(next_state, p0 * strategy[action], 
                                           p1, timeout_event, result, iteration)
                else:
                    util[action] = -self.cfr(next_state, p0, 
                                           p1 * strategy[action], timeout_event, 
                                           result, iteration)
                node_util += strategy[action] * util[action]

            if not timeout_event.is_set():
                with self._lock:
                    if player == 0:
                        for action in node.actions:
                            node.regret_sum[action] += p1 * (util[action] - node_util)
                    else:
                        for action in node.actions:
                            node.regret_sum[action] += p0 * (util[action] - node_util)

            return node_util

        except Exception as e:
            logger.exception(f"Error in CFR calculation: {e}")
            return 0

    def get_move(self, game_state: GameState, timeout_event: Event, result: SafeResult):
        """Получение следующего хода."""
        try:
            actions = game_state.get_actions()
            if not actions:
                result.set_move({'error': 'No available moves'})
                return

            info_set = game_state.get_information_set()
            with self._lock:
                if info_set in self.nodes:
                    strategy = self.nodes[info_set].get_average_strategy()
                    best_move = max(strategy.items(), key=lambda x: x[1])[0]
                else:
                    best_move = random.choice(actions)

            result.set_move(best_move)

        except Exception as e:
            logger.exception(f"Error getting move: {e}")
            result.set_move({'error': str(e)})

    def save_ai_progress_to_github(self) -> bool:
        """Сохранение прогресса на GitHub."""
        try:
            with self._lock:
                data = {
                    'nodes': self.nodes,
                    'iterations': self.iterations,
                    'stop_threshold': self.stop_threshold
                }
            return github_utils.save_ai_progress(data, 'cfr_data.pkl')
        except Exception as e:
            logger.error(f"Error saving AI progress: {e}")
            return False

    def load_ai_progress_from_github(self) -> bool:
        """Загрузка прогресса с GitHub."""
        try:
            if github_utils.load_ai_progress_from_github():
                data = utils.load_ai_progress()
                if data and isinstance(data, dict):
                    with self._lock:
                        self.nodes = data.get('nodes', {})
                        self.iterations = data.get('iterations', self.iterations)
                        self.stop_threshold = data.get('stop_threshold', self.stop_threshold)
                    return True
            return False
        except Exception as e:
            logger.error(f"Error loading AI progress: {e}")
            return False

    def check_convergence(self) -> bool:
        """Проверка сходимости."""
        try:
            with self._lock:
                for node in self.nodes.values():
                    avg_strategy = node.get_average_strategy()
                    uniform_strategy = 1.0 / len(node.actions)
                    for prob in avg_strategy.values():
                        if abs(prob - uniform_strategy) > self.stop_threshold:
                            return False
                return True
        except Exception as e:
            logger.error(f"Error checking convergence: {e}")
            return False
