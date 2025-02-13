import random
import itertools
from collections import defaultdict, Counter
import utils
from threading import Event, Thread
import time
import math
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

class Card:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['♥', '♦', '♣', '♠']

    def __init__(self, rank, suit):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Rank must be one of: {self.RANKS}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Suit must be one of: {self.SUITS}")
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

    def to_dict(self):
        return {'rank': self.rank, 'suit': self.suit}

    @staticmethod
    def from_dict(card_dict):
        return Card(card_dict['rank'], card_dict['suit'])

    @staticmethod
    def get_all_cards(): # Исправлено: убрали self
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

class Hand:
    def __init__(self, cards=None):
        self.cards = cards if cards is not None else []

    def add_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def remove_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        try:
            self.cards.remove(card)
        except ValueError:
            print(f"Card {card} not found in hand: {self.cards}")

    def __repr__(self):
        return ', '.join(map(str, self.cards))

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

class Board:
    def __init__(self):
        self.top = []
        self.middle = []
        self.bottom = []

    def place_card(self, line, card):
        if line == 'top':
            if len(self.top) >= 3:
                raise ValueError("Top line is full")
            self.top.append(card)
        elif line == 'middle':
            if len(self.middle) >= 5:
                raise ValueError("Middle line is full")
            self.middle.append(card)
        elif line == 'bottom':
            if len(self.bottom) >= 5:
                raise ValueError("Bottom line is full")
            self.bottom.append(card)
        else:
            raise ValueError(f"Invalid line: {line}. Line must be one of: 'top', 'middle', 'bottom'")

    def is_full(self):
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self):
        self.top = []
        self.middle = []
        self.bottom = []

    def __repr__(self):
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def get_cards(self, line):
        if line == 'top':
            return self.top
        elif line == 'middle':
            return self.middle
        elif line == 'bottom':
            return self.bottom
        else:
            raise ValueError("Invalid line specified")

class GameState:
    def __init__(self, selected_cards=None, board=None, discarded_cards=None, ai_settings=None, deck=None):
        self.selected_cards = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board = board if board is not None else Board()
        self.discarded_cards = discarded_cards if discarded_cards is not None else []
        self.ai_settings = ai_settings if ai_settings is not None else {}
        self.current_player = 0
        self.deck = deck if deck is not None else self.create_deck() # Use provided deck or create a new one
        self.rank_map = {rank: i for i, rank in enumerate(Card.RANKS)}
        self.suit_map = {suit: i for i, suit in enumerate(Card.SUITS)}

    def create_deck(self):
        """Creates a standard deck of 52 cards."""
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

    def get_current_player(self):
        return self.current_player

    def is_terminal(self):
        """Checks if the game is in a terminal state (all lines are full)."""
        return self.board.is_full()

    def get_num_cards_to_draw(self):
        """Returns the number of cards to draw based on the current game state."""
        placed_cards = sum(len(row) for row in [self.board.top, self.board.middle, self.board.bottom])
        if placed_cards == 5:
            return 3
        elif placed_cards == 7 or placed_cards == 10: # Combined conditions for 7 and 10 cards
            return 3
        elif placed_cards >= 13:
            return 0
        else:
            return 0

    def get_available_cards(self):
        """Returns a list of cards that are still available in the deck."""

        used_cards = set(self.discarded_cards + self.board.top + self.board.middle + self.board.bottom + list(self.selected_cards))
        available_cards = [card for card in self.deck if card not in used_cards]
        return available_cards

    def get_actions(self):
        """Returns the valid actions for the current state."""
        logger.debug("get_actions - START") # ADDED LOG
        if self.is_terminal():
            logger.debug("get_actions - Game is terminal, returning empty actions") # ADDED LOG
            return []  # No actions in a terminal state

        num_cards = len(self.selected_cards)
        actions = []
        placement_mode = self.ai_settings.get('placementMode', 'standard')

        # Добавляем проверку на выбывшие карты
        # ИСПРАВЛЕНО: used_cards теперь вычисляется корректно
        used_cards = set()
        for line in [self.board.top, self.board.middle, self.board.bottom]:
            used_cards.update([card for card in line if card is not None])  # Только карты на доске
        used_cards.update(
                self.discarded_cards
            )

        if num_cards > 0:
            try:
                if placement_mode == "first_deal":  # Первая раздача (5 карт)
                    # Размещаем все 5 карт
                    for p in itertools.permutations(self.selected_cards.cards):
                        action = {
                            'top': list(p[:1]),
                            'middle': list(p[1:3]),
                            'bottom': list(p[3:5]),
                            'discarded': None  # Ничего не сбрасываем
                        }


                elif placement_mode == "standard":  # Стандартный ход (3 карты)
                    placed_cards = sum(len(row) for row in [self.board.top, self.board.middle, self.board.bottom])
                    if placed_cards == 11:  # Последний ход
                        # Размещаем 2 карты, 1 сбрасываем
                        for discarded_index in range(3):
                            remaining_cards = [card for i, card in enumerate(self.selected_cards.cards) if i != discarded_index]
                            for top_count in range(min(len(remaining_cards) + 1, 3 - len(self.board.top))):
                                for middle_count in range(min(len(remaining_cards) - top_count + 1, 5 - len(self.board.middle))):
                                    bottom_count = len(remaining_cards) - top_count - middle_count
                                    if bottom_count <= (5 - len(self.board.bottom)):
                                        action = {
                                            'top': remaining_cards[:top_count],
                                            'middle': remaining_cards[top_count:top_count + middle_count],
                                            'bottom': remaining_cards[top_count + middle_count:],
                                            'discarded': self.selected_cards.cards[discarded_index]
                                        }
                                        actions.append(action)

                    else: # Обычный ход (не последний)
                        # Размещаем 2 карты, 1 сбрасываем
                        for discarded_index in range(3):
                            remaining_cards = [card for i, card in enumerate(self.selected_cards.cards) if i != discarded_index]
                            for top_count in range(min(len(remaining_cards) + 1, 3 - len(self.board.top))):
                                for middle_count in range(min(len(remaining_cards) - top_count + 1, 5 - len(self.board.middle))):
                                    bottom_count = len(remaining_cards) - top_count - middle_count
                                    if bottom_count <= (5 - len(self.board.bottom)):
                                        action = {
                                            'top': remaining_cards[:top_count],
                                            'middle': remaining_cards[top_count:top_count + middle_count],
                                            'bottom': remaining_cards[top_count + middle_count:],
                                            'discarded': self.selected_cards.cards[discarded_index]
                                        }
                                        actions.append(action)

                        if len(self.selected_cards) == 3: # Только если 3 карты выбрано
                            cards_to_place = list(self.selected_cards.cards) # Преобразуем в список для индексации
                            for i in range(3): # 3 варианта сброса
                                discard_card = cards_to_place[i]
                                remaining_cards = cards_to_place[:i] + cards_to_place[i+1:] # 2 оставшиеся карты

                                # Варианты размещения 2 карт по 3 линиям
                                for top_cards_count in range(3): # 0, 1, 2 карты в верхний ряд
                                    for middle_cards_count in range(3 - top_cards_count): # Остаток в средний, до 2
                                        bottom_cards_count = 2 - top_cards_count - middle_cards_count # В нижний все что осталось

                                        action = {
                                            'top': remaining_cards[:top_cards_count],
                                            'middle': remaining_cards[top_cards_count:top_cards_count + middle_cards_count],
                                            'bottom': remaining_cards[top_cards_count + middle_cards_count:],
                                            'discarded': discard_card
                                        }
                                        actions.append(action)


                elif placement_mode == "fantasy":
                    # ... (логика для фантазии) ...
                    if self.ai_settings.get('fantasyMode'):
                        valid_fantasy_repeats = []
                        for p in itertools.permutations(self.selected_cards.cards):
                            action = {
                                'top': list(p[:3]),
                                'middle': list(p[3:8]),
                                'bottom': list(p[8:13]),
                                'discarded': list(p[13:])  # Всегда сбрасываем одну карту
                            }
                            if self.is_valid_fantasy_repeat(action):
                                valid_fantasy_repeats.append(action)
                        if valid_fantasy_repeats:
                            actions = sorted(valid_fantasy_repeats, key=lambda a: self.calculate_action_royalty(a), reverse=True)
                        else:  # Если повтор фантазии невозможен
                            actions = sorted([
                                {
                                    'top': list(p[:3]),
                                    'middle': list(p[3:8]),
                                    'bottom': list(p[8:13]),
                                    'discarded': list(p[13:])  # Всегда сбрасываем одну карту
                                } for p in itertools.permutations(self.selected_cards.cards)
                            ], key=lambda a: self.calculate_action_royalty(a), reverse=True)

                    else: # Если не в режиме фантазии, то проверяем можем ли войти
                        valid_fantasy_entries = []
                        for p in itertools.permutations(self.selected_cards.cards):
                                action = {
                                    'top': list(p[:3]),
                                    'middle': list(p[3:8]),
                                    'bottom': list(p[8:13]),
                                    'discarded': list(p[13:])  # Всегда сбрасываем одну карту
                                }
                                if self.is_valid_fantasy_entry(action):
                                    valid_fantasy_entries.append(action)

                        if valid_fantasy_entries: # Если можем войти в фантазию, предлагаем эти варианты
                            actions = valid_fantasy_entries
                        else: # Не можем войти в фантазию
                            actions = [
                                {
                                    'top': list(p[:3]),
                                    'middle': list(p[3:8]),
                                    'bottom': list(p[8:13]),
                                    'discarded': list(p[13:])  # Всегда сбрасываем одну карту
                                } for p in itertools.permutations(self.selected_cards.cards)
                            ]

                elif placement_mode == "free":  # Свободный режим
                    # Размещаем *все* выбранные карты, если возможно
                    remaining_cards = list(self.selected_cards.cards)
                    for top_count in range(min(len(remaining_cards) + 1, 3 - len(self.board.top))):
                        for middle_count in range(min(len(remaining_cards) - top_count + 1, 5 - len(self.board.middle))):
                            bottom_count = len(remaining_cards) - top_count - middle_count
                            if bottom_count <= (5 - len(self.board.bottom)):
                                action = {
                                    'top': remaining_cards[:top_count],
                                    'middle': remaining_cards[top_count:top_count + middle_cards_count],
                                    'bottom': remaining_cards[top_count + middle_cards_count:],
                                    'discarded': None  # Ничего не сбрасываем
                                }
                                actions.append(action)


            except Exception as e:
                logger.exception(f"Error in get_actions: {e}")
                return []

        logger.debug(f"Generated actions: {actions}")
        logger.debug("get_actions - END") # ADDED LOG
        return actions

    # ... (остальной код ai_engine.py без изменений)
    def is_valid_fantasy_entry(self, action):
        """Checks if an action leads to a valid fantasy mode entry."""
        new_board = Board()
        new_board.top = self.board.top + action.get('top', [])
        new_board.middle = self.board.middle + action.get('middle', [])
        new_board.bottom = self.board.bottom + action.get('bottom', [])

        temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(new_board.top)
        return top_rank <= 8 and new_board.top[0].rank in ['Q', 'K', 'A']

    def is_valid_fantasy_repeat(self, action):
        """Checks if an action leads to a valid fantasy mode repeat."""
        new_board = Board()
        new_board.top = self.board.top + action.get('top', [])
        new_board.middle = self.board.middle + action.get('middle', [])
        new_board.bottom = self.board.bottom + action.get('bottom', [])

        temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(new_board.top)
        bottom_rank, _ = temp_state.evaluate_hand(new_board.bottom)

        if top_rank == 7: # Set in top row
            return True
        if bottom_rank <= 3: # Four of a Kind or better in bottom row
            return True

        return False

    def calculate_action_royalty(self, action):
        """Calculates the royalty for a given action."""
        new_board = Board()
        new_board.top = self.board.top + action.get('top', [])
        new_board.middle = self.board.middle + action.get('middle', [])
        new_board.bottom = self.board.bottom + action.get('bottom', [])

        temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
        return temp_state.calculate_royalties()

    def apply_action(self, action):
        """Applies an action to the current state and returns the new state."""

        new_board = Board()
        new_board.top = self.board.top + action.get('top', [])
        new_board.middle = self.board.middle + action.get('middle', [])
        new_board.bottom = self.board.bottom + action.get('bottom', [])

        new_discarded_cards = self.discarded_cards[:]
        if 'discarded' in action and action['discarded']:
            if isinstance(action['discarded'], list):
                new_discarded_cards.extend(action['discarded'])
            else:
                new_discarded_cards.append(action['discarded'])

        new_game_state = GameState(
            selected_cards=Hand(),  # Selected cards are now on the board or discarded
            board=new_board,
            discarded_cards=new_discarded_cards,
            ai_settings=self.ai_settings,
            deck=self.deck[:] # Pass a copy of the deck
        )

        return new_game_state

    def get_information_set(self):
        """Returns a string representation of the current information set."""

        def card_to_string(card):
            return str(card)

        def sort_cards(cards):
            return sorted(cards, key=lambda card: (self.rank_map[card.rank], self.suit_map[card.suit]))

        top_str = ','.join(map(card_to_string, sort_cards(self.board.top)))
        middle_str = ','.join(map(card_to_string, sort_cards(self.board.middle)))
        bottom_str = ','.join(map(card_to_string, sort_cards(self.board.bottom)))
        discarded_str = ','.join(map(card_to_string, sort_cards(self.discarded_cards)))
        selected_str = ','.join(map(card_to_string, sort_cards(self.selected_cards.cards)))


        return f"T:{top_str}|M:{middle_str}|B:{bottom_str}|D:{discarded_str}|S:{selected_str}"

    def get_payoff(self):
        """Calculates the payoff for the current state."""
        if not self.is_terminal():
            raise ValueError("Game is not in a terminal state")

        if self.is_dead_hand():
            return -self.calculate_royalties()  # Negative royalties for a dead hand

        return self.calculate_royalties()

    def is_dead_hand(self):
        """Checks if the hand is a dead hand (invalid combination order)."""
        if not self.board.is_full():
            return False

        top_rank, _ = self.evaluate_hand(self.board.top)
        middle_rank, _ = self.evaluate_hand(self.board.middle)
        bottom_rank, _ = self.evaluate_hand(self.board.bottom)

        return top_rank > middle_rank or middle_rank > bottom_rank

    def get_line_royalties(self, line):
        """Calculates royalties for a specific line."""
        cards = getattr(self.board, line)
        if not cards:
            return 0

        rank, _ = self.evaluate_hand(cards)
        if line == 'top':
            if rank == 7:  # Three of a Kind
                return 10 + Card.RANKS.index(cards[0].rank)
            elif rank == 8:  # One Pair
                return self.get_pair_bonus(cards)
            elif rank == 9:  # High Card
                return self.get_high_card_bonus(cards)
        elif line == 'middle':
            if rank <= 6:
                return self.get_royalties_for_hand(rank) * 2
        elif line == 'bottom':
            if rank <= 6:
                return self.get_royalties_for_hand(rank)
        return 0

    def calculate_royalties(self):
        """Calculates royalties for the current state based on the rules."""
        if self.is_dead_hand():
            return 0

        royalties = {}
        lines = {'top': self.board.top, 'middle': self.board.middle, 'bottom': self.board.bottom}

        for line_name, cards in lines.items():
            royalties[line_name] = self.get_line_royalties(line_name)

        return royalties

    def get_royalties_for_hand(self, hand_rank):
        if hand_rank == 1: # Royal Flush
            return 25
        elif hand_rank == 2: # Straight Flush
            return 15
        elif hand_rank == 3: # Four of a Kind
            return 10
        elif hand_rank == 4: # Full House
            return 6
        elif hand_rank == 5: # Flush
            return 4
        elif hand_rank == 6: # Straight
            return 2
        return 0

    def get_line_score(self, line, cards):
        """Calculates the score for a specific line based on hand rankings."""
        if not cards:
            return 0

        rank, score = self.evaluate_hand(cards)
        return score

    def get_pair_bonus(self, cards):
        """Calculates the bonus for a pair in the top line."""
        if len(cards) != 3:
            return 0
        ranks = [card.rank for card in cards]
        for rank in Card.RANKS[::-1]:  # Iterate in reverse to find the highest pair first
            if ranks.count(rank) == 2:
                return 1 + Card.RANKS.index(rank) - Card.RANKS.index('6') if rank >= '6' else 0

    def get_high_card_bonus(self, cards):
        """Calculates the bonus for a high card in the top line."""
        if len(cards) != 3 or not all(isinstance(card, Card) for card in cards):
            return 0
        ranks = [card.rank for card in cards]
        if len(set(ranks)) == 3:  # Three different ranks
            high_card = max(ranks, key=Card.RANKS.index)
            return 1 if high_card == 'A' else 0

    def get_fantasy_bonus(self):
        """Calculates the bonus for fantasy mode."""
        bonus = 0
        top_rank, _ = self.evaluate_hand(self.board.top)

        if top_rank <= 8 and self.board.top[0].rank in ['Q', 'K', 'A']: # QQ, KK, AA and better
            if self.ai_settings.get('fantasyType') == 'progressive':
                if self.board.top[0].rank == 'Q':
                    bonus += 14 # 14 cards for QQ
                elif self.board.top[0].rank == 'K':
                    bonus += 15 # 15 cards for KK
                elif self.board.top[0].rank == 'A':
                    bonus += 16 # 16 cards for AA
                elif top_rank == 7: # Set
                    bonus += 17 # 17 cards for set from 222 to AAA
            else: # Normal fantasy
                bonus += 14 # 14 cards

            if self.is_fantasy_repeat():
                bonus += 14 # Fantasy repeat - 14 cards (regardless of type)

        return bonus

    def is_fantasy_repeat(self):
        """Checks if the conditions for fantasy repeat are met."""
        top_rank, _ = self.evaluate_hand(self.board.top)
        bottom_rank, _ = self.evaluate_hand(self.board.bottom)

        if top_rank == 7: # Set in top row
            return True
        if bottom_rank <= 3: # Four of a Kind or better in bottom row
            return True

        return False

    def evaluate_hand(self, cards):
        """Evaluates the hand and returns a rank (lower is better) and a score."""

        if not cards or not all(isinstance(card, Card) for card in cards):
            return 11, 0  # Return a low rank for invalid hands

        n = len(cards)
        if n == 5:
            if self.is_royal_flush(cards):
                return 1, 25
            if self.is_straight_flush(cards):
                return 2, 15
            if self.is_four_of_a_kind(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 4][0]
                return 3, 10 + Card.RANKS.index(rank) / 100
            if self.is_full_house(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 3][0]
                return 4, 6 + Card.RANKS.index(rank) / 100
            if self.is_flush(cards):
                score = 4 + sum(Card.RANKS.index(card.rank) for card in cards) / 1000  # Adjusted score for Flush
                return 5, score
            if self.is_straight(cards):
                score = 2 + sum(Card.RANKS.index(card.rank) for card in cards) / 1000  # Adjusted score for Straight
                return 6, score
            if self.is_three_of_a_kind(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 3][0]
                return 7, 2 + Card.RANKS.index(rank) / 100
            if self.is_two_pair(cards):
                ranks = sorted([Card.RANKS.index(card.rank) for card in cards if [card.rank for card in cards].count(card.rank) == 2], reverse=True)
                return 8, sum(ranks) / 1000  # Adjusted score for Two Pair
            if self.is_one_pair(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 2][0]
                return 9, Card.RANKS.index(rank) / 1000  # Adjusted score for One Pair
            # High Card
            score = sum(Card.RANKS.index(card.rank) for card in cards) / 10000 # Adjusted score for High Card
            return 10, score

        elif n == 3:
            if self.is_three_of_a_kind(cards):
                rank = cards[0].rank  # All cards have the same rank in a set
                return 7, 10 + Card.RANKS.index(rank)
            if self.is_one_pair(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 2][0]
                return 8, self.get_pair_bonus(cards)  # Use pair bonus for 3-card hands
            # High Card
            return 9, self.get_high_card_bonus(cards) # Use high card bonus for 3-card hands

        else:
            return 11, 0  # Return a low rank for invalid hands

    def is_royal_flush(self, cards):
        if not self.is_flush(cards):
            return False
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        return ranks == [8, 9, 10, 11, 12]  # 10, J, Q, K, A

    def is_straight_flush(self, cards):
        return self.is_straight(cards) and self.is_flush(cards)

    def is_four_of_a_kind(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 4 for r in ranks)

    def is_full_house(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks) and any(ranks.count(r) == 2 for r in ranks)

    def is_flush(self, cards):
        suits = [card.suit for card in cards]
        return len(set(suits)) == 1

    def is_straight(self, cards):
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        if ranks == [0, 1, 2, 3, 12]:  # Special case for A, 2, 3, 4, 5
            return True
        return all(ranks[i + 1] - ranks[i] == 1 for i in range(len(ranks) - 1))

    def is_three_of_a_kind(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks)

    def is_two_pair(self, cards):
        ranks = [card.rank for card in cards]
        pairs = [r for r in set(ranks) if ranks.count(r) == 2]
        return len(pairs) == 2

    def is_one_pair(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 2 for r in ranks)

class CFRNode:
    def __init__(self, actions):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.actions = actions

    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        strategy = defaultdict(float)
        for a in self.actions:
            strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += strategy[a]

        for a in self.actions:
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0 / len(self.actions)
            self.strategy_sum[a] += realization_weight * strategy[a]
        return strategy

    def get_average_strategy(self):
        avg_strategy = defaultdict(float)
        normalizing_sum = sum(self.strategy_sum.values())
        if normalizing_sum > 0:
            for a in self.actions:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
        else:
            for a in self.actions:
                avg_strategy[a] = 1.0 / len(self.actions)
        return avg_strategy

class CFRAgent:
    def __init__(self, iterations=1000, stop_threshold=0.001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 100  # Сохраняем каждые 100

    def cfr(self, game_state, p0, p1, timeout_event, result, iteration):
        if timeout_event.is_set():
            logger.info("CFR timed out!")
            return 0

        if game_state.is_terminal():
            payoff = game_state.get_payoff()
            logger.debug(f"cfr called in terminal state. Payoff: {payoff}")
            return payoff

        player = game_state.get_current_player()
        info_set = game_state.get_information_set()
        logger.debug(f"cfr called for info_set: {info_set}, player: {player}")

        if info_set not in self.nodes:
            actions = game_state.get_actions()
            if not actions:
                logger.debug("No actions available for this state.")
                return 0
            self.nodes[info_set] = CFRNode(actions)
        node = self.nodes[info_set]

        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = defaultdict(float)
        node_util = 0

        for a in node.actions:
            if timeout_event.is_set():
                logger.info("CFR timed out during action loop!")
                return 0

            next_state = game_state.apply_action(a)
            if player == 0:
                util[a] = -self.cfr(next_state, p0 * strategy[a], p1, timeout_event, result, iteration)
            else:
                util[a] = -self.cfr(next_state, p0, p1 * strategy[a], timeout_event, result, iteration)
            node_util += strategy[a] * util[a]

        if player == 0:
            for a in node.actions:
                node.regret_sum[a] += p1 * (util[a] - node_util)
        else:
            for a in node.actions:
                node.regret_sum[a] += p0 * (util[a] - node_util)

        logger.debug(f"cfr returning for info_set: {info_set}, node_util: {node_util}")
        return node_util

    def train(self, timeout_event, result):
        for i in range(self.iterations):
            if timeout_event.is_set():
                logger.info(f"Training interrupted after {i} iterations due to timeout.")
                break  # Exit the loop if timeout is signaled

            all_cards = Card.get_all_cards()
            random.shuffle(all_cards)
            game_state = GameState(deck=all_cards) # Pass the shuffled deck to GameState
            game_state.selected_cards = Hand(all_cards[:5])  # Сразу выбираем 5 карт
            self.cfr(game_state, 1, 1, timeout_event, result, i + 1) # Передаем номер итерации

            if (i + 1) % self.save_interval == 0: # Check every save_interval iterations
                logger.info(f"Iteration {i+1} of {self.iterations} complete. Saving progress...")
                self.save_progress() # Сохраняем прогресс
                if self.check_convergence():
                    logger.info(f"CFR agent converged after {i + 1} iterations.")
                    break

    def check_convergence(self):
        for node in self.nodes.values():
            avg_strategy = node.get_average_strategy()
            for action, prob in avg_strategy.items():
                if abs(prob - 1.0 / len(node.actions)) > self.stop_threshold:
                    return False
        return True

    def get_move(self, game_state, timeout_event, result):
        logger.debug("Inside get_move")
        actions = game_state.get_actions()
        logger.debug(f"Available actions: {actions}")

        if not actions:
            result['move'] = {'error': 'Нет доступных ходов'}
            logger.debug("No actions available, returning error.")
            return

        info_set = game_state.get_information_set()
        logger.debug(f"Info set: {info_set}")

        if info_set in self.nodes:
            strategy = self.nodes[info_set].get_average_strategy()
            logger.debug(f"Strategy: {strategy}")
            best_move = max(strategy, key=strategy.get) if strategy else None
        else:
            logger.debug("Info set not found in nodes, choosing random action.")
            best_move = random.choice(actions) if actions else None

        logger.debug(f"Selected move: {best_move}")
        result['move'] = best_move


    def evaluate_move(self, game_state, action, timeout_event):
        """
        Оценивает ход, используя комбинацию обученной стратегии MCCFR и эвристики.
        """
        next_state = game_state.apply_action(action)  # Применяем действие
        info_set = next_state.get_information_set()    # Получаем инфосет

        if info_set in self.nodes:
            # Если узел есть в памяти MCCFR, используем его стратегию
            node = self.nodes[info_set]
            strategy = node.get_average_strategy()
            expected_value = 0
            for a, prob in strategy.items():
                if timeout_event.is_set():
                    return 0  # Return 0 if timeout occurred
                expected_value += prob * self.get_action_value(next_state, a, timeout_event)
            return expected_value
        else:
            # Если узла нет, используем shallow_search с baseline_evaluation
            return self.shallow_search(next_state, 2, timeout_event)

    def shallow_search(self, state, depth, timeout_event):
        """
        Поверхностный поиск с ограниченной глубиной.
        """
        if depth == 0 or state.is_terminal() or timeout_event.is_set():
            return self.baseline_evaluation(state) # Используем новую baseline_evaluation

        best_value = float('-inf')
        for action in state.get_actions():
            if timeout_event.is_set():
                return 0  # Return 0 if timeout occurred
            value = -self.shallow_search(state.apply_action(action), depth - 1, timeout_event)
            best_value = max(best_value, value)
        return best_value

    def get_action_value(self, state, action, timeout_event):
        """
        Оценивает ценность действия, усредняя результаты нескольких симуляций.
        """
        num_simulations = 10
        total_score = 0

        for _ in range(num_simulations):
            if timeout_event.is_set():
                return 0  # Return 0 if timeout occurred
            simulated_state = state.apply_action(action)
            while not simulated_state.is_terminal():
                actions = simulated_state.get_actions()
                if not actions:
                    break  # No valid actions available
                random_action = random.choice(actions)
                simulated_state = simulated_state.apply_action(random_action)
            total_score += self.baseline_evaluation(simulated_state) # Используем новую

        return total_score / num_simulations if num_simulations > 0 else 0


    def calculate_potential(self, cards, line, board, available_cards):
        """Calculates the potential for improvement of a given hand."""
        potential = 0
        num_cards = len(cards)

        if num_cards < 5 and line != 'top':
            # Check for straight potential
            if self.is_straight_potential(cards, available_cards):
                potential += 0.5

            # Check for flush potential
            if self.is_flush_potential(cards, available_cards):
                potential += 0.7

        if num_cards == 2 and line == 'top':
            # Check for pair potential to make a set
            if self.is_pair_potential(cards, available_cards):
                potential += 0.3

        return potential

    def is_flush_potential(self, cards, available_cards):
        """Checks if there's potential to make a flush."""
        if len(cards) < 2:
            return False

        suit_counts = defaultdict(int)
        for card in cards:
            suit_counts[card.suit] += 1

        for suit, count in suit_counts.items():
            if count >= 2:  # At least 2 cards of the same suit
                remaining_needed = 5 - count
                available_of_suit = sum(1 for card in available_cards if card.suit == suit)
                if available_of_suit >= remaining_needed:
                    return True
        return False

    def is_straight_potential(self, cards, available_cards):
        """Checks if there's potential to make a straight."""
        if len(cards) < 2:
            return False

        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        # Check for consecutive ranks
        for i in range(len(ranks) - 1):
            if ranks[i + 1] - ranks[i] == 1:
                return True

        # Check for one-gap straights (e.g., 2, 4, 5 or 7, 9, 10)
        if len(ranks) >= 2:
            for i in range(len(ranks) - 1):
                if ranks[i + 1] - ranks[i] == 2:
                    needed_rank = ranks[i] + 1
                    if any(Card.RANKS.index(card.rank) == needed_rank for card in available_cards):
                        return True

        # Check for two-gap straights (e.g., 2, 5 or 6, 9)
        if len(ranks) >= 2:
            for i in range(len(ranks) - 1):
                if ranks[i + 1] - ranks[i] == 3:
                    needed_ranks = [ranks[i] + 1, ranks[i] + 2]
                    if sum(1 for card in available_cards if Card.RANKS.index(card.rank) in needed_ranks) >= 1:
                        return True

        # Special case for A, 2, 3, 4, 5 straight
        if ranks == [0, 1, 2, 3]:
            if any(card.rank == 'A' for card in available_cards):
                return True
        if ranks == [0, 1, 2, 12]:
            if any(card.rank == '3' for card in available_cards):
                return True
        if ranks == [0, 1, 11, 12]:
            if any(card.rank == '2' for card in available_cards):
                return True
        if ranks == [0, 10, 11, 12]:
            if any(card.rank == '1' for card in available_cards):
                return True

        return False

    def is_pair_potential(self, cards, available_cards):
        """Checks if there's potential to make a set (three of a kind) from a pair."""
        if len(cards) != 2:
            return False

        if cards[0].rank == cards[1].rank:
            rank = cards[0].rank
            if sum(1 for card in available_cards if card.rank == rank) >= 1:
                return True

        return False

    def evaluate_line_strength(self, cards, line):
        """Evaluates the strength of a line with more granularity."""
        if not cards:
            return 0

        rank, _ = self.evaluate_hand(cards)
        score = 0

        if line == 'top':
            if rank == 7:  # Three of a Kind
                score = 15 + Card.RANKS.index(cards[0].rank) * 0.1
            elif rank == 8:  # One Pair
                score = 5 + self.get_pair_bonus(cards)
            elif rank == 9:  # High Card
                score = 1 + self.get_high_card_bonus(cards)
        elif line == 'middle':
            if rank == 1:  # Royal Flush
                score = 150
            elif rank == 2:  # Straight Flush
                score = 100 + Card.RANKS.index(cards[-1].rank) * 0.1  # High card matters
            elif rank == 3:  # Four of a Kind
                score = 80 + Card.RANKS.index(cards[1].rank) * 0.1
            elif rank == 4:  # Full House
                score = 60 + Card.RANKS.index(cards[2].rank) * 0.1  # Rank of the three-of-a-kind
            elif rank == 5:  # Flush
                score = 40 + Card.RANKS.index(cards[-1].rank) * 0.1  # High card matters
            elif rank == 6:  # Straight
                score = 20 + Card.RANKS.index(cards[-1].rank) * 0.1  # High card matters
            elif rank == 7:  # Three of a Kind
                score = 10 + Card.RANKS.index(cards[0].rank) * 0.1
            elif rank == 8:  # Two Pair
                score = 5 + Card.RANKS.index(cards[1].rank) * 0.01 + Card.RANKS.index(cards[3].rank) * 0.001
            elif rank == 9:  # One Pair
                score = 2 + Card.RANKS.index(cards[1].rank) * 0.01
            elif rank == 10:  # High Card
                score = Card.RANKS.index(cards[-1].rank) * 0.001
        elif line == 'bottom':
            # Similar scoring as middle line, but with slightly lower weights
            if rank == 1:  # Royal Flush
                score = 120
            elif rank == 2:  # Straight Flush
                score = 80 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 3:  # Four of a Kind
                score = 60 + Card.RANKS.index(cards[1].rank) * 0.1
            elif rank == 4:  # Full House
                score = 40 + Card.RANKS.index(cards[2].rank) * 0.1
            elif rank == 5:  # Flush
                score = 30 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 6:  # Straight
                score = 15 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 7:  # Three of a Kind
                score = 8 + Card.RANKS.index(cards[0].rank) * 0.1
            elif rank == 8:  # Two Pair
                score = 4 + Card.RANKS.index(cards[1].rank) * 0.01 + Card.RANKS.index(cards[3].rank) * 0.001
            elif rank == 9:  # One Pair
                score = 1 + Card.RANKS.index(cards[1].rank) * 0.01
            elif rank == 10:  # High Card
                score = Card.RANKS.index(cards[-1].rank) * 0.001

        return score

    def baseline_evaluation(self, state):
        """
        Улучшенная эвристическая оценка состояния игры.
        """
        if state.is_dead_hand():
            return -1000  # Существенный штраф за мертвую руку

        COMBINATION_WEIGHTS = {
            'royal_flush': 100,
            'straight_flush': 90,
            'four_of_a_kind': 80,
            'full_house': 70,
            'flush': 60,
            'straight': 50,
            'three_of_a_kind': 40,
            'two_pair': 30,
            'pair': 20,
            'high_card': 10
        }

        ROW_MULTIPLIERS = {
            'top': 1.0,
            'middle': 1.2,
            'bottom': 1.5
        }

        total_score = 0

        def evaluate_partial_combination(cards, row_type):
            """Оценка потенциала неполной комбинации"""
            if not cards:
                return 0

            score = 0
            ranks = [card.rank for card in cards]
            suits = [card.suit for card in cards]

            # Потенциал флеша
            suit_counts = Counter(suits)
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            if row_type in ['middle', 'bottom']:
                if max_suit_count >= 3:
                    score += 15 * max_suit_count

            # Потенциал стрита
            rank_values = sorted([Card.RANKS.index(rank) for rank in ranks])
            if len(rank_values) >= 3:
                gaps = sum(rank_values[i+1] - rank_values[i] - 1 for i in range(len(rank_values)-1))
                if gaps <= 2:  # Возможность собрать стрит
                    score += 10 * (5 - gaps)

            # Оценка парных комбинаций
            rank_counts = Counter(ranks)
            for rank, count in rank_counts.items():
                rank_value = Card.RANKS.index(rank)
                if count == 2:
                    score += 20 + rank_value
                elif count == 3:
                    score += 40 + rank_value * 1.5

            return score

        # Оценка по рядам
        rows = {'top': state.board.top, 'middle': state.board.middle, 'bottom': state.board.bottom}
        for row_name, cards in rows.items():
            row_score = 0

            # Определение текущей комбинации
            combination = self.identify_combination(cards)
            if combination:
                row_score += COMBINATION_WEIGHTS[combination]

            # Оценка потенциала неполной комбинации
            potential_score = evaluate_partial_combination(cards, row_name)
            row_score += potential_score

            # Штраф за неправильное количество карт
            max_cards = {'top': 3, 'middle': 5, 'bottom': 5}
            if len(cards) > max_cards[row_name]:
                row_score -= 50

            # Применяем множитель ряда
            row_score *= ROW_MULTIPLIERS[row_name]

            total_score += row_score

        # Дополнительные стратегические бонусы
        if self.is_bottom_stronger_than_middle(state):
            total_score += 30
        if self.is_middle_stronger_than_top(state):
            total_score += 20

        # Штраф за нарушение правила силы рядов
        if not self.check_row_strength_rule(state):
            total_score -= 100

        # Учет сброшенных карт
        for card in state.discarded_cards:
            rank_value = Card.RANKS.index(card.rank)
            total_score -= rank_value * 0.5  # Небольшой штраф за сброс высоких карт

        return total_score

    def identify_combination(self, cards):
        """Определяет тип комбинации."""
        if not cards:
            return None
        if len(cards) < 3 and len(cards) !=5 : # добавили and len(cards) !=5
            return None
        if len(cards) == 3:
            if self.is_three_of_a_kind(cards):
                return 'three_of_a_kind'
            if self.is_one_pair(cards):
                return 'pair'
            else: return 'high_card'
        if self.is_royal_flush(cards):
            return 'royal_flush'
        elif self.is_straight_flush(cards):
            return 'straight_flush'
        elif self.is_four_of_a_kind(cards):
            return 'four_of_a_kind'
        elif self.is_full_house(cards):
            return 'full_house'
        elif self.is_flush(cards):
            return 'flush'
        elif self.is_straight(cards):
            return 'straight'
        elif self.is_three_of_a_kind(cards):
            return 'three_of_a_kind'
        elif self.is_two_pair(cards):
            return 'two_pair'
        elif self.is_one_pair(cards):
            return 'pair'
        else:
            return 'high_card'

    def is_bottom_stronger_than_middle(self, state):
        """Проверяет, сильнее ли нижний ряд среднего."""
        if len(state.board.bottom) < 5 or len(state.board.middle) < 5:
            return False  # Недостаточно карт для сравнения
        bottom_rank, _ = self.evaluate_hand(state.board.bottom)
        middle_rank, _ = self.evaluate_hand(state.board.middle)
        return bottom_rank <= middle_rank

    def is_middle_stronger_than_top(self, state):
        """Проверяет, сильнее ли средний ряд верхнего."""
        if len(state.board.middle) < 5 or len(state.board.top) < 3:
            return False
        middle_rank, _ = self.evaluate_hand(state.board.middle)
        top_rank, _ = self.evaluate_hand(state.board.top)
        return middle_rank <= top_rank


    def check_row_strength_rule(self, state):
        """
        Проверяет, соблюдается ли правило силы рядов (bottom >= middle >= top).
        Возвращает True, если правило соблюдается, и False в противном случае.
        """
        if not state.board.is_full():
            return True  # Если доска не заполнена, правило считается соблюденным

        top_rank, _ = self.evaluate_hand(state.board.top)
        middle_rank, _ = self.evaluate_hand(state.board.middle)
        bottom_rank, _ = self.evaluate_hand(state.board.bottom)

        # Чем меньше rank, тем сильнее комбинация.
        return bottom_rank <= middle_rank <= top_rank

    def save_progress(self):
        data = {
            'nodes': self.nodes,
            'iterations': self.iterations,
            'stop_threshold': self.stop_threshold
        }
        utils.save_ai_progress(data, 'cfr_data.pkl')


    def load_progress(self):
        data = utils.load_ai_progress('cfr_data.pkl')
        if data:
            self.nodes = data['nodes']
            self.iterations = data['iterations']
            self.stop_threshold = data.get('stop_threshold', 0.0001) # Default value if not present

class RandomAgent:
    def __init__(self):
        pass  # No initialization needed for a random agent

    def get_move(self, game_state, timeout_event, result):
        """Chooses a random valid move."""
        logger.debug("Inside RandomAgent get_move")
        actions = game_state.get_actions()
        logger.debug(f"Available actions: {actions}")

        if not actions:
            result['move'] = {'error': 'Нет доступных ходов'}
            logger.debug("No actions available, returning error.")
            return

        best_move = random.choice(actions) if actions else None
        logger.debug(f"Selected move: {best_move}")
        result['move'] = best_move  # Correctly return the move

    def evaluate_move(self, game_state, action, timeout_event):
        pass

    def shallow_search(self, state, depth, timeout_event):
        pass

    def get_action_value(self, state, action, timeout_event):
        pass

    def calculate_potential(self, cards, line, board, available_cards):
        pass

    def is_flush_potential(self, cards, available_cards):
        pass

    def is_straight_potential(self, cards, available_cards):
        pass

    def is_pair_potential(self, cards, available_cards):
        pass

    def evaluate_line_strength(self, cards, line):
        pass

    def baseline_evaluation(self, state):
        pass

    def identify_combination(self, cards):
        pass

    def is_bottom_stronger_than_middle(self, state):
        pass

    def is_middle_stronger_than_top(self, state):
        pass

    def check_row_strength_rule(self, state):
        pass

    def save_progress(self):
        pass

    def load_progress(self):
        pass
