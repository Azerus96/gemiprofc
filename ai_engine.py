import random
import itertools
from collections import defaultdict, Counter
import github_utils
import utils
from threading import Event, Thread, Lock
import time
import math
import logging
from typing import List, Dict, Optional, Tuple

# Настройка логирования
logger = logging.getLogger(__name__)

class SafeResult:
    """Потокобезопасная обертка для результата AI"""
    def __init__(self):
        self._result = {'move': None}
        self._lock = Lock()
    
    def set_move(self, move):
        with self._lock:
            self._result['move'] = move
    
    def get_move(self):
        with self._lock:
            return self._result.get('move')

class Card:
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
        return {'rank': self.rank, 'suit': self.suit}

    @staticmethod
    def from_dict(card_dict: dict) -> 'Card':
        if not isinstance(card_dict, dict):
            raise ValueError("Input must be a dictionary")
        if 'rank' not in card_dict or 'suit' not in card_dict:
            raise ValueError("Dictionary must contain 'rank' and 'suit' keys")
        return Card(card_dict['rank'], card_dict['suit'])

    @staticmethod
    def get_all_cards() -> List['Card']:
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

class Hand:
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards = cards if cards is not None else []
        self._lock = Lock()  # Добавляем блокировку для потокобезопасности

    def add_card(self, card: Card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        with self._lock:
            self.cards.append(card)

    def remove_card(self, card: Card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        with self._lock:
            try:
                self.cards.remove(card)
            except ValueError:
                logger.error(f"Card {card} not found in hand: {self.cards}")
                raise

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
        self.top: List[Optional[Card]] = []
        self.middle: List[Optional[Card]] = []
        self.bottom: List[Optional[Card]] = []
        self._lock = Lock()  # Добавляем блокировку для потокобезопасности

    def place_card(self, line: str, card: Card):
        with self._lock:
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
                raise ValueError(f"Invalid line: {line}")

    def is_full(self) -> bool:
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self):
        with self._lock:
            self.top = []
            self.middle = []
            self.bottom = []

    def __repr__(self):
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def get_cards(self, line: str) -> List[Optional[Card]]:
        if line not in ['top', 'middle', 'bottom']:
            raise ValueError("Invalid line specified")
        return getattr(self, line)
        
class GameState:
    def __init__(self, selected_cards: Optional[List[Card]] = None,
                 board: Optional[Board] = None,
                 discarded_cards: Optional[List[Card]] = None,
                 ai_settings: Optional[Dict] = None,
                 deck: Optional[List[Card]] = None):
        # Тело метода должно быть с отступом
        self.selected_cards = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board = board if board is not None else Board()
        self.discarded_cards = discarded_cards if discarded_cards is not None else []
        self.ai_settings = ai_settings if ai_settings is not None else {}
        self.current_player = 0
        self.deck = deck if deck is not None else self.create_deck()
        self.rank_map = {rank: i for i, rank in enumerate(Card.RANKS)}
        self.suit_map = {suit: i for i, suit in enumerate(Card.SUITS)}
        self._lock = Lock()

    def create_deck(self) -> List[Card]:
        """Creates a standard deck of 52 cards."""
        return Card.get_all_cards()

    def get_current_player(self) -> int:
        return self.current_player

    def is_terminal(self) -> bool:
        """Checks if the game is in a terminal state."""
        return self.board.is_full()

    def get_num_cards_to_draw(self) -> int:
        """Returns the number of cards to draw based on the current game state."""
        with self._lock:
            placed_cards = sum(len(row) for row in [self.board.top, self.board.middle, self.board.bottom])
            if placed_cards == 5:
                return 3
            elif placed_cards in [7, 10]:
                return 3
            return 0

    def get_available_cards(self) -> List[Card]:
        """Returns available cards with thread safety."""
        with self._lock:
            used_cards = set(self.discarded_cards + self.board.top + 
                           self.board.middle + self.board.bottom + 
                           list(self.selected_cards))
            return [card for card in self.deck if card not in used_cards]

    def get_actions(self) -> List[Dict]:
        """Returns valid actions with improved error handling."""
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
        """Handle first deal placement (5 cards)."""
        actions = []
        try:
            for p in itertools.permutations(self.selected_cards.cards):
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
        """Handle standard placement (3 cards, discard 1)."""
        actions = []
        try:
            for discarded_index in range(3):
                remaining_cards = [card for i, card in enumerate(self.selected_cards.cards) 
                                 if i != discarded_index]
                for top_count in range(min(len(remaining_cards) + 1, 3 - len(self.board.top))):
                    for middle_count in range(min(len(remaining_cards) - top_count + 1, 
                                               5 - len(self.board.middle))):
                        bottom_count = len(remaining_cards) - top_count - middle_count
                        if bottom_count <= (5 - len(self.board.bottom)):
                            action = {
                                'top': remaining_cards[:top_count],
                                'middle': remaining_cards[top_count:top_count + middle_count],
                                'bottom': remaining_cards[top_count + middle_count:],
                                'discarded': self.selected_cards.cards[discarded_index]
                            }
                            actions.append(action)
        except Exception as e:
            logger.error(f"Error in standard actions: {e}")
        return actions
    def _get_fantasy_actions(self) -> List[Dict]:
        """Handle fantasy mode actions with improved validation."""
        try:
            if self.ai_settings.get('fantasyMode'):
                return self._get_fantasy_repeat_actions()
            else:
                return self._get_fantasy_entry_actions()
        except Exception as e:
            logger.error(f"Error in fantasy actions: {e}")
            return []

    def _get_fantasy_repeat_actions(self) -> List[Dict]:
        """Handle fantasy repeat actions."""
        valid_repeats = []
        try:
            for p in itertools.permutations(self.selected_cards.cards):
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
                       for p in itertools.permutations(self.selected_cards.cards)]
        except Exception as e:
            logger.error(f"Error in fantasy repeat actions: {e}")
            return []

    def _get_fantasy_entry_actions(self) -> List[Dict]:
        """Handle fantasy entry actions."""
        try:
            valid_entries = []
            for p in itertools.permutations(self.selected_cards.cards):
                action = self._create_standard_fantasy_action(p)
                if self.is_valid_fantasy_entry(action):
                    valid_entries.append(action)
            
            return valid_entries if valid_entries else [
                self._create_standard_fantasy_action(p)
                for p in itertools.permutations(self.selected_cards.cards)
            ]
        except Exception as e:
            logger.error(f"Error in fantasy entry actions: {e}")
            return []

    def _create_standard_fantasy_action(self, cards: tuple) -> Dict:
        """Create a standard fantasy action from cards."""
        return {
            'top': list(cards[:3]),
            'middle': list(cards[3:8]),
            'bottom': list(cards[8:13]),
            'discarded': list(cards[13:])
        }

    def _get_free_actions(self) -> List[Dict]:
        """Handle free placement mode."""
        actions = []
        try:
            remaining_cards = list(self.selected_cards.cards)
            for top_count in range(min(len(remaining_cards) + 1, 3 - len(self.board.top))):
                for middle_count in range(min(len(remaining_cards) - top_count + 1, 
                                           5 - len(self.board.middle))):
                    bottom_count = len(remaining_cards) - top_count - middle_count
                    if bottom_count <= (5 - len(self.board.bottom)):
                        action = {
                            'top': remaining_cards[:top_count],
                            'middle': remaining_cards[top_count:top_count + middle_count],
                            'bottom': remaining_cards[top_count + middle_count:],
                            'discarded': None
                        }
                        actions.append(action)
        except Exception as e:
            logger.error(f"Error in free actions: {e}")
        return actions

    def is_valid_fantasy_entry(self, action: Dict) -> bool:
        """Validates fantasy mode entry with improved error handling."""
        try:
            new_board = Board()
            new_board.top = self.board.top + action.get('top', [])
            new_board.middle = self.board.middle + action.get('middle', [])
            new_board.bottom = self.board.bottom + action.get('bottom', [])

            temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
            if temp_state.is_dead_hand():
                return False

            top_rank, _ = temp_state.evaluate_hand(new_board.top)
            return (top_rank <= 8 and 
                    new_board.top and 
                    new_board.top[0].rank in ['Q', 'K', 'A'])
        except Exception as e:
            logger.error(f"Error in fantasy entry validation: {e}")
            return False

    def is_valid_fantasy_repeat(self, action: Dict) -> bool:
        """Validates fantasy repeat with improved error handling."""
        try:
            new_board = Board()
            new_board.top = self.board.top + action.get('top', [])
            new_board.middle = self.board.middle + action.get('middle', [])
            new_board.bottom = self.board.bottom + action.get('bottom', [])

            temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
            if temp_state.is_dead_hand():
                return False

            top_rank, _ = temp_state.evaluate_hand(new_board.top)
            bottom_rank, _ = temp_state.evaluate_hand(new_board.bottom)

            return top_rank == 7 or bottom_rank <= 3
        except Exception as e:
            logger.error(f"Error in fantasy repeat validation: {e}")
            return False
    def calculate_action_royalty(self, action: Dict) -> int:
        """Calculates royalty for an action with error handling."""
        try:
            new_board = Board()
            new_board.top = self.board.top + action.get('top', [])
            new_board.middle = self.board.middle + action.get('middle', [])
            new_board.bottom = self.board.bottom + action.get('bottom', [])

            temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
            return temp_state.calculate_royalties()
        except Exception as e:
            logger.error(f"Error calculating action royalty: {e}")
            return 0

    def apply_action(self, action: Dict) -> 'GameState':
        """Applies an action to create new state with validation."""
        try:
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
        """Returns string representation of current information set."""
        try:
            def sort_cards(cards):
                return sorted(cards, key=lambda card: (
                    self.rank_map[card.rank], 
                    self.suit_map[card.suit]
                ))

            components = []
            for prefix, cards in [
                ('T', self.board.top),
                ('M', self.board.middle),
                ('B', self.board.bottom),
                ('D', self.discarded_cards),
                ('S', self.selected_cards.cards)
            ]:
                sorted_cards = sort_cards(cards)
                cards_str = ','.join(str(card) for card in sorted_cards)
                components.append(f"{prefix}:{cards_str}")

            return '|'.join(components)
        except Exception as e:
            logger.error(f"Error generating information set: {e}")
            return "ERROR"

    def get_payoff(self) -> float:
        """Calculates payoff for terminal state with validation."""
        if not self.is_terminal():
            raise ValueError("Game is not in terminal state")

        try:
            return -self.calculate_royalties() if self.is_dead_hand() else self.calculate_royalties()
        except Exception as e:
            logger.error(f"Error calculating payoff: {e}")
            return 0

    def is_dead_hand(self) -> bool:
        """Checks if hand is dead with improved validation."""
        try:
            if not self.board.is_full():
                return False

            top_rank, _ = self.evaluate_hand(self.board.top)
            middle_rank, _ = self.evaluate_hand(self.board.middle)
            bottom_rank, _ = self.evaluate_hand(self.board.bottom)

            return top_rank > middle_rank or middle_rank > bottom_rank
        except Exception as e:
            logger.error(f"Error checking dead hand: {e}")
            return True  # Безопаснее считать руку мертвой в случае ошибки

    @staticmethod
    def _validate_cards_for_evaluation(cards: List[Card]) -> bool:
        """Validates cards before evaluation."""
        if not cards:
            return False
        if not all(isinstance(card, Card) for card in cards):
            return False
        return True

    def evaluate_hand(self, cards: List[Card]) -> Tuple[int, float]:
        """Evaluates poker hand with improved validation and error handling."""
        try:
            if not self._validate_cards_for_evaluation(cards):
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
        """Evaluates five-card poker hand with detailed scoring."""
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
        """Evaluates three-card poker hand."""
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

    def is_royal_flush(self, cards: List[Card]) -> bool:
        """Checks for royal flush with validation."""
        try:
            if not self.is_flush(cards):
                return False
            ranks = sorted([self.rank_map[card.rank] for card in cards])
            return ranks == [8, 9, 10, 11, 12]  # 10, J, Q, K, A
        except Exception as e:
            logger.error(f"Error checking royal flush: {e}")
            return False

    def is_straight_flush(self, cards: List[Card]) -> bool:
        """Checks for straight flush."""
        try:
            return self.is_straight(cards) and self.is_flush(cards)
        except Exception as e:
            logger.error(f"Error checking straight flush: {e}")
            return False

    def is_four_of_a_kind(self, cards: List[Card]) -> bool:
        """Checks for four of a kind with improved counting."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return any(count == 4 for count in rank_counts.values())
        except Exception as e:
            logger.error(f"Error checking four of a kind: {e}")
            return False

    def is_full_house(self, cards: List[Card]) -> bool:
        """Checks for full house with improved counting."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return 3 in rank_counts.values() and 2 in rank_counts.values()
        except Exception as e:
            logger.error(f"Error checking full house: {e}")
            return False

    def is_flush(self, cards: List[Card]) -> bool:
        """Checks for flush."""
        try:
            return len(set(card.suit for card in cards)) == 1
        except Exception as e:
            logger.error(f"Error checking flush: {e}")
            return False

    def is_straight(self, cards: List[Card]) -> bool:
        """Checks for straight with special case for wheel straight."""
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
        """Checks for three of a kind with improved counting."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return 3 in rank_counts.values()
        except Exception as e:
            logger.error(f"Error checking three of a kind: {e}")
            return False

    def is_two_pair(self, cards: List[Card]) -> bool:
        """Checks for two pair."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            pairs = [rank for rank, count in rank_counts.items() if count == 2]
            return len(pairs) == 2
        except Exception as e:
            logger.error(f"Error checking two pair: {e}")
            return False

    def is_one_pair(self, cards: List[Card]) -> bool:
        """Checks for one pair."""
        try:
            rank_counts = Counter(card.rank for card in cards)
            return 2 in rank_counts.values()
        except Exception as e:
            logger.error(f"Error checking one pair: {e}")
            return False

    def get_pair_bonus(self, cards: List[Card]) -> float:
        """Calculates bonus for pairs in top line."""
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

    def get_high_card_bonus(self, cards: List[Card]) -> float:
        """Calculates bonus for high cards in top line."""
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
    """Node in the CFR tree with thread-safe operations."""
    def __init__(self, actions):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.actions = actions
        self._lock = Lock()

    def get_strategy(self, realization_weight: float) -> Dict[str, float]:
        """Calculates current strategy with thread safety."""
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
        """Calculates average strategy with thread safety."""
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
    """CFR agent with improved error handling and thread safety."""
    def __init__(self, iterations: int = 1000, stop_threshold: float = 0.001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 100
        self._lock = Lock()

    def load_ai_progress_from_github(self) -> bool:
        """Loads AI progress from GitHub with error handling."""
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
            def save_ai_progress_to_github(self) -> bool:
        """Saves AI progress to GitHub with error handling."""
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

    def cfr(self, game_state: GameState, p0: float, p1: float, 
            timeout_event: Event, result: Dict, iteration: int) -> float:
        """Executes CFR algorithm with improved timeout handling."""
        if timeout_event.is_set():
            logger.info("CFR timed out!")
            return 0

        try:
            if game_state.is_terminal():
                payoff = game_state.get_payoff()
                logger.debug(f"Terminal state reached. Payoff: {payoff}")
                return payoff

            player = game_state.get_current_player()
            info_set = game_state.get_information_set()
            logger.debug(f"Processing info_set: {info_set}, player: {player}")

            # Инициализация узла при необходимости
            with self._lock:
                if info_set not in self.nodes:
                    actions = game_state.get_actions()
                    if not actions:
                        logger.debug("No actions available")
                        return 0
                    self.nodes[info_set] = CFRNode(actions)
                node = self.nodes[info_set]

            # Получение стратегии
            strategy = node.get_strategy(p0 if player == 0 else p1)
            util = defaultdict(float)
            node_util = 0

            # Рекурсивный обход
            for action in node.actions:
                if timeout_event.is_set():
                    logger.info("CFR timeout during action processing")
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

            # Обновление сожалений
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

    def train(self, timeout_event: Event, result: Dict):
        """Trains the agent with improved progress tracking."""
        try:
            for i in range(self.iterations):
                if timeout_event.is_set():
                    logger.info(f"Training interrupted after {i} iterations")
                    break

                # Подготовка начального состояния
                all_cards = Card.get_all_cards()
                random.shuffle(all_cards)
                game_state = GameState(deck=all_cards)
                game_state.selected_cards = Hand(all_cards[:5])

                # Выполнение итерации CFR
                self.cfr(game_state, 1, 1, timeout_event, result, i + 1)

                # Сохранение прогресса
                if (i + 1) % self.save_interval == 0:
                    logger.info(f"Iteration {i+1}/{self.iterations} complete")
                    self.save_ai_progress_to_github()
                    
                    if self.check_convergence():
                        logger.info(f"Convergence reached after {i + 1} iterations")
                        break

        except Exception as e:
            logger.exception(f"Error during training: {e}")

    def check_convergence(self) -> bool:
        """Checks for convergence with improved criteria."""
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

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict):
        """Gets the next move with improved error handling."""
        logger.debug("Calculating next move")
        try:
            actions = game_state.get_actions()
            logger.debug(f"Available actions: {len(actions)}")

            if not actions:
                result['move'] = {'error': 'No available moves'}
                logger.debug("No actions available")
                return

            info_set = game_state.get_information_set()
            logger.debug(f"Current info set: {info_set}")

            with self._lock:
                if info_set in self.nodes:
                    strategy = self.nodes[info_set].get_average_strategy()
                    logger.debug(f"Using learned strategy: {strategy}")
                    best_move = max(strategy.items(), key=lambda x: x[1])[0]
                else:
                    logger.debug("Using random strategy (info set not found)")
                    best_move = random.choice(actions)

            logger.debug(f"Selected move: {best_move}")
            result['move'] = best_move

        except Exception as e:
            logger.exception(f"Error getting move: {e}")
            result['move'] = {'error': str(e)}

class RandomAgent:
    """Simple random agent for baseline comparison."""
    def __init__(self):
        self._lock = Lock()

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict):
        """Makes a random move with thread safety."""
        logger.debug("RandomAgent: Calculating move")
        try:
            with self._lock:
                actions = game_state.get_actions()
                if not actions:
                    result['move'] = {'error': 'No available moves'}
                    return
                    
                result['move'] = random.choice(actions)
                logger.debug(f"RandomAgent selected move: {result['move']}")

        except Exception as e:
            logger.exception("Error in RandomAgent move selection")
            result['move'] = {'error': str(e)}
        
