"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging
import time
import chess.polyglot


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)

    
class MyBot(ExampleEngine):
    """Template code for hackathon participants to modify.

    This is intentionally a very small, simple, and weak example engine
    meant for learning and quick prototyping only.

    Key limitations:
    - Fixed-depth search with only a very naive time-to-depth mapping (no true time management).
    - Plain minimax: no alpha-beta pruning, so the search is much slower than it
      could be for the same depth.
    - No iterative deepening: the engine does not progressively deepen and use PV-based ordering.
    - No move ordering or capture heuristics: moves are searched in arbitrary order.
    - No transposition table or caching: repeated positions are re-searched.
    - Evaluation is material-only and very simplistic; positional factors are ignored.

    Use this as a starting point: replace minimax with alpha-beta, add
    iterative deepening, quiescence search, move ordering (MVV/LVA, history),
    transposition table, and a richer evaluator to make it competitive.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }

        self.PAWN_PST = [
              0,   0,   0,   0,   0,   0,   0,   0,
             50,  50,  50,  50,  50,  50,  50,  50,
             10,  10,  20,  30,  30,  20,  10,  10,
              5,   5,  15,  30,  30,  15,   5,   5,
              0,   0,  10,  25,  25,  10,   0,   0,
              5,  -5, -10,   0,   0, -10,  -5,   5,
              5,  10,  10, -20, -20,  10,  10,   5,
              0,   0,   0,   0,   0,   0,   0,   0
        ]
        self.KNIGHT_PST = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20,   0,   0,   0,   0, -20, -40,
            -30,   0,  10,  15,  15,  10,   0, -30,
            -30,   5,  15,  20,  20,  15,   5, -30,
            -30,   0,  15,  20,  20,  15,   0, -30,
            -30,   5,  10,  15,  15,  10,   5, -30,
            -40, -20,   0,   5,   5,   0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50,
        ]
        self.BISHOP_PST = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10,   0,   0,   0,   0,   0,   0, -10,
            -10,   0,   5,  10,  10,   5,   0, -10,
            -10,   5,   5,  10,  10,   5,   5, -10,
            -10,   0,  10,  10,  10,  10,   0, -10,
            -10,  10,  10,  10,  10,  10,  10, -10,
            -10,   5,   0,   0,   0,   0,   5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20,
        ]
        self.ROOK_PST = [
              0,   0,   0,   0,   0,   0,   0,   0,
              5,  10,  10,  10,  10,  10,  10,   5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
             -5,   0,   0,   0,   0,   0,   0,  -5,
              0,   0,   0,   5,   5,   0,   0,   0
        ]
        self.QUEEN_PST = [
            -20, -10, -10,  -5,  -5, -10, -10, -20,
            -10,   0,   0,   0,   0,   0,   0, -10,
            -10,   0,   5,   5,   5,   5,   0, -10,
             -5,   0,   5,   5,   5,   5,   0,  -5,
              0,   0,   5,   5,   5,   5,   0,  -5,
            -10,   5,   5,   5,   5,   5,   0, -10,
            -10,   0,   5,   0,   0,   0,   0, -10,
            -20, -10, -10,  -5,  -5, -10, -10, -20
        ]
        # King safety (mid-game)
        self.KING_PST = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
             20,  20,   0,   0,   0,   0,  20,  20,
             20,  30,  10,   0,   0,  10,  30,  20
        ]

        # Master lookup dictionary
        self.piece_psts = {
            chess.PAWN: self.PAWN_PST,
            chess.KNIGHT: self.KNIGHT_PST,
            chess.BISHOP: self.BISHOP_PST,
            chess.ROOK: self.ROOK_PST,
            chess.QUEEN: self.QUEEN_PST,
            chess.KING: self.KING_PST,
        }

        self.transposition_table = {}
        self.TT_FLAG_EXACT = 0
        self.TT_FLAG_LOWER = 1
        self.TT_FLAG_UPPER = 2

        self.max_ply = 100
        self.killer_moves = [[None, None] for _ in range (self.max_ply)]

        self.history_scores = [[[0 for _ in range(64)] for _ in range(64)],
                               [[0 for _ in range(64)] for _ in range(64)]]
        
        self.PAWN_SHIELD_PENALTY = 25
        self.QUEEN_PAWN_SHIELD_PENALTY = 25
        self.ATTACKER_PROXIMITY_BONUS = 3

        self.NMP_REDUCTION = 2
        self.NMP_MIN_DEPTH = 3

        self.LMR_MIN_DEPTH = 3
        self.LMR_MIN_MOVE_COUNT = 4
        self.LMR_BASE_REDUCTION = 1

        self.ROOK_SEVENTH_RANK_BONUS = 35
        self.ROOK_OPEN_FILE_BONUS = 20
        self.ROOK_SEMI_OPEN_FILE_BONUS = 10
        self.KNIGHT_OUTPOST_BONUS = 30
        self.BISHOP_PAIR_BONUS = 50
        self.DOUBLED_PAWN_PENALTY = 10
        self.MOBILITY_WEIGHT = 1
        self.BISHOP_OPEN_DIAGONAL_BONUS = 8
        self.BISHOP_ATTACKING_DIAGONAL_BONUS = 12

        self.KING_ENDGAME_PST = [
            -50, -30, -10,   0,   0, -10, -30, -50,
            -30, -10,  20,  30,  30,  20, -10, -30,
            -10,  20,  40,  50,  50,  40,  20, -10,
              0,  30,  50,  55,  55,  50,  30,   0,
              0,  30,  50,  55,  55,  50,  30,   0,
            -10,  20,  40,  50,  50,  40,  20, -10,
            -30, -10,  20,  30,  30,  20, -10, -30,
            -50, -30, -10,   0,   0, -10, -30, -50,
        ]

        self.CASTLING_BONUS = 30
        self.UNDEVELOPED_MINOR_PENALTY = 8

        self.ENDGAME_MATERIAL_THRESHOLD = (self.piece_values[chess.ROOK] * 2 +
                                           self.piece_values[chess.KNIGHT] +
                                           self.piece_values[chess.BISHOP])

    def _get_game_phase(self, b: chess.Board) -> int:

        material_count = 0
        for color in [chess.WHITE, chess.BLACK]:
            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                material_count += len(b.pieces(pt, color)) * self.piece_values[pt]

        if material_count < self.ENDGAME_MATERIAL_THRESHOLD:
            return 1
        else:
            return 0

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        # NOTE: The sections below are intentionally simple to keep the example short.
        # They demonstrate the structure of a search but also highlight the engine's
        # weaknesses (fixed depth, naive time handling, no pruning, no quiescence, etc.).

        # --- very simple time-based depth selection (naive) ---
        # Expect args to be (time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE)
        time_limit = args[0] if (args and isinstance(args[0], Limit)) else None
        my_time = my_inc = None
        if time_limit is not None:
            if isinstance(time_limit.time, (int, float)):
                my_time = time_limit.time
                my_inc = 0
            elif board.turn == chess.WHITE:
                my_time = time_limit.white_clock if isinstance(time_limit.white_clock, (int, float)) else 0
                my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, (int, float)) else 0
            else:
                my_time = time_limit.black_clock if isinstance(time_limit.black_clock, (int, float)) else 0
                my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, (int, float)) else 0

        start_time = time.monotonic()
        move_budget = 3 # in seconds

        self.transposition_table = {}

        self.killer_moves = [[None, None] for _ in range(self.max_ply)]
        self.history_scores = [[[0 for _ in range(64)] for _ in range(64)],
                               [[0 for _ in range(64)] for _ in range(64)]]

        if my_time is not None:
            ideal_time = (my_time / 35) + (my_inc * 0.75)
            max_time = my_time / 8

            move_budget = min(ideal_time, max_time)

            if my_time < 10:
                move_budget = min(my_time / 20, 0.4)

            move_budget = min(move_budget, 5)
            move_budget = max(move_budget, 0.1)

        maximizing = board.turn == chess.WHITE
        best_move = None
        pv = []

        for i in range(1, 100):
            time_spent = time.monotonic() - start_time
            if time_spent > move_budget:
                break

            # --- root move selection ---
            legal = self.root_order_moves(board, pv) # TODO: Sort with most likely best moves
            if not legal:
                # Should not happen during normal play; fall back defensively
                return PlayResult(random.choice(list(board.legal_moves)), None)

            best_eval = -10**12 if maximizing else 10**12
            current_best_move = None
            current_best_pv = []

            # Lookahead depth chosen by the simple time heuristic; subtract one for the root move
            for m in legal:
                board.push(m)
                val, returned_pv = self.minimax(board, i - 1, 0, -10**12, 10**12, not maximizing)
                board.pop()

                if maximizing and val > best_eval:
                    best_eval, current_best_move = val, m
                    current_best_pv = [m]+ returned_pv
                elif not maximizing and val < best_eval:
                    best_eval, current_best_move = val, m
                    current_best_pv = [m] + returned_pv

            best_move = current_best_move
            pv = current_best_pv

            # force check?
            if abs(best_eval) > 9_000_000:
                break


            # if (time_spent > budget / 2): break ?? improve this

            # Fallback in rare cases (shouldn't trigger)
            if best_move is None:
                best_move = legal[0]

        return PlayResult(best_move, None)
    
    def get_ordered_captures(self, board: chess.Board):
        moves = []
        for m in board.legal_moves:
            if board.is_capture(m):
                victim_type = board.piece_type_at(m.to_square)
                attacker_type = board.piece_type_at(m.from_square)

                if board.is_en_passant(m):
                    victim_type = chess.PAWN

                if victim_type is None or attacker_type is None:
                    continue

                score = 1_000_000 + (self.piece_values[victim_type] * 10) - self.piece_values[attacker_type]
                moves.append((m, score))
        
        moves.sort(key=lambda item: item[1], reverse=True)
        return [m for m, _ in moves]
    
    def get_ordered_moves(self, board: chess.Board, ply: int = 0, tt_move: chess.Move | None = None) -> list[chess.Move]:
        killers = self.killer_moves[ply] if ply < self.max_ply else []
        player_history = self.history_scores[board.turn]

        moves = []
        for m in board.legal_moves:
            score = 0

            if m == tt_move:
                score = 2_000_000

            elif board.is_capture(m):
                victim_type = board.piece_type_at(m.to_square)
                attacker_type = board.piece_type_at(m.from_square)

                if board.is_en_passant(m):
                    victim_type = chess.PAWN

                if victim_type is None or attacker_type is None:
                    continue

                score = 1_000_000 + (self.piece_values[victim_type] * 10) - self.piece_values[attacker_type]
                # TODO

            elif m in killers:
                score = 800_000

            else:
                score = player_history[m.from_square][m.to_square]

            moves.append((m, score))
        
        moves.sort(key=lambda move_score: move_score[1], reverse=True)
        return [m for m, _ in moves]

    def root_order_moves(self, board: chess.Board, pv: list[chess.Move]) -> list[chess.Move]:
        pv_move = pv[0] if pv else None
        
        def get_sort_key(m: chess.Move):
            if m == pv_move:
                return 3_000_000

            score = 0
            if board.is_capture(m):
                victim_type = board.piece_type_at(m.to_square)
                attacker_type = board.piece_type_at(m.from_square)

                if board.is_en_passant(m): victim_type = chess.PAWN
                if victim_type is None or attacker_type is None: return 0

                # MVV/LVA
                score = 1_000_000 + (self.piece_values[victim_type] * 10) - self.piece_values[attacker_type]
    
            return score

        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=get_sort_key, reverse=True)
        return legal_moves

     # --- plain minimax (no alpha-beta) ---
    def minimax(self, b: chess.Board, depth: int, ply: int, alpha: int, beta: int, maximizing: bool) -> tuple[int, list[chess.Move]]:
        o_alpha = alpha
        hash_key = chess.polyglot.zobrist_hash(b)

        tt_entry = self.transposition_table.get(hash_key)
        tt_move = None

        if tt_entry:
            tt_depth, tt_score, tt_flag, tt_best_move = tt_entry
            tt_move = tt_best_move

            if tt_depth >= depth:
                if tt_flag == self.TT_FLAG_EXACT:
                    return (tt_score, [tt_best_move] if tt_best_move else [])
                elif tt_flag == self.TT_FLAG_LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == self.TT_FLAG_UPPER:
                    beta = min(beta, tt_score)
                
                if beta <= alpha:
                    return (tt_score, [tt_best_move] if tt_best_move else [])

        if b.is_game_over():
            return (self.evaluate(b), [])
        if depth == 0:
            q_score = self.q_search(b, alpha, beta, maximizing)
            return (q_score, [])

        # Base case done

        is_in_check = b.is_check()
        has_major_pieces = (b.occupied_co[b.turn] & ~b.pawns & ~b.kings) != 0

        if (depth >= self.NMP_MIN_DEPTH and not is_in_check and has_major_pieces and ply > 0):
            b.push(chess.Move.null())

            new_depth = depth - 1 - self.NMP_REDUCTION

            val, _ = self.minimax(b, new_depth, ply + 1, -beta, -alpha, not maximizing)
            val = -val

            b.pop()

            if val >= beta:
                if not tt_entry or new_depth >= tt_entry[0]:
                     self.transposition_table[hash_key] = (new_depth, beta, self.TT_FLAG_LOWER, None)
                return (beta, [])

        moves = self.get_ordered_moves(b, ply, tt_move)
        killers = self.killer_moves[ply] if ply < self.max_ply else []

        best_move = None
        best_pv = []
        move_count = 0

        if maximizing:
            best = -10**12

            for m in moves:
                move_count += 1
                reduction = 0

                apply_lmr = (depth >= self.LMR_MIN_DEPTH and
                             move_count >= self.LMR_MIN_MOVE_COUNT and
                             not is_in_check and
                             not b.is_capture(m) and
                             m != tt_move and
                             m not in killers)

                b.push(m)

                if apply_lmr:
                    reduction = self.LMR_BASE_REDUCTION
                    if move_count > 6: reduction += 1
                    reduction = min(reduction, depth-1)

                    val, current_pv = self.minimax(b, depth - 1, ply + 1, alpha, beta, False)

                    if val > alpha:
                        val, current_pv = self.minimax(b, depth - 1, ply + 1, alpha, beta, False)
                else:
                        val, current_pv = self.minimax(b, depth - 1, ply + 1, alpha, beta, False)

                b.pop()


                if val > best:
                    best = val
                    best_pv = [m] + current_pv
                    best_move = m
                alpha = max(alpha, val)

                if beta <= alpha:
                    if not b.is_capture(m):
                        self.store_killer_move(m, ply)
                        self.history_scores[b.turn][m.from_square][m.to_square] += depth * depth
                    break
        else:
            best = 10**12
            move_count = 0
            for m in moves:
                move_count += 1
                reduction = 0

                apply_lmr = (depth >= self.LMR_MIN_DEPTH and
                             move_count >= self.LMR_MIN_MOVE_COUNT and
                             not is_in_check and
                             not b.is_capture(m) and
                             m != tt_move and
                             m not in killers)

                b.push(m)

                if apply_lmr:
                    reduction = self.LMR_BASE_REDUCTION
                    if move_count > 6: reduction += 1
                    reduction = min(reduction, depth - 1)

                    val, current_pv = self.minimax(b, depth - 1 - reduction, ply + 1, alpha, beta, True)
                    if val < beta:
                        val, current_pv = self.minimax(b, depth - 1, ply + 1, alpha, beta, True) 
                else:
                    val, current_pv = self.minimax(b, depth - 1, ply + 1, alpha, beta, True)

                b.pop()
                if val < best:
                    best = val
                    best_pv = [m] + current_pv
                    best_move = m
                beta = min(beta, val)

                if beta <= alpha:
                    if (not b.is_capture(m)):
                        self.store_killer_move(m, ply)
                        self.history_scores[b.turn][m.from_square][m.to_square] += depth * depth
                    break

        flag_to_store = self.TT_FLAG_EXACT
        if best <= o_alpha:
            flag_to_store = self.TT_FLAG_UPPER
        elif best >= beta:
            flag_to_store = self.TT_FLAG_LOWER

        if not tt_entry or depth >= tt_entry[0]:
            self.transposition_table[hash_key] = (depth, best, flag_to_store, best_move)

        return (best, best_pv)

    def store_killer_move(self, move: chess.Move, ply: int):
        if ply >= self.max_ply:
            return
        
        killers = self.killer_moves[ply]
        if move != killers[0]:
            killers[1] = killers[0]
            killers[0] = move

    def q_search(self, b: chess.Board, alpha: int, beta: int, maximizing: bool) -> int:
            standard_score = self.evaluate(b)

            if maximizing:
                if standard_score >= beta:
                    return beta
                alpha = max(alpha, standard_score)
            else:
                if standard_score <= alpha:
                    return alpha
                beta = min(beta, standard_score)

            capture_moves = self.get_ordered_captures(b)

            for m in capture_moves:

                b.push(m)
                score = self.q_search(b, alpha, beta, not maximizing)
                b.pop()

                if maximizing:
                    if score >= beta:
                        return beta
                    alpha = max(alpha, score)
                else:
                    if score <= alpha:
                        return alpha
                    beta = min(beta, score)

            return alpha if maximizing else beta
        
    def evaluate(self, b: chess.Board) -> int:
        if b.is_game_over():
            outcome = b.outcome()
            if outcome is None or outcome.winner is None: return 0
            return 10_000_000 if outcome.winner is chess.WHITE else -10_000_000

        game_phase = self._get_game_phase(b)
        is_endgame = (game_phase == 1)

        score = 0
        white_pawns = b.pieces(chess.PAWN, chess.WHITE)
        black_pawns = b.pieces(chess.PAWN, chess.BLACK)
        all_pawns = white_pawns | black_pawns

        for pt in self.piece_values:
            material_value = self.piece_values[pt]
            white_pieces = b.pieces(pt, chess.WHITE)
            black_pieces = b.pieces(pt, chess.BLACK)

            if pt != chess.KING:
                score += material_value * (len(white_pieces) - len(black_pieces))

            if pt == chess.KING:
                pst = self.KING_ENDGAME_PST if is_endgame else self.KING_PST
            else:
                pst = self.piece_psts.get(pt)

            if pst:
                for sq in white_pieces: score += pst[sq]
                for sq in black_pieces: score -= pst[chess.square_mirror(sq)]

        for file_index in range(8):
            white_pawns_on_file = 0
            for sq in white_pawns:
                if chess.square_file(sq) == file_index:
                    white_pawns_on_file += 1

            black_pawns_on_file = 0
            for sq in black_pawns:
                if chess.square_file(sq) == file_index:
                    black_pawns_on_file += 1

            if white_pawns_on_file > 1: score -= (white_pawns_on_file - 1) * self.DOUBLED_PAWN_PENALTY
            if black_pawns_on_file > 1: score += (black_pawns_on_file - 1) * self.DOUBLED_PAWN_PENALTY

        mobility_score = 0
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in b.pieces(pt, chess.WHITE):
                mobility_score += bin(b.attacks_mask(sq)).count('1')
            for sq in b.pieces(pt, chess.BLACK):
                mobility_score -= bin(b.attacks_mask(sq)).count('1')
        score += mobility_score * self.MOBILITY_WEIGHT

        if len(b.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += self.BISHOP_PAIR_BONUS
        if len(b.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= self.BISHOP_PAIR_BONUS

        white_bishops = b.pieces(chess.BISHOP, chess.WHITE)
        black_bishops = b.pieces(chess.BISHOP, chess.BLACK)
        b_king_sq = b.king(chess.BLACK)
        w_king_sq = b.king(chess.WHITE)

        black_king_zone = {sq for sq in chess.SQUARES if chess.square_distance(sq, b_king_sq) <= 2}
        white_king_zone = {sq for sq in chess.SQUARES if chess.square_distance(sq, w_king_sq) <= 2}

        for sq in white_bishops:
            attacked_squares = b.attacks(sq)
            open_diag_bonus_applied = False
            attacking_diag_bonus_applied = False
            for target_sq in attacked_squares:
                 path_mask = chess.BB_RAYS[sq][target_sq]
                 is_open = True
                 for path_sq in chess.scan_forward(path_mask):
                     if path_sq in all_pawns:
                         is_open = False
                         break

                 if is_open and not open_diag_bonus_applied:
                      score += self.BISHOP_OPEN_DIAGONAL_BONUS
                      open_diag_bonus_applied = True
                 if is_open and not attacking_diag_bonus_applied:
                     if any(atk_sq in black_king_zone for atk_sq in attacked_squares):
                          score += self.BISHOP_ATTACKING_DIAGONAL_BONUS
                          attacking_diag_bonus_applied = True
                 if open_diag_bonus_applied and attacking_diag_bonus_applied: break

        for sq in black_bishops:
            attacked_squares = b.attacks(sq)
            open_diag_bonus_applied = False
            attacking_diag_bonus_applied = False
            for target_sq in attacked_squares:
                 path_mask = chess.BB_RAYS[sq][target_sq]
                 is_open = True
                 for path_sq in chess.scan_forward(path_mask):
                     if path_sq in all_pawns:
                         is_open = False
                         break

                 if is_open and not open_diag_bonus_applied:
                      score -= self.BISHOP_OPEN_DIAGONAL_BONUS
                      open_diag_bonus_applied = True
                 if is_open and not attacking_diag_bonus_applied:
                     if any(atk_sq in white_king_zone for atk_sq in attacked_squares):
                          score -= self.BISHOP_ATTACKING_DIAGONAL_BONUS
                          attacking_diag_bonus_applied = True
                 if open_diag_bonus_applied and attacking_diag_bonus_applied: break

        white_rooks = b.pieces(chess.ROOK, chess.WHITE)
        black_rooks = b.pieces(chess.ROOK, chess.BLACK)
        for file_index in range(8):
            white_pawns_on_file_bool = False
            for sq in white_pawns:
                 if chess.square_file(sq) == file_index:
                      white_pawns_on_file_bool = True
                      break
            black_pawns_on_file_bool = False
            for sq in black_pawns:
                 if chess.square_file(sq) == file_index:
                      black_pawns_on_file_bool = True
                      break

            is_open = not white_pawns_on_file_bool and not black_pawns_on_file_bool
            is_semi_open_white = not white_pawns_on_file_bool

            for sq in white_rooks:
                if chess.square_file(sq) == file_index:
                    if is_open: score += self.ROOK_OPEN_FILE_BONUS
                    elif is_semi_open_white: score += self.ROOK_SEMI_OPEN_FILE_BONUS
                    if chess.square_rank(sq) == 6: score += self.ROOK_SEVENTH_RANK_BONUS

            is_semi_open_black = not black_pawns_on_file_bool
            for sq in black_rooks:
                 if chess.square_file(sq) == file_index:
                    if is_open: score -= self.ROOK_OPEN_FILE_BONUS
                    elif is_semi_open_black: score -= self.ROOK_SEMI_OPEN_FILE_BONUS
                    if chess.square_rank(sq) == 1: score -= self.ROOK_SEVENTH_RANK_BONUS

        WHITE_OUTPOST_RANKS = {3, 4, 5}
        BLACK_OUTPOST_RANKS = {4, 3, 2}
        for sq in b.pieces(chess.KNIGHT, chess.WHITE):
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            if rank in WHITE_OUTPOST_RANKS:
                supported = False
                if file > 0 and b.piece_at(sq - 9) == chess.Piece(chess.PAWN, chess.WHITE): supported = True
                if file < 7 and b.piece_at(sq - 7) == chess.Piece(chess.PAWN, chess.WHITE): supported = True

                attacked_by_pawn = False
                if file > 0 and b.piece_at(sq + 7) == chess.Piece(chess.PAWN, chess.BLACK): attacked_by_pawn = True
                if file < 7 and b.piece_at(sq + 9) == chess.Piece(chess.PAWN, chess.BLACK): attacked_by_pawn = True

                if supported and not attacked_by_pawn:
                    score += self.KNIGHT_OUTPOST_BONUS

        for sq in b.pieces(chess.KNIGHT, chess.BLACK):
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            if rank in BLACK_OUTPOST_RANKS:
                supported = False
                if file > 0 and b.piece_at(sq + 9) == chess.Piece(chess.PAWN, chess.BLACK): supported = True
                if file < 7 and b.piece_at(sq + 7) == chess.Piece(chess.PAWN, chess.BLACK): supported = True

                attacked_by_pawn = False
                if file > 0 and b.piece_at(sq - 7) == chess.Piece(chess.PAWN, chess.WHITE): attacked_by_pawn = True
                if file < 7 and b.piece_at(sq - 9) == chess.Piece(chess.PAWN, chess.WHITE): attacked_by_pawn = True

                if supported and not attacked_by_pawn:
                    score -= self.KNIGHT_OUTPOST_BONUS

        if not is_endgame:
            # White Knights on starting squares
            if b.piece_at(chess.B1) == chess.Piece(chess.KNIGHT, chess.WHITE): score -= self.UNDEVELOPED_MINOR_PENALTY
            if b.piece_at(chess.G1) == chess.Piece(chess.KNIGHT, chess.WHITE): score -= self.UNDEVELOPED_MINOR_PENALTY
            # White Bishops on starting squares
            if b.piece_at(chess.C1) == chess.Piece(chess.BISHOP, chess.WHITE): score -= self.UNDEVELOPED_MINOR_PENALTY
            if b.piece_at(chess.F1) == chess.Piece(chess.BISHOP, chess.WHITE): score -= self.UNDEVELOPED_MINOR_PENALTY

            # Black Knights on starting squares
            if b.piece_at(chess.B8) == chess.Piece(chess.KNIGHT, chess.BLACK): score += self.UNDEVELOPED_MINOR_PENALTY
            if b.piece_at(chess.G8) == chess.Piece(chess.KNIGHT, chess.BLACK): score += self.UNDEVELOPED_MINOR_PENALTY
            # Black Bishops on starting squares
            if b.piece_at(chess.C8) == chess.Piece(chess.BISHOP, chess.BLACK): score += self.UNDEVELOPED_MINOR_PENALTY
            if b.piece_at(chess.F8) == chess.Piece(chess.BISHOP, chess.BLACK): score += self.UNDEVELOPED_MINOR_PENALTY

            w_king_sq = b.king(chess.WHITE)
            b_king_sq = b.king(chess.BLACK)
            w_king_file = chess.square_file(w_king_sq)
            b_king_file = chess.square_file(b_king_sq)

            WHITE_KINGSIDE_SHIELD = {chess.F2, chess.G2, chess.H2, chess.F3, chess.G3, chess.H3}
            WHITE_QUEENSIDE_SHIELD = {chess.A2, chess.B2, chess.C2, chess.A3, chess.B3, chess.C3}
            BLACK_KINGSIDE_SHIELD = {chess.F7, chess.G7, chess.H7, chess.F6, chess.G6, chess.H6}
            BLACK_QUEENSIDE_SHIELD = {chess.A7, chess.B7, chess.C7, chess.A6, chess.B6, chess.C6}

            if w_king_file >= 5:
                shield_missing = WHITE_KINGSIDE_SHIELD.difference(white_pawns)
                score -= len(shield_missing) * self.PAWN_SHIELD_PENALTY
            elif w_king_file <= 2:
                shield_missing = WHITE_QUEENSIDE_SHIELD.difference(white_pawns)
                score -= len(shield_missing) * self.QUEEN_PAWN_SHIELD_PENALTY

            if b_king_file >= 5:
                shield_missing = BLACK_KINGSIDE_SHIELD.difference(black_pawns)
                score += len(shield_missing) * self.PAWN_SHIELD_PENALTY
            elif b_king_file <= 2:
                shield_missing = BLACK_QUEENSIDE_SHIELD.difference(black_pawns)
                score += len(shield_missing) * self.QUEEN_PAWN_SHIELD_PENALTY

            def get_coords(sq): return (chess.square_file(sq), chess.square_rank(sq))
            (w_king_file_coord, w_king_rank) = get_coords(w_king_sq)
            (b_king_file_coord, b_king_rank) = get_coords(b_king_sq)

            for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in b.pieces(pt, chess.WHITE):
                    (file, rank) = get_coords(sq)
                    dist = abs(file - b_king_file_coord) + abs(rank - b_king_rank)
                    score += (14 - dist) * self.ATTACKER_PROXIMITY_BONUS
                for sq in b.pieces(pt, chess.BLACK):
                    (file, rank) = get_coords(sq)
                    dist = abs(file - w_king_file_coord) + abs(rank - w_king_rank)
                    score -= (14 - dist) * self.ATTACKER_PROXIMITY_BONUS

        return score
    