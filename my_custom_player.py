
from sample_players import DataPlayer
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state, technique='pvs'):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        elif technique == 'minmax':
            self.queue.put(self.minimax(state, depth=3))
        elif technique == 'alphabeta':
            self.queue.put(self.alpha_beta(state, depth=5))
        elif technique == 'alphabeta_iterative':
            dlimit = 5
            best_move = None            
            for depth in range(1, dlimit+1):
                best_move = self.alpha_beta(state, depth)                
                self.queue.put(best_move)
        elif technique == 'pvs':
            dlimit = 5
            best_move = None            
            for depth in range(1, dlimit+1):
                best_move = self.pvs(state, depth)                
                self.queue.put(best_move)
        else:
            self.queue.put(random.choice(state.actions()))
     #Basic #my_moves - #opponent_moves heuristic from lecture
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
    def minimax(self, state, depth):
        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))      
  
    def min_value_ab(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("inf")
        for action in state.actions():
            value = min(value, self.max_value_ab(state.result(action), alpha, beta, depth - 1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value_ab(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.min_value_ab(state.result(action), alpha, beta, depth - 1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value
        
    def alpha_beta(self, state, depth):        
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for move in state.actions():
            value = self.min_value_ab(state.result(move), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            if value > best_score:
                best_score = value
                best_move = move
        return best_move    
    
    def pvs(self, state, depth):        
        alpha = float("-inf")
        beta = float("inf")
        #Assume the first move is the best move available
        # Calculate value of the first move and use it to compare it with the remaining moves
        best_move = state.actions()[0]
        best_value = self.min_value_ab(state.result(best_move), alpha, beta, depth -1)
        #Search all reminaing moves
        for move in state.actions()[1:]:
            #Perform search with a null window
            value = self.min_value_ab(state.result(move), -alpha, -alpha-1, depth - 1)
            if value > alpha and value < beta:
                #null window search failed, try with a normal search
                value = self.min_value_ab(state.result(move), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            if value > best_value:
                best_score = value
                best_move = move
        return best_move    
