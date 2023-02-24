# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        oldFood = currentGameState.getFood()
        food_list = newFood.asList()
        distance_to_food_list = []
        distance_to_ghost_list = []
        for i in newGhostStates:
            distance_to_ghost_list.append(util.manhattanDistance(i.getPosition(), newPos))
        for i in food_list:
            distance_to_food_list.append((util.manhattanDistance(i, newPos)))
        closest_ghost_distance = min(distance_to_ghost_list)
        food_left = len(distance_to_food_list)
        
        if sum(distance_to_food_list) != 0:
            closest_food_distance = min(distance_to_food_list)
            if oldFood[newPos[0]][newPos[1]]:
                eating = 100
            else:
                eating = 0

            if closest_ghost_distance < 3:
                    valuation = closest_ghost_distance
            else:
                valuation = -closest_food_distance + eating
        else:
            if oldFood[newPos[0]][newPos[1]]:
                eating = 100
            else:
                eating = 0

            if closest_ghost_distance < 3:
                    valuation = closest_ghost_distance
            else:
                valuation = eating
        return valuation
    

        """
        if sum(newScaredTimes) != 0 and sum(distance_to_food_list) == 0:
            return 50*-len(distance_to_food_list) - min(distance_to_ghost_list)
        elif sum(newScaredTimes) != 0 and sum(distance_to_food_list) != 0:
            return 50*-len(distance_to_food_list) - min(distance_to_food_list) - min(distance_to_ghost_list)
        elif sum(distance_to_food_list) == 0:
            #print(100*-len(distance_to_food_list) + max(distance_to_ghost_list))
            return 100*-len(distance_to_food_list) + max(distance_to_ghost_list)
        elif sum(distance_to_food_list) != 0:
            #print(100*-len(distance_to_food_list) - min(distance_to_food_list) + max(distance_to_ghost_list))
            return 100*-len(distance_to_food_list) - min(distance_to_food_list) + max(distance_to_ghost_list)
        #return max(distance_to_ghost_list) - len(distance_to_food_list)
        # return successorGameState.getScore()
        """
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, score = self.MiniMax(gameState,0,0)
        return action
        
    def MiniMax(self, gameState, depth, agentIndex):

        if agentIndex == gameState.getNumAgents():
            depth += 1
            agentIndex = 0

        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return None, self.evaluationFunction(gameState)

        if agentIndex == 0:
            bestScore = float("-inf")
            possibleActions = [action for action in gameState.getLegalActions(agentIndex) if action != Directions.STOP]
            for action in possibleActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                placeholder , currScore = self.MiniMax(nextState, depth, agentIndex+1)
                if currScore > bestScore:
                    bestScore = currScore
                    bestAction = action
            
        if agentIndex != 0:
            bestScore = float("inf")
            possibleActions = [action for action in gameState.getLegalActions(agentIndex) if action != Directions.STOP]
            for action in possibleActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                placeholder, currScore = self.MiniMax(nextState, depth, agentIndex+1)
                if currScore < bestScore:
                    bestScore = currScore
                    bestAction = action
            
        return bestAction, bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = self.AlphaBeta(gameState,0,0, float("-inf"), float("inf"))
        return action
        
    def AlphaBeta(self, gameState, depth, agentIndex, alpha, beta):

        if agentIndex == gameState.getNumAgents():
            depth += 1
            agentIndex = 0

        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return None, self.evaluationFunction(gameState)

        if agentIndex == 0:
            bestScore = float("-inf")
            possibleActions = [action for action in gameState.getLegalActions(agentIndex) if action != Directions.STOP]
            for action in possibleActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                placeholder , currScore = self.AlphaBeta(nextState, depth, agentIndex+1, alpha, beta)
                if currScore > bestScore:
                    bestScore = currScore
                    bestAction = action
                alpha = max(alpha, currScore)
                if beta < alpha:
                    break
                 
        if agentIndex != 0:
            bestScore = float("inf")
            possibleActions = [action for action in gameState.getLegalActions(agentIndex) if action != Directions.STOP]
            for action in possibleActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                placeholder, currScore = self.AlphaBeta(nextState, depth, agentIndex+1, alpha, beta)
                if currScore < bestScore:
                    bestScore = currScore
                    bestAction = action
                beta = min(beta, currScore)
                if beta < alpha:
                    break

        return bestAction, bestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.ExpectiMax(gameState,0,0)
        return action
        
    def ExpectiMax(self, gameState, depth, agentIndex):

        if agentIndex == gameState.getNumAgents():
            depth += 1
            agentIndex = 0

        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return None, self.evaluationFunction(gameState)
        
        bestAction, bestScore = None, None
        if agentIndex == 0:
            bestScore = float("-inf")
            possibleActions = [action for action in gameState.getLegalActions(agentIndex)]
            for action in possibleActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                placeholder , currScore = self.ExpectiMax(nextState, depth, agentIndex+1)
                if currScore > bestScore:
                    bestScore = currScore
                    bestAction = action
            
        if agentIndex != 0:
            possibleActions = [action for action in gameState.getLegalActions(agentIndex)]    
            actionProbability = 1 / len(possibleActions)
            for action in possibleActions:
                if bestScore == None:
                    bestScore = 0.0
                nextState = gameState.generateSuccessor(agentIndex, action)
                placeholder, currScore = self.ExpectiMax(nextState, depth, agentIndex+1)
                expected = currScore * actionProbability
                bestScore += expected
                bestAction = action
        
        return bestAction, bestScore

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
   

# Abbreviation
better = betterEvaluationFunction
