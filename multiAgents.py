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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        "*** YOUR CODE HERE ***"
        value = successorGameState.getScore()
        #print(newGhostStates[0])
        distanceToGhost = manhattanDistance(newPos, newGhostStates[0].getPosition())
        value -= 15 / (distanceToGhost + 1)
        distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(distancesToFood):
          #print(len(distancesToFood))
          value += 10 / min(distancesToFood)
        return value  
def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        global bestAct

        def minimax(agent, depth, state):
            #print(state.getPacmanPosition())
            #print(depth)
            global bestAct
            #print(self.depth, depth)
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agent == 0:
                score = -1000
                actions = state.getLegalActions(agent)
                #print(actions)
                for action in actions:
                    newState = state.generateSuccessor(agent, action)
                    nextAgent = (agent + 1) % state.getNumAgents()
                    v = minimax(nextAgent, depth, newState)
                    if v > score and depth == self.depth:
                        bestAct = action
                    score = max(score, v)
                return score
            else:
                score = 1000
                actions = state.getLegalActions(agent)
                #print(actions)
                for action in actions:
                    newState = state.generateSuccessor(agent, action)
                    nextAgent = (agent + 1) % state.getNumAgents()
                    nextDepth = depth - 1 if nextAgent == 0 else depth
                    v = minimax(nextAgent, nextDepth, newState)
                    score = min(score, v)
                return score

        minimax(0, self.depth, gameState)
        # print(bestAct)
        return bestAct
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        global bestAct
        def alphaBeta(agent, depth, state, alpha, beta):
          #print(depth)
          global bestAct
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if agent == 0:
            score = -1000
            actions = state.getLegalActions(agent)
            for action in actions:
              newState = state.generateSuccessor(agent, action)
              nextAgent = (agent + 1) % state.getNumAgents()
              v = alphaBeta(nextAgent, depth, newState, alpha, beta)
              if v > score and depth == self.depth:
                bestAct = action
              score = max(v, score)
              if score > beta:
                return score
              alpha = max(score, alpha)
            return score            
          else :
            score = 1000
            actions = state.getLegalActions(agent)
            for action in actions:
              newState = state.generateSuccessor(agent, action)
              nextAgent = (agent + 1) % state.getNumAgents()
              nextDepth = depth - 1 if nextAgent == 0 else depth
              v = alphaBeta(nextAgent, nextDepth, newState, alpha, beta)
              score = min(score, v)
              if score < alpha:
                  return score
              beta = min(beta, score)
            return score
        alphaBeta(0, self.depth, gameState, -1e9, 1e9)
        return bestAct

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        global bestAct
        def expectimax(agent, depth, state):
          global bestAct
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if agent == 0:
            score = -1000
            actions = state.getLegalActions(agent)
            for action in actions:
              newState = state.generateSuccessor(agent, action)
              nextAgent = (agent + 1) % state.getNumAgents()
              v = expectimax(nextAgent, depth, newState)
              if v > score and depth == self.depth:
                bestAct = action
              score = max(v, score)
              #print(score)
            return score
          else:
            score = 0
            actions = state.getLegalActions(agent)
            for action in actions:
              newState = state.generateSuccessor(agent, action)
              nextAgent = (agent + 1) % state.getNumAgents()
              nextDepth = depth-1 if nextAgent == 0 else depth
              v = expectimax(nextAgent, nextDepth, newState)
              score += v
            score /= len(actions)
            #print(score)
            return score

        expectimax(0, self.depth, gameState)
        return bestAct

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    disToNearGhost, ghostFeature = float("inf"), 0
    for ghost in currentGameState.getGhostStates():
        ghostDist = manhattanDistance(pacPos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            disToNearGhost = min(ghostDist, disToNearGhost)
        elif ghost.scaredTimer > ghostDist:
            ghostFeature += 200 - ghostDist
    if disToNearGhost == float("inf"):
        disToNearGhost = 0
    ghostFeature += disToNearGhost

    disToNearFood = 0
    foodPosList = sorted(currentGameState.getFood().asList())
    if foodPosList:
        disToNearFood = manhattanDistance(pacPos, foodPosList[0])
    numFood = currentGameState.getNumFood()
    score = currentGameState.getScore()
    numCaps = len(currentGameState.getCapsules())  # thuc an to

    return score + ghostFeature + 2 * numCaps - 2 * disToNearFood - 10 * numFood
    
# Abbreviation
better = betterEvaluationFunction

