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
        currentPos = currentGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()
        
        # Start with successor game state score (required)
        score = successorGameState.getScore()
        
        # Food evaluation - add bonus for getting closer to food
        foodList = newFood.asList()
        if foodList:
            # Distance to closest food
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDistance = min(foodDistances)
            
            # Add reciprocal of distance as bonus
            score += 10.0 / closestFoodDistance
            
            # Bonus if we ate food (food count decreased)
            if len(currentFood.asList()) > len(foodList):
                score += 100
        
        # Ghost evaluation
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            
            if newScaredTimes[i] > 0:
                # Ghost is scared - chase it
                if distance == 0:
                    score += 200  # Eat the ghost
                else:
                    score += 10.0 / distance  # Get closer
            else:
                # Ghost is dangerous - avoid it
                if distance <= 1:
                    score -= 500  # Very dangerous
                elif distance == 2:
                    score -= 50   # Still risky
        
        # Penalize stopping
        if action == Directions.STOP:
            score -= 10
        
        return score

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
        def minimax(state, depth, agentIndex):
        # Terminal conditions: game over or reached maximum depth
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:  # Pacman's turn (maximizing player)
                bestValue = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    # Next agent is the first ghost (index 1)
                    value = minimax(successor, depth, 1)
                    bestValue = max(bestValue, value)
                return bestValue
                
            else:  # Ghost's turn (minimizing player)
                bestValue = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                
                # If this is the last ghost, next turn goes back to Pacman
                # and we complete one ply, so decrement depth
                if nextAgent == numAgents:
                    nextAgent = 0  # Back to Pacman
                    nextDepth = depth - 1  # Decrement depth after complete ply
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(successor, nextDepth, nextAgent)
                    bestValue = min(bestValue, value)
                return bestValue
        
        # Find the best action for Pacman at the root level
        bestAction = None
        bestValue = float('-inf')
        
        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            # Start minimax with first ghost (index 1) at current depth
            value = minimax(successor, self.depth, 1)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Terminal conditions: game over or reached maximum depth
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:  # Pacman's turn (maximizing player)
                v = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    # Next agent is the first ghost (index 1)
                    v = max(v, alphaBeta(successor, depth, 1, alpha, beta))
                    # Pruning condition: if v > beta, return v (prune)
                    if v > beta:
                        return v
                    # Update alpha
                    alpha = max(alpha, v)
                return v
                
            else:  # Ghost's turn (minimizing player)
                v = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                
                # If this is the last ghost, next turn goes back to Pacman
                # and we complete one ply, so decrement depth
                if nextAgent == numAgents:
                    nextAgent = 0  # Back to Pacman
                    nextDepth = depth - 1  # Decrement depth after complete ply
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                    # Pruning condition: if v < alpha, return v (prune)
                    if v < alpha:
                        return v
                    # Update beta
                    beta = min(beta, v)
                return v
        
        # Find the best action for Pacman at the root level
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            # Start alpha-beta with first ghost (index 1) at current depth
            value = alphaBeta(successor, self.depth, 1, alpha, beta)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
            
            # Update alpha at root level
            alpha = max(alpha, value)
        
        return bestAction

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
        def expectimax(state, depth, agentIndex):
            # Terminal conditions: game over or reached maximum depth
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:  # Pacman's turn (maximizing player)
                bestValue = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    # Next agent is the first ghost (index 1)
                    value = expectimax(successor, depth, 1)
                    bestValue = max(bestValue, value)
                return bestValue
                
            else:  # Ghost's turn (chance node - random behavior)
                expectedValue = 0
                nextAgent = agentIndex + 1
                nextDepth = depth
                
                # If this is the last ghost, next turn goes back to Pacman
                # and we complete one ply, so decrement depth
                if nextAgent == numAgents:
                    nextAgent = 0  # Back to Pacman
                    nextDepth = depth - 1  # Decrement depth after complete ply
                
                # Calculate expected value (average of all possible actions)
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimax(successor, nextDepth, nextAgent)
                    expectedValue += value
                
                # Return expected value (uniform probability distribution)
                return expectedValue / len(legalActions)
        
        # Find the best action for Pacman at the root level
        bestAction = None
        bestValue = float('-inf')
        
        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            # Start expectimax with first ghost (index 1) at current depth
            value = expectimax(successor, self.depth, 1)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction
    
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Win/Lose states - highest priority
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')
    
    # Base game score
    score = currentGameState.getScore()
    
    # Get game state information
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    
    # FOOD EVALUATION
    foodList = food.asList()
    if foodList:
        # Distance to closest food - major component
        foodDistances = [manhattanDistance(pos, food) for food in foodList]
        closestFoodDistance = min(foodDistances)
        score += 10.0 / closestFoodDistance
        
        # Penalty for remaining food - encourages eating
        score -= len(foodList) * 4
        
        # Bonus for fewer food dots (progress toward winning)
        totalFood = food.width * food.height
        foodEaten = totalFood - len(foodList)
        score += foodEaten * 2
    
    # GHOST EVALUATION
    for i, ghost in enumerate(ghostStates):
        ghostPos = ghost.getPosition()
        distance = manhattanDistance(pos, ghostPos)
        
        if scaredTimes[i] > 0:
            # Ghost is scared - hunt it aggressively
            if distance == 0:
                score += 200  # Ate the ghost!
            elif distance <= 3:
                # Close to scared ghost - very good
                score += 100.0 / (distance + 1)
            else:
                # Far from scared ghost - still good to chase
                score += 50.0 / (distance + 1)
        else:
            # Ghost is dangerous - avoid it
            if distance == 0:
                return float('-inf')  # Death
            elif distance == 1:
                score -= 1000  # Extremely dangerous
            elif distance == 2:
                score -= 500   # Very dangerous
            elif distance == 3:
                score -= 100   # Dangerous
            else:
                # Safe distance - small bonus for staying away
                score += distance * 0.5
    
    # POWER PELLET EVALUATION
    if capsules:
        # Distance to closest power pellet
        capsuleDistances = [manhattanDistance(pos, capsule) for capsule in capsules]
        closestCapsuleDistance = min(capsuleDistances)
        
        # Bonus for being close to power pellets, especially when ghosts are near
        nearbyGhosts = sum(1 for ghost in ghostStates 
                          if manhattanDistance(pos, ghost.getPosition()) <= 5 
                          and not ghost.scaredTimer > 0)
        
        if nearbyGhosts > 0:
            # Power pellet is more valuable when ghosts are nearby
            score += 50.0 / (closestCapsuleDistance + 1)
        else:
            # Still valuable but less urgent
            score += 20.0 / (closestCapsuleDistance + 1)
    
    # SCARED GHOST BONUS
    scaredGhostCount = sum(1 for timer in scaredTimes if timer > 0)
    if scaredGhostCount > 0:
        score += 100 * scaredGhostCount  # Bonus for having scared ghosts
        
        # Extra bonus for longer scared time remaining
        totalScaredTime = sum(timer for timer in scaredTimes)
        score += totalScaredTime * 2
    
    # STRATEGIC POSITIONING
    # Slight bonus for central positions (more options)
    width, height = food.width, food.height
    centerX, centerY = width // 2, height // 2
    distanceFromCenter = manhattanDistance(pos, (centerX, centerY))
    score += 5.0 / (distanceFromCenter + 1)
    
    # EFFICIENCY BONUS
    # Bonus for being in areas with more food density
    if foodList:
        nearbyFood = sum(1 for food in foodList 
                        if manhattanDistance(pos, food) <= 3)
        score += nearbyFood * 2
    
    return score

# Abbreviation
better = betterEvaluationFunction
