import numpy as np
from state import State

########## EDIT THESE PARAMETERS ##########
INIT_L2_PENALTY = None
INIT_LEARNING_RATE = None
INIT_DISCOUNT_FACTOR = None
###########################################


class GameScheduler:
    '''
    Game scheduler for training a tag agent.

    This callable object specifies the game configuration for each episode
    during training.
    '''

    def __init__(self, n_episodes: int) -> None:
        '''
        Initialize the game scheduler.

        Args:
            n_episodes (int): Total number of episodes used during training.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement GameScheduler.__init__, then delete this line of code.'
        )
        ###########################################

    def __call__(self, episode: int | None) -> dict[str, int]:
        '''
        Schedule game configuration based on episode number during training.

        Args:
            episode (int | None): Current episode index during training. If
                no episode is provided, assume training is over.

        Returns:
            Dict[str, int]: A dictionary with the keys `n_rows`, `n_cols`,
                `n_players`, and `n_steps`, specifying the grid dimensions,
                number of players, and number of steps for initializing the
                game for episode `episode`.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement GameScheduler.__call__, then delete this line of code.'
        )
        ###########################################


class RewardFunction:
    '''
    Reward function for the tag game.

    This callable object computes a scalar reward for a state transition.
    '''

    def __init__(self, player: str) -> None:
        '''
        Initialize the reward function.

        Args:
            player (str): Identifier of the player associated with this reward function.
        '''
        self.player = player
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement RewardFunction.__init__, then delete this line of code.'
        )
        ###########################################

    def __call__(self, state: State, action: int, next_state: State) -> float:
        '''
        Compute scalar reward for a transition `(state, action, next_state)`.

        Args:
            state (State): Tag state before the action.
            action (int): Action taken by the agent.
            next_state (State): Tag state after the action.

        Returns:
            float: Reward value.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement RewardFunction.__call__, then delete this line of code.'
        )
        ###########################################


class FeatureFunction:
    '''
    Feature mapping from states to feature vectors.

    This callable object maps a state to a fixed-length numeric feature
    vector used by the linear function approximator.
    '''

    def __init__(self, player: str) -> None:
        '''
        Initialize the feature function.

        Args:
            player (str): Identifier of the player associated with this feature function.
            N_FEATURES (int): Number of features produced.
        '''
        self.player = player
        ########## EDIT THESE PARAMETERS ##########
        self.N_FEATURES = None
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement FeatureFunction.__init__, then delete this line of code.'
        )
        ###########################################

    def __call__(self, state: State) -> np.ndarray:
        '''
        Compute feature vector for the given state.

        Args:
            state (State): Current tag state.

        Returns:
            np.ndarray: Feature vector of shape `(N_FEATURES,)` and dtype `float32`.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement FeatureFunction.__call__, then delete this line of code.'
        )
        ###########################################


class Agent:
    '''
    Linear function approximation Q-learning agent.

    The agent maintains separate parameter vectors for each action and
    uses feature-based Q-learning:

    `Q(s, a) = φ(s)θ[a]`

    where `φ(s)` is the feature vector of the state.
    '''

    def __init__(
        self,
        player: str,
        l2_penalty: float,
        learning_rate: float,
        discount_factor: float,
        reward_function: RewardFunction,
        feature_function: FeatureFunction,
    ) -> None:
        '''
        Initialize the agent.

        Args:
            player (str): Player identifier controlled by the agent.
            l2_penalty (float): L2 regularization penalty (λ).
            learning_rate (float): Learning rate (α) for parameter updates.
            discount_factor (float): Discount factor (γ) for future rewards.
            reward_function (RewardFunction): Reward function instance.
            feature_function (FeatureFunction): Feature function instance.
        '''
        self.player = player
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.reward_function = reward_function
        self.feature_function = feature_function
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement Agent.__init__, then delete this line of code.'
        )
        ###########################################

    def ensure_action(self, action: int) -> None:
        '''
        Ensure that a parameter vector exists for the given action. If
        the action has no parameter vector, create one.

        Args:
            action (int): Action index for which to initialize parameters.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement Agent.ensure_action, then delete this line of code.'
        )
        ###########################################

    def calculate_q_value(self, state: State, action: int) -> float:
        '''
        Compute the Q-value for a given state and action.

        Args:
            state (State): Current environment state.
            action (int): Action index whose Q-value is to be computed.

        Returns:
            float: The estimated Q-value computed as the dot product of
                the action-specific parameter vector θ[a] and the feature vector φ(s).
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement Agent.calculate_q_value, then delete this line of code.'
        )
        ###########################################

    def calculate_state_value(self, state: State) -> float:
        '''
        Compute the maximum Q-value among all legal actions for a given state.

        Args:
            state (State): Current environment state.

        Returns:
            float: The state value `V(s)` defined as max_a Q(s, a).
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement Agent.calculate_state_value, then delete this line of code.'
        )
        ###########################################

    def calculate_gradient(
        self,
        state: State,
        action: int,
        q_value: float,
        target_q_value: float,
    ) -> np.ndarray:
        '''
        Compute the gradient of the squared TD loss under a linear
        function approximation with L2 regularization.

        Args:
            state (State): Environment state used to compute features φ(state).
            action (int): Action index corresponding to parameters θ[action].
            q_value (float): Current Q-value estimate for (state, action).
            target_q_value (float): Target Q-value for (state, action).

        Returns:
            np.ndarray: The gradient vector for θ[action] computed as
                g_a = (Q(s, a) - target_Q(s, a)) · φ(s) + λ θ_a.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement Agent.calculate_gradient, then delete this line of code.'
        )
        ###########################################

    def select_action(self, state: State, training: bool) -> int:
        '''
        Select an action.

        When `training` is False, return the action that maximizes
        the current Q-value. Otherwise, use whatever
        policy you choose to balance exploration with exploitation.

        Args:
            state (State): Current environment state.
            training (bool): Whether the agent is in training mode.

        Returns:
            int: Selected action.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement Agent.select_action, then delete this line of code.'
        )
        ###########################################

    def update_model(
        self,
        state: State,
        action: int,
        next_state: State,
        terminal: bool,
    ) -> None:
        '''
        Update the internal Q-function parameters using TD learning.

        The update is based on the temporal-difference error between the
        current Q-value and the target:

        Equations:
            TD target:
                target = r + γ * max_{a'} Q(s', a')    if not terminal
                target = r                             if terminal

            Parameter update:
                θ[a] ← θ[a] - α * gradient

        Args:
            state (State): State before taking the action.
            action (int): Action taken in the given state.
            next_state (State): Resulting state after the action.
            terminal (bool): Whether `next_state` is a terminal state.
        '''
        ########## INSERT YOUR CODE HERE ##########

        raise NotImplementedError(
            'Please implement Agent.update_model, then delete this line of code.'
        )
        ###########################################
