import numpy as np
import pytest

from agent import (
    INIT_L2_PENALTY,
    INIT_LEARNING_RATE,
    INIT_DISCOUNT_FACTOR,
    GameScheduler,
    RewardFunction,
    FeatureFunction,
    Agent,
)
from state import State, WAIT_ACTION


# ------------------------------------------------------------
# Helper fixtures for State
# ------------------------------------------------------------

@pytest.fixture
def simple_state():
    return State(n_rows=2, n_cols=2, players=["A", "B"])


# ------------------------------------------------------------
# Tests for module-level constants
# ------------------------------------------------------------

def test_init_l2_penalty_constant():
    assert isinstance(INIT_L2_PENALTY, float)
    assert INIT_L2_PENALTY >= 0.0


def test_init_learning_rate_constant():
    assert isinstance(INIT_LEARNING_RATE, float)
    assert 0.0 < INIT_LEARNING_RATE <= 1.0


def test_init_discount_factor_constant():
    assert isinstance(INIT_DISCOUNT_FACTOR, float)
    assert 0.0 <= INIT_DISCOUNT_FACTOR < 1.0


# ------------------------------------------------------------
# Tests for GameScheduler helper class
# ------------------------------------------------------------

class TestGameScheduler:
    def test_init_creates_instance(self):
        scheduler = GameScheduler(n_episodes=10)
        assert isinstance(scheduler, GameScheduler)

    def test_call_returns_valid_config_dict(self):
        n_episodes = 5
        scheduler = GameScheduler(n_episodes=n_episodes)
        cfg = scheduler(episode=0)

        assert isinstance(cfg, dict)
        for key in ("n_rows", "n_cols", "n_players", "n_steps"):
            assert key in cfg
            assert isinstance(cfg[key], int)
            assert cfg[key] > 0


# ------------------------------------------------------------
# Tests for RewardFunction helper class
# ------------------------------------------------------------

class TestRewardFunction:
    def test_init_sets_player(self):
        rf = RewardFunction(player="A")
        assert rf.player == "A"

    def test_call_returns_float(self, simple_state):
        rf = RewardFunction(player="A")
        state = simple_state
        action = WAIT_ACTION
        next_state = state.copy()
        reward = rf(state, action, next_state)

        assert isinstance(reward, float)


# ------------------------------------------------------------
# Tests for FeatureFunction helper class
# ------------------------------------------------------------

class TestFeatureFunction:
    def test_init_sets_number_of_features(self):
        ff = FeatureFunction(player="A")
        assert isinstance(ff.N_FEATURES, int)
        assert ff.N_FEATURES > 0

    def test_call_returns_correct_shape_and_dtype(self, simple_state):
        ff = FeatureFunction(player="A")
        phi = ff(simple_state)

        assert isinstance(phi, np.ndarray)
        assert phi.shape == (ff.N_FEATURES,)
        assert phi.dtype == np.float32


# ------------------------------------------------------------
# Dummy helper classes for Agent tests
# (Agent tests are decoupled from concrete RewardFunction/FeatureFunction)
# ------------------------------------------------------------

class DummyRewardFunction:
    def __init__(self, player: str) -> None:
        self.player = player

    def __call__(self, state, action, next_state) -> float:
        # Constant reward for deterministic tests
        return 1.0


class DummyFeatureFunction:
    def __init__(self, player: str) -> None:
        self.player = player
        self.N_FEATURES = 2

    def __call__(self, state) -> np.ndarray:
        # Simple non-zero feature vector
        return np.array([1.0, 0.0], dtype=np.float32)


@pytest.fixture
def dummy_agent():
    reward = DummyRewardFunction(player="A")
    features = DummyFeatureFunction(player="A")
    agent = Agent(
        player="A",
        l2_penalty=0.1,
        learning_rate=0.5,
        discount_factor=0.9,
        reward_function=reward,
        feature_function=features,
    )
    return agent


# ------------------------------------------------------------
# Tests for Agent.__init__
# ------------------------------------------------------------

def test_agent_init_sets_basic_attributes(dummy_agent):
    agent = dummy_agent
    assert agent.player == "A"
    assert agent.l2_penalty == 0.1
    assert agent.learning_rate == 0.5
    assert agent.discount_factor == 0.9
    assert isinstance(agent.reward_function, DummyRewardFunction)
    assert isinstance(agent.feature_function, DummyFeatureFunction)
    assert isinstance(agent.theta, dict)
    assert hasattr(agent, "epsilon")


# ------------------------------------------------------------
# Tests for Agent.ensure_action
# ------------------------------------------------------------

def test_agent_ensure_action_creates_parameters(dummy_agent, simple_state):
    agent = dummy_agent
    action = 1
    agent.ensure_action(action)

    assert action in agent.theta
    assert isinstance(agent.theta[action], np.ndarray)
    assert agent.theta[action].shape == (agent.feature_function.N_FEATURES,)

    q = agent.calculate_q_value(simple_state, action)
    assert isinstance(q, (float, np.floating))


# ------------------------------------------------------------
# Tests for Agent.calculate_q_value
# ------------------------------------------------------------

def test_agent_calculate_q_value_returns_float_and_is_deterministic(dummy_agent, simple_state):
    agent = dummy_agent
    action = 0
    agent.ensure_action(action)
    q1 = agent.calculate_q_value(simple_state, action)
    q2 = agent.calculate_q_value(simple_state, action)

    assert isinstance(q1, (float, np.floating))
    assert q1 == q2


# ------------------------------------------------------------
# Tests for Agent.calculate_state_value
# ------------------------------------------------------------

def test_agent_calculate_state_value_matches_max_over_actions(dummy_agent, simple_state):
    agent = dummy_agent
    state = simple_state

    legal_actions = state.get_legal_actions()
    for a in legal_actions:
        agent.ensure_action(a)

    q_values = [agent.calculate_q_value(state, a) for a in legal_actions]
    v = agent.calculate_state_value(state)

    assert isinstance(v, (float, np.floating))
    assert v == max(q_values)


# ------------------------------------------------------------
# Tests for Agent.calculate_gradient
# ------------------------------------------------------------

def test_agent_calculate_gradient_shape_and_uses_l2(dummy_agent, simple_state):
    agent = dummy_agent
    action = 0
    agent.ensure_action(action)

    state = simple_state
    q_value = agent.calculate_q_value(state, action)
    target_q_value = q_value + 1.0

    grad = agent.calculate_gradient(state, action, q_value, target_q_value)

    assert isinstance(grad, np.ndarray)
    assert grad.shape == (agent.feature_function.N_FEATURES,)

    pure_td_grad = (q_value - target_q_value) * agent.feature_function(state)
    assert not np.allclose(grad, pure_td_grad)


# ------------------------------------------------------------
# Tests for Agent.select_action
# ------------------------------------------------------------

def test_agent_select_action_training_returns_legal_action(dummy_agent, simple_state):
    agent = dummy_agent
    state = simple_state

    for a in state.get_legal_actions():
        agent.ensure_action(a)

    action = agent.select_action(state, training=True)
    assert action in state.get_legal_actions()


def test_agent_select_action_eval_is_greedy(dummy_agent, simple_state, monkeypatch):
    agent = dummy_agent
    state = simple_state

    state.player_positions = {
        "A": (0, 0),
        "B": (1, 1),
    }
    state.next_player = "A"
    state.tagged_player = "A"
    state.just_tagged = True

    legal_actions = state.get_legal_actions()
    for a in legal_actions:
        agent.ensure_action(a)

    def fake_calculate_q_value(s, a):
        return float(a)

    monkeypatch.setattr(agent, "calculate_q_value", fake_calculate_q_value)

    q_values = [agent.calculate_q_value(state, a) for a in legal_actions]
    best_q = max(q_values)

    selected = agent.select_action(state, training=False)

    assert selected in legal_actions
    assert agent.calculate_q_value(state, selected) == best_q


# ------------------------------------------------------------
# Tests for Agent.update_model
# ------------------------------------------------------------

def test_agent_update_model_non_terminal_changes_q_value(dummy_agent, simple_state):
    agent = dummy_agent
    state = simple_state

    state.player_positions = {
        "A": (0, 0),
        "B": (1, 1),
    }
    state.next_player = "A"
    state.tagged_player = "A"
    state.just_tagged = True

    action = state.get_legal_actions()[0]
    agent.ensure_action(action)

    next_state = state.copy()
    next_state.step(action)

    q_before = agent.calculate_q_value(state, action)
    agent.update_model(state, action, next_state, terminal=False)
    q_after = agent.calculate_q_value(state, action)

    assert isinstance(q_after, (float, np.floating))
    assert q_before != q_after


def test_agent_update_model_terminal_changes_q_value(dummy_agent, simple_state):
    agent = dummy_agent
    state = simple_state

    state.player_positions = {
        "A": (0, 0),
        "B": (1, 1),
    }
    state.next_player = "A"
    state.tagged_player = "A"
    state.just_tagged = True

    action = state.get_legal_actions()[0]
    agent.ensure_action(action)

    next_state = state.copy()

    q_before = agent.calculate_q_value(state, action)
    agent.update_model(state, action, next_state, terminal=True)
    q_after = agent.calculate_q_value(state, action)

    assert isinstance(q_after, (float, np.floating))
    assert q_before != q_after


class TestAgentUpdateDirection:
    def test_update_moves_q_towards_higher_target_for_that_action_only(
        self, dummy_agent, simple_state, monkeypatch
    ):
        agent = dummy_agent
        state = simple_state

        state.player_positions = {
            "A": (0, 0),
            "B": (1, 1),
        }
        state.next_player = "A"
        state.tagged_player = "A"
        state.just_tagged = True

        legal_actions = state.get_legal_actions()
        assert len(legal_actions) >= 2
        action_update = legal_actions[0]
        action_other = legal_actions[1]

        agent.ensure_action(action_update)
        agent.ensure_action(action_other)

        monkeypatch.setattr(agent, "calculate_state_value", lambda s: 0.0)

        q_update_before = agent.calculate_q_value(state, action_update)
        q_other_before = agent.calculate_q_value(state, action_other)

        def reward_fn_higher(s, a, ns):
            return q_update_before + 1.0

        monkeypatch.setattr(agent.reward_function, "__call__", reward_fn_higher)

        next_state = state.copy()
        terminal = False

        for _ in range(3):
            agent.update_model(state, action_update, next_state, terminal)

        q_update_after = agent.calculate_q_value(state, action_update)
        q_other_after = agent.calculate_q_value(state, action_other)

        assert q_update_after > q_update_before
        assert q_other_after == pytest.approx(q_other_before)

    def test_update_with_lower_target_changes_q_for_that_action_only(
        self, dummy_agent, simple_state, monkeypatch
    ):
        agent = dummy_agent
        state = simple_state

        state.player_positions = {
            "A": (0, 0),
            "B": (1, 1),
        }
        state.next_player = "A"
        state.tagged_player = "A"
        state.just_tagged = True

        legal_actions = state.get_legal_actions()
        assert len(legal_actions) >= 2
        action_update = legal_actions[0]
        action_other = legal_actions[1]

        agent.ensure_action(action_update)
        agent.ensure_action(action_other)

        # Make next-state value 0 so target = reward
        monkeypatch.setattr(agent, "calculate_state_value", lambda s: 0.0)

        # Record current Q-values
        q_update_before = agent.calculate_q_value(state, action_update)
        q_other_before = agent.calculate_q_value(state, action_other)

        # Reward strictly below q_update_before, so the target is lower
        def reward_fn_lower(s, a, ns):
            return q_update_before - 1.0

        monkeypatch.setattr(agent.reward_function, "__call__", reward_fn_lower)

        next_state = state.copy()
        terminal = False

        for _ in range(3):
            agent.update_model(state, action_update, next_state, terminal)

        q_update_after = agent.calculate_q_value(state, action_update)
        q_other_after = agent.calculate_q_value(state, action_other)

        # Only requirement: updated action's Q changes, other action is unchanged
        assert q_update_after != q_update_before
        assert q_other_after == pytest.approx(q_other_before)
