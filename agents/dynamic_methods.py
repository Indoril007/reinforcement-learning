import numpy as np
from .core import Greedy, EpsilonGreedy

class DynamicMethods:

    def __init__(self, env, agent):
        self.env = env
        self.transitions = env.P
        self.end_states = env.end_states
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        self.agent = agent
        self.policy = agent.policy
        self.greedy = Greedy(agent.values)
        self.epsilon_greedy = EpsilonGreedy(agent.values)
        self.discount_factor = agent.discount_factor

    def policy_evaluation(self, stop_threshold=0.00001, max_iterations=None):
        iterations = 0

        while True:
            max_diff = 0
            for state in range(self.num_states):
                if state in self.end_states:
                    self.agent.set_value(state, 0)
                    continue

                new_value = self._get_state_value(state)
                max_diff = max(max_diff, abs(self.agent.get_value(state) - new_value))
                self.agent.set_value(state, new_value)

            iterations += 1
            if (max_iterations is not None and iterations >= max_iterations) or (max_diff < stop_threshold):
                break

    def q_evaluation(self):
        for state in range(self.num_states):
            if state in self.end_states:
                continue
            for action in range(self.num_actions):
                self.agent.set_q_value(state, action, self._get_action_value(state, action))

    def _get_state_value(self, state):
        value = 0
        action_probs = self.policy.get_action_probs(state)

        for action, action_prob in enumerate(action_probs):

            action_value = self._get_action_value(state, action)
            value += action_prob*action_value

        return value

    def _get_opt_state_value(self, state):
        maxval = None

        for a in range(self.num_actions):
            transitions = self.transitions[state][a]
            action_value = self._get_action_value(state, a)
            maxval = action_value if maxval is None else max(maxval, action_value)

        return maxval

    def _get_action_value(self, state, action):
        action_value = 0
        transitions = self.transitions[state][action]
        for transition in transitions:
            prob, next_state, reward, done = transition
            next_value = 0 if done else self.agent.get_value(next_state)
            action_value += prob * (reward + self.discount_factor*next_value)

        return action_value

    def policy_improvement(self):
        changed = False
        for state in range(self.num_states):
            optimal_action = 0
            maxval = None
            for action in range(self.num_actions):
                action_value = self._get_action_value(state, action)

                if maxval is None or action_value > maxval:
                    maxval = action_value
                    optimal_action = action

            changed = changed or self.policy.set_optimal_action(state, optimal_action)
        return changed

    def policy_iteration(self, eval_iterations = 100):

        changed = True
        while changed:
            self.policy_evaluation(max_iterations=eval_iterations)
            changed = self.policy_improvement()

    def value_iteration(self, stop_threshold = 0.00001, max_iterations = None):
        iterations = 0

        while True:
            max_diff = 0
            for state in range(self.num_states):
                if state in self.end_states:
                    self.agent.set_value(state, 0)
                    continue

                new_value = self._get_opt_state_value(state)
                max_diff = max(max_diff, abs(self.agent.get_value(state) - new_value))
                self.agent.set_value(state, new_value)

            iterations += 1
            if (max_iterations is not None and iterations >= max_iterations) or (max_diff < stop_threshold):
                break
