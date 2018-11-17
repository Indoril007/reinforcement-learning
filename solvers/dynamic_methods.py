import numpy as np
from agents.core import Greedy, EpsilonGreedy

class DynamicMethods:

    def __init__(self, env, agent):
        self.env = env
        self.nS = env.nS
        self.nA = env.nA

        self.agent = agent
        self.policy = agent.policy
        self.discount = agent.discount

    def policy_evaluation(self, threshold: float = 0.00001, iterations: int = None) -> None:
        """
        This function performs policy evaluation by repeatedly propagating values back through the state space
        :param threshold: Once the maximum difference between old values and new values is less than this threshold
            the evaluation will stop
        :param iterations: The maximum number of iterations to run policy evaluation
        """

        V = self.agent.values
        pi = self.agent.policy.policy
        gamma = self.discount
        T = np.transpose(self.env.P, (3,0,1,2))
        probs, next_states, rewards, dones = T

        next_states = next_states.astype(np.int64)

        i = 0
        while True:

            action_values = np.sum(probs * (rewards + gamma*(1-dones)*V[next_states]), axis=2)
            state_next_values = np.sum(pi*action_values, axis=1)
            max_diff = np.max(np.abs(state_next_values - V))
            np.copyto(V, state_next_values)
            V[self.env.end_states] = 0

            i += 1
            if (iterations is not None and i >= iterations) or (max_diff < threshold):
                break

    # def q_evaluation(self):
    #     Q = self.agent.q_values
    #     for state in range(self.nS):
    #         if state in self.env.end_states:
    #             continue
    #         for action in range(self.nA):
    #             Q[state][action] = self.action_value(state, action)

    # def optimal_action_value(self, state):
    #     maxval = None

    #     for action in range(self.nA):
    #         action_value = self.action_value(state, action)
    #         maxval = action_value if maxval is None else max(maxval, action_value)

    #     return maxval

    # def policy_improvement(self):
    #     changed = False

    #     for state in range(self.nS):
    #         optimal_action = None
    #         maxval = None

    #         for action in range(self.nA):
    #             action_value = self.action_value(state, action)

    #             if maxval is None or action_value > maxval:
    #                 maxval = action_value
    #                 optimal_action = action

    #         changed = changed or self.policy.set(state, optimal_action)
    #     return changed

    # def policy_iteration(self, eval_iterations = 100):

    #     changed = True
    #     while changed:
    #         self.policy_evaluation(max_iterations=eval_iterations)
    #         changed = self.policy_improvement()

    # def value_iteration(self, stop_threshold = 0.00001, max_iterations = None):
    #     iterations = 0

    #     while True:
    #         max_diff = 0
    #         for state in range(self.nS):
    #             if state in self.env.end_states:
    #                 self.agent.set_value(state, 0)
    #                 continue

    #             new_value = self.optimal_action_value(state)
    #             max_diff = max(max_diff, abs(self.agent.get_value(state) - new_value))
    #             self.agent.set_value(state, new_value)

    #         iterations += 1
    #        if (max_iterations is not None and iterations >= max_iterations) or (max_diff < stop_threshold):
    #            break
