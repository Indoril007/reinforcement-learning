import numpy as np

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

        i = 0
        while True:
            state_next_values = self.get_state_next_values()
            max_diff = np.abs(state_next_values - V).max()
            np.copyto(V, state_next_values)
            i += 1
            if (iterations is not None and i >= iterations) or (max_diff < threshold):
                break

    def get_action_values(self) -> np.ndarray:
        """
        This functions calculate the values of an action in a state by propagating back values from subsequent states
        :return: Returns an shape=(nS, nA) numpy array with elements giving action values
        """
        V = self.agent.values
        gamma = self.discount
        T = np.transpose(self.env.numpized_transitions, (3, 0, 1, 2))
        probs, next_states, rewards, dones = T
        next_states = next_states.astype(np.int64)

        action_values = np.sum(probs * (rewards + gamma*(1-dones)*V[next_states]), axis=2)

        return action_values

    def get_state_next_values(self) -> np.ndarray:
        """
        This function calculates the values of states using values of actions in that state backpropgated from
        subsequent state values
        :return: Returns a shape=(nS) numpy array with elements giving state values
        """
        pi = self.agent.policy.get()
        action_values = self.get_action_values()
        state_next_values = np.sum(pi*action_values, axis=1)
        state_next_values[self.env.end_states] = 0

        return state_next_values

    def policy_improvement(self) -> bool:
        """
        This function improves the policy by making greedy selections according to values propagated from the next
        states
        :return: Returns a boolean variable indicating whether or not the policy has changed
        """
        action_values = self.get_action_values()
        optimal_actions = np.argmax(action_values, axis=1)
        set_all = np.vectorize(self.policy.set)
        changed = np.any(set_all(range(len(optimal_actions)), optimal_actions))

        return changed

    def policy_iteration(self, eval_iterations: int = 100) -> None:
        """
        This function performs policy iterations. That is to say it repeatedly evaluated the policy and then improves
        the policy according to the evaluated values
        :param eval_iterations: The number of iterations to perform evaluation before each policy improvement step
        """
        changed = True
        while changed:
            self.policy_evaluation(iterations=eval_iterations)
            changed = self.policy_improvement()

    def value_iteration(self, threshold: float = None, iterations: int = None) -> None:
        """
        This function performs value iterations. That is to say it essentialy performs policy iteration with only a
        the evaluation step performed for only one iteration
        :param threshold: Once the maximum difference between old and new values exceeds this threshold the value
            iteration will stop
        :param iterations: The maximum number of iterations to perform value iteration for
        """
        V = self.agent.values

        i = 0
        while True:
            state_next_values = np.amax(self.get_action_values(), axis=1)
            state_next_values[self.env.end_states] = 0
            max_diff = np.abs(state_next_values - V).max()
            np.copyto(V, state_next_values)
            i += 1
            if (iterations is not None and i >= iterations) or (threshold is not None and max_diff < threshold):
                break

