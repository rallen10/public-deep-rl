import numpy as np
import pickle

def episode_returns(rewards, gamma):
    """Compute discounted returns from single episode
    
    Args:
        rewards (List[float]): reward at each time step
        gamma (float): discount factor

    Returns:
        List[float]: discounted return at each timestep in episode

    Notes:
        Always assumes final timestep is terminal and return is 0
    """
    n = len(rewards)
    next_G = 0
    G = np.zeros(n)
    for i in reversed(range(n)):
        G[i] = rewards[i] + gamma * next_G
        next_G = G[i]
    return list(G)

def episode_deltas(rewards, values, gamma):
    """Compute the TD residual at each time step in episode with discount gamma
    
    Params
    ======
        rewards (np.array): array of rewards at each time step
        values (np.array): array of value or value estimates at each time step
        gamma (float): discount factor

    Returns
    =======
        List[float]: delta values at each time step in episode

    Notes
    =====
        Assumes episode always ends with done
        Reference https://arxiv.org/pdf/1506.02438.pdf
    """
    assert 0.0 <= gamma and gamma <= 1.0
    deltas = [r + gamma*V_1 - V_0 for r, V_1, V_0 in zip(rewards[:-1], values[1:], values[:-1])]
    deltas.append(rewards[-1] - values[-1])
    return deltas


def episode_general_advantages(rewards, values, lam, gamma):
    """ Compute the general advantage estimate at each timestep in episode

    Params
    ======
        rewards (np.array): array of rewards at each time step
        values (np.array): array of value or value estimates at each time step
        lam (float): lambda parameter
        gamma (float): discount factor

    Returns
    =======
        List[float]: general advantage estimate at each timestep

    Notes
    =====
        Assumes episode always ends with done
        Reference https://arxiv.org/pdf/1506.02438.pdf
    """
    assert 0.0 <= lam and lam <= 1.0
    assert 0.0 <= gamma and gamma <= 1.0

    # compute TD residuals
    deltas = episode_deltas(rewards, values, gamma)
    n = len(deltas)

    # iteratively compute advantage
    A_next = 0.0
    A = np.zeros(n)
    for t in reversed(range(n)):
        A[t] = deltas[t] + gamma*lam*A_next
        A_next = A[t]
    return A


def general_advantage_estimation(rewards, values, dones, lam, gamma, next_value, next_done):
    """ Compute the general advantage estimate at each timestep in episode

    Params
    ======
        rewards (np.array): array of rewards at each time step
        values (np.array): array of value or value estimates at each time step
        dones (np.array): array of dones (absorbing state) at each time step
        lam (float): lambda parameter
        gamma (float): discount factor
        next_value (float): value of the N+1 state (e.g. 0 for absorbing state)
        next_done (bool): is N+1 state an absorbing or terminal state

    Returns
    =======
        List[float]: general advantage estimate at each timestep

    Notes
    =====
        Assumes episode always ends with done
        Reference https://arxiv.org/pdf/1506.02438.pdf
    """
    n = len(rewards)
    # if not len(dones) == len(values) == n + 1: raise ValueError
    # if not (values.shape == dones.shape): raise ValueError
    # if not (rewards.shape[1] == values.shape[1] == dones.shape[1] == 1): raise ValueError
    if not (rewards.shape == values.shape == dones.shape): raise ValueError
    if not (0.0 <= lam and lam <= 1.0): raise ValueError
    if not (0.0 <= gamma and gamma <= 1.0): raise ValueError

    # iteratively compute advantage
    adv_last = 0.0
    advantages = np.zeros(n)
    deltas = np.zeros(n)
    returns = np.zeros(n)
    for t in reversed(range(n)):
        if t == n - 1: # handle the final time step
            nonterm_next = 1.0 - next_done   # check if next is non-terminal (nt)
            val_next = next_value
        else:
            nonterm_next = 1.0 - dones[t+1]
            val_next = values[t+1]
        deltas[t] = rewards[t] + gamma * val_next * nonterm_next - values[t]
        advantages[t] = adv_last = deltas[t] + gamma * lam * adv_last * nonterm_next
    returns = advantages + values
    return advantages, returns, deltas