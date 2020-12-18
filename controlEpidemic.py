from collections import defaultdict
import cvxpy as cp
import numpy as np

from simulators.epidemics.RegionElements import Region
from simulators.epidemics.WestAfrica import WestAfrica


def solve_region_alp(eta=0.14, delta_nu=0.12, gamma=0.9):
    """
    Solve a linear program to approximate the state-action function for a single Region.

    :param eta: virus spread parameter.
    :param delta_nu: control effectiveness parameter.
    :param gamma: discount factor.

    :return: weights of the basis function parameterization of the value function for a single Region.
    """

    # dynamics and reward function definitions
    region = Region(eta, model='linear')
    number_neighbors = 4

    def reward(region_state, region_next_state):
        reward = 0
        if region_state == region.healthy:
            reward += 1
        if region_next_state == region.infected:
            reward -= 1
        return reward

    weights = cp.Variable(4)
    phi = cp.Variable()

    # iterate through combinations of region states and region neighbor's states
    objective = cp.Minimize(phi)
    constraints = []
    for xi_t in region.state_space:
        for j in range(3**number_neighbors):
            for apply_control in [False, True]:
                xj_t = np.base_repr(j, base=3).zfill(number_neighbors)
                number_healthy_neighbors = xj_t.count(str(region.healthy))
                number_infected_neighbors = xj_t.count(str(region.infected))

                control = (0, delta_nu) if apply_control else (0, 0)

                expected_reward = 0
                for xi_tp1 in region.state_space:
                    probability = region.dynamics((xi_t, number_infected_neighbors, xi_tp1), control)
                    expected_reward += probability*reward(xi_t, xi_tp1)

                # constraints based on Bellman operator approximations
                bias = weights[0] + weights[1]*region.is_healthy(xi_t) + weights[2]*region.is_infected(xi_t)
                control_effect = weights[3]*region.is_infected(xi_t)*number_healthy_neighbors

                constraints += [phi >= bias + apply_control*control_effect - expected_reward - gamma*bias]
                constraints += [phi >= expected_reward + gamma*bias - bias - apply_control*control_effect]
                constraints += [phi >= expected_reward + gamma*bias + gamma*control_effect - bias
                                - apply_control*control_effect]

    alp = cp.Problem(objective, constraints)
    print('Approximate Linear Program for single Region')
    print('number of constraints: {0:d}'.format(len(constraints)))
    alp.solve()
    print('problem status: {0}'.format(alp.status))
    print('error: {0:0.2f}'.format(phi.value))
    print('weight(s): ')
    print(weights.value)

    return weights.value


def controller(simulation, delta_nu=0.12, capacity=3):
    """
    Implementation of policy resulting from the basis function parameterization.

    :param simulation: WestAfrica simulation object.
    :param delta_nu: control effectiveness parameter.
    :param capacity: maximum amount of control effort allowed per time step.

    :return: dictionary describing the action. keys are the names of regions and values are (delta_eta, delta_nu)
    indicating the control effort.
    """
    # weights are stored so that the ALP does not need to be solved repeatedly
    weights = [-0.05215494, 7.2515494, -9.42372431, 0.03041379]

    action = []
    for name in simulation.group.keys():

        element = simulation.group[name]
        if element.is_infected(element.state):

            number_healthy_neighbors = 0
            for neighbor_name in element.neighbors:
                if element.is_healthy(simulation.group[neighbor_name].state):
                    number_healthy_neighbors += 1

            element_weight = weights[3]*number_healthy_neighbors
            action.append((element_weight, name))

    action = sorted(action, key=lambda x: x[0], reverse=True)[:capacity]
    action = [x[1] for x in action]
    action = {x: (0, delta_nu) if x in action else (0, 0) for x in simulation.group.keys()}

    return action


def run_simulation(simulation):
    """
    Run a simulation with the control policy, and return the length of time each Region spent in the infected state.
    """

    while not simulation.end:
        action = controller(simulation)
        simulation.update(action)

    return [simulation.counter[name] for name in sim.group.keys()]


if __name__ == '__main__':
    # compute the weights of the state-action function parameterization
    # weights = solve_region_alp()

    # initial conditoins
    outbreak = {('guinea', 'gueckedou'): 1, ('sierra leone', 'kailahun'): 1, ('liberia', 'lofa'): 1}
    sim = WestAfrica(outbreak, region_model='linear', eta=defaultdict(lambda: 0.14))

    # run many simulations and report statistics
    dt_batch = []
    for seed in range(100):
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        dt = run_simulation(sim)
        dt_batch.append(np.median(dt))

    print('mean infection time [weeks]: {0:0.2f}'.format(np.mean(dt_batch)))
    print('median infection time [weeks]: {0:0.2f}'.format(np.median(dt_batch)))
