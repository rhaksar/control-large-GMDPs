from collections import defaultdict
import cvxpy as cp
import numpy as np
import os
import pickle
import sys

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from epidemics.RegionElements import Region
from epidemics.WestAfrica import WestAfrica


def solve_region_alp(eta=0.14, delta_nu=0.12, gamma=0.9):

    region = Region(eta, model='linear')
    number_neighbors = 4

    def reward(region_state, region_next_state):
        reward = 0
        if region_state == region.healthy:
            reward += 1
        elif region_next_state == region.infected:
            reward -= 1
        return reward

    weights = cp.Variable(4)
    phi = cp.Variable()

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

                bias = weights[0] + weights[1]*region.is_healthy(xi_t) + weights[2]*region.is_infected(xi_t)
                control_effect = weights[3]*region.is_infected(xi_t)*number_healthy_neighbors

                constraints += [phi >= bias + apply_control*control_effect - expected_reward - gamma*bias]
                constraints += [phi >= expected_reward + gamma*bias - bias - apply_control*control_effect]
                constraints += [phi >= expected_reward + gamma*bias + gamma*control_effect - bias
                                - apply_control*control_effect]

    alp = cp.Problem(objective, constraints)
    print('Approximate Linear Program for single Region')
    print('number of constraints: %d' % len(constraints))
    alp.solve()
    print('problem status: %s' % alp.status)
    print('error: %e' % phi.value)
    print('weight(s): ')
    print(weights.value)

    return weights.value


def controller(simulation, delta_nu=0.12, capacity=3):
    weights = [-0.20321269, 9.99989824, -9.19026325, 0.02976983]
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

    while not simulation.end:
        action = controller(simulation)
        simulation.update(action)

    return [simulation.counter[name] for name in sim.group.keys()]


if __name__ == '__main__':
    # weights = solve_region_alp()

    file = open(base_path + '/simulators/west_africa_graph.pkl', 'rb')
    graph = pickle.load(file)
    file.close()

    outbreak = {('guinea', 'gueckedou'): 1, ('sierra leone', 'kailahun'): 1, ('liberia', 'lofa'): 1}
    sim = WestAfrica(graph, outbreak, region_model='linear', eta=defaultdict(lambda: 0.14))

    dt_batch = []
    for seed in range(1000):
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        dt = run_simulation(sim)
        dt_batch.append(np.median(dt))

    print('mean infection time [weeks]: %0.4f' % np.mean(dt_batch))
    print('median infection time [weeks]: %0.4f' % np.median(dt_batch))
