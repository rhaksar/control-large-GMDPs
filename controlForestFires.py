from collections import defaultdict
import cvxpy as cp
import numpy as np
import os
import sys
import time

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from fires.ForestElements import Tree
from fires.LatticeForest import LatticeForest


def solve_tree_Vw_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    healthy = 0
    on_fire = 1
    burnt = 2

    active = 1

    tree = Tree(alpha, beta, model='linear')
    number_neighbors = 4
    fj = {i: None for i in range(number_neighbors)}

    def reward(tree_state, number_healthy_neighbors):
        return (tree_state == healthy) - (tree_state == on_fire)*number_healthy_neighbors

    phi = cp.Variable()
    weights = cp.Variable(3)

    objective = cp.Minimize(phi)
    constraints = []

    for state in tree.state_space:

        if state in [healthy, on_fire]:

            for hi in range(number_neighbors+1):
                for fi in range(number_neighbors+1-hi):

                    basis = weights[0] + weights[1]*(state == healthy) + weights[2]*(state == on_fire)*hi

                    neighbors_of_hi = 3*hi - 2*(hi == 3) - 4*(hi == 4)
                    for k in range(2**neighbors_of_hi):
                        xk = np.base_repr(k, base=2).zfill(2*number_neighbors)

                        fj[3] = xk[0:3].count(str(active))
                        fj[2] = xk[2:5].count(str(active))
                        fj[1] = xk[4:7].count(str(active))
                        fj[0] = xk[6:8].count(str(active)) + (xk[0] == str(active))

                        xk_sum = 0
                        for n in range(hi):
                            xk_sum += tree.dynamics((healthy, fj[n], healthy))

                        for apply_control in [False, True]:
                            control = (0, delta_beta) if apply_control else (0, 0)

                            expected_basis = weights[0] + weights[1]*tree.dynamics((state, fi, healthy), control) \
                                             + weights[2]*tree.dynamics((state, fi, on_fire), control)*xk_sum

                            if not apply_control:
                                constraints += [phi >= basis - reward(state, hi) - gamma*expected_basis]

                            constraints += [phi >= -basis + reward(state, hi) + gamma*expected_basis]

        elif state == burnt:
            constraints += [phi >= weights[0] - gamma*weights[0]]
            constraints += [phi >= -weights[0] + gamma*weights[0]]

    alp = cp.Problem(objective, constraints)
    print('Approximate Linear Program for single Tree')
    print('number of constraints: %d' % len(constraints))
    tic = time.clock()
    alp.solve(solver=cp.ECOS)
    toc = time.clock()
    print('completed in %0.2fs = %0.2fm' % (toc-tic, (toc-tic)/60))
    print('problem status: %s' % alp.status)
    print('error: %0.2f' % phi.value)
    print('weight(s): ')
    print(weights.value)

    return weights.value


def solve_tree_Qw_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    healthy, on_fire, burnt = 0, 1, 2
    active = 1

    tree = Tree(alpha, beta, model='linear')
    number_neighbors = 4
    fj = {i: None for i in range(number_neighbors)}

    def reward(tree_state, action, tree_next_state):
        return (tree_state == healthy) - (1 - action)*(tree_next_state == on_fire)

    phi = cp.Variable()
    weights = cp.Variable(4)

    objective = cp.Minimize(phi)
    constraints = []

    for state in tree.state_space:

        if state in [healthy, on_fire]:

            for hi in range(number_neighbors+1):
                for fi in range(number_neighbors+1-hi):

                    neighbors_of_hi = 3*hi - 2*(hi == 3) - 4*(hi == 4)
                    for k in range(2**neighbors_of_hi):
                        xk = np.base_repr(k, base=2).zfill(2*number_neighbors)

                        fj[3] = xk[0:3].count(str(active))
                        fj[2] = xk[2:5].count(str(active))
                        fj[1] = xk[4:7].count(str(active))
                        fj[0] = xk[6:8].count(str(active)) + (xk[0] == str(active))

                        xk_sum = 0
                        for n in range(hi):
                            xk_sum += tree.dynamics((healthy, fj[n], healthy))

                        for apply_control in [False, True]:
                            control = (0, delta_beta) if apply_control else (0, 0)
                            basis = weights[0] + weights[1]*(state==healthy) + weights[2]*(state==on_fire) + \
                                apply_control*weights[3]*(state==on_fire)*hi

                            expected_reward = 0
                            expected_bias = 0
                            expected_action = 0
                            for next_state in tree.state_space:
                                expected_reward += tree.dynamics((state, fi, next_state), control)*reward(state, apply_control, next_state)
                                expected_bias += tree.dynamics((state, fi, next_state), control)*(weights[0] +
                                                                                                  weights[1]*(next_state==healthy) +
                                                                                                  weights[2]*(next_state==on_fire))
                                expected_action += tree.dynamics((state, fi, next_state), control)*weights[3]*(next_state==on_fire)*xk_sum

                            constraints += [phi >= basis - expected_reward - gamma*expected_bias]
                            constraints += [phi >= expected_reward + gamma*expected_bias - basis]
                            constraints += [phi >= expected_reward + gamma*expected_bias + gamma*expected_action - basis]

        elif state == burnt:
            constraints += [phi >= weights[0] - gamma*weights[0]]
            constraints += [phi >= -weights[0] + gamma*weights[0]]

    alp = cp.Problem(objective, constraints)
    print('Approximate Qw Linear Program for single Tree')
    print('number of constraints: %d' % len(constraints))
    tic = time.clock()
    alp.solve(solver=cp.ECOS)
    toc = time.clock()
    print('completed in %0.2fs = %0.2fm' % (toc-tic, (toc-tic)/60))
    print('problem status: %s' % alp.status)
    print('error: %e' % phi.value)
    print('weight(s): ')
    print(weights.value)

    return weights.value


def new_controller_Vw(simulation, delta_beta=0.54, gamma=0.95, capacity=4):
    weights = [-24.6317755, 5.18974134, -1.37294744]
    action = []

    for fire in simulation.fires:
        element = simulation.group[fire]

        value = -weights[2]*delta_beta*gamma

        neighbor_sum = 0
        for n in element.neighbors:
            neighbor = simulation.group[n]
            if neighbor.is_healthy(neighbor.state):
                neighbor.query_neighbors(simulation.group)
                fj = neighbor.neighbors_states.count(True)
                neighbor_sum += neighbor.dynamics((neighbor.healthy, fj, neighbor.healthy))

        value *= neighbor_sum

        action.append((value, fire))

    action = sorted(action, key=lambda x: x[0], reverse=True)[:capacity]
    action = [x[1] for x in action]
    action = {x: (0, delta_beta) if x in action else (0, 0) for x in simulation.group.keys()}

    return action


def new_controller_Qw(simulation, delta_beta=0.54, gamma=0.95, capacity=4):

    if capacity <= 0:
        return {x: (0, 0) for x in simulation.group.keys()}

    weights = [3.43034056, -0.37151703, -1.54798762,  0.27803541]

    action = []

    for fire in simulation.fires:
        element = simulation.group[fire]

        number_healthy_neighbors = 0
        for n in element.neighbors:
            neighbor = simulation.group[n]
            if neighbor.is_healthy(neighbor.state):
                number_healthy_neighbors += 1

        element_weight = weights[3]*number_healthy_neighbors
        action.append((element_weight, fire))

    action = sorted(action, key=lambda x: x[0], reverse=True)[:capacity]
    action = [x[1] for x in action]
    action = {x: (0, delta_beta) if x in action else (0, 0) for x in simulation.group.keys()}

    return action


def solve_priorwork_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    healthy = 0
    on_fire = 1
    burnt = 2

    tree = Tree(alpha, beta, model='linear')
    number_neighbors = 4

    def reward(tree_state, number_healthy_numbers):
        return (tree_state == healthy) - (tree_state == on_fire)*number_healthy_numbers

    phi = cp.Variable()
    weights = cp.Variable(3)

    objective = cp.Minimize(phi)
    constraints = []

    for state in tree.state_space:
        for hi in range(number_neighbors+1):
            for fi in range(number_neighbors+1-hi):

                basis = weights[0]*(state == healthy) + weights[1]*(state == on_fire) \
                        + weights[2]*(state == burnt)

                for apply_control in [False, True]:
                    control = (0, delta_beta) if apply_control else (0, 0)

                    expected_basis = weights[0]*tree.dynamics((state, fi, healthy), control) \
                                     + weights[1]*tree.dynamics((state, fi, on_fire), control) \
                                     + weights[2]*tree.dynamics((state, fi, burnt), control)

                    constraints += [phi >= basis - reward(state, hi) - gamma*expected_basis]
                    constraints += [phi >= -basis + reward(state, hi) + gamma*expected_basis]

    alp = cp.Problem(objective, constraints)
    print('Approximate Linear Program from prior work')
    print('number of constraints: %d' % len(constraints))
    tic = time.clock()
    alp.solve(solver=cp.ECOS)
    toc = time.clock()
    print('completed in %0.2fs = %0.2fm' % (toc-tic, (toc-tic)/60))
    print('problem status: %s' % alp.status)
    print('error: %0.2f' % phi.value)
    print('weight(s): ')
    print(weights.value)

    return weights.value


def prior_controller(simulation, delta_beta=0.54, gamma=0.95, capacity=4):
    weights = [-25.89014413, -31.92832099, -30.78014474]
    action = []

    for fire in simulation.fires:
        value = gamma*delta_beta*(weights[2]-weights[1])
        action.append((value, fire))

    action = sorted(action, key=lambda x: x[0], reverse=True)[:capacity]
    action = [x[1] for x in action]
    action = {x: (0, delta_beta) if x in action else (0, 0) for x in simulation.group.keys()}

    return action


def run_simulation(simulation, method='prior'):

    while not simulation.end:

        action = None
        if method == 'prior':
            action = prior_controller(simulation)
        elif method == 'new_v':
            action = new_controller_Vw(simulation)
        elif method == 'new_q':
            action = new_controller_Qw(simulation)

        simulation.update(action)

    return simulation.stats


if __name__ == '__main__':
    # prior_weights = solve_priorwork_alp()
    # new_weights = solve_tree_Vw_alp()
    # new_weights = solve_tree_Qw_alp()

    sim = LatticeForest(50, tree_model='linear')
    control_method = 'new_q'

    stats_batch = []
    for seed in range(100):
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        stats = run_simulation(sim, method=control_method)
        stats_batch.append(stats[0]/np.sum(stats))

        if (seed+1) % 100 == 0:
            print('completed %d simulations' % (seed+1))

    print('method: %s' % control_method)
    print('mean healthy trees [percent]: %0.4f' % (np.mean(stats_batch)*100))
    print('median healthy trees [percent]: %0.4f' % (np.median(stats_batch)*100))
