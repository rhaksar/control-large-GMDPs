import cvxpy as cp
import numpy as np
import time

from simulators.fires.ForestElements import Tree
from simulators.fires.LatticeForest import LatticeForest


def solve_tree_Vw_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    """
    Solve a linear program to approximate the value function for a single Tree.

    :param alpha: fire spread parameter in Tree dynamics.
    :param beta: fire persistence parameter in Tree dynamics.
    :param delta_beta: control effectiveness parameter in Tree dynamics.
    :param gamma: discount factor.

    :return: weights of basis function parameterization.
    """

    # setup and definitions
    healthy, on_fire, burnt = 0, 1, 2

    active = on_fire

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

            # reward and expectations are based on the number of healthy neighbors (hi)
            # and number of on fire neighbors (fi)
            # iterate through the possible values of these quantities to implement the constraints
            for hi in range(number_neighbors+1):
                for fi in range(number_neighbors+1-hi):

                    basis = weights[0] + weights[1]*(state == healthy) + weights[2]*(state == on_fire)*hi

                    # need to consider states of neighbor of neighbors to compute expected reward
                    neighbors_of_hi = 3*hi - 2*(hi == 3) - 4*(hi == 4)
                    for k in range(2**neighbors_of_hi):
                        xk = np.base_repr(k, base=2).zfill(2*number_neighbors)

                        # compute number of neighbors on fire for the neighbors of neighbors of the Tree
                        fj[3] = xk[0:3].count(str(active))
                        fj[2] = xk[2:5].count(str(active))
                        fj[1] = xk[4:7].count(str(active))
                        fj[0] = xk[6:8].count(str(active)) + (xk[0] == str(active))

                        # constraints based on Bellman operator approximations
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
            # constraints simplify when the Tree is in the burnt state
            constraints += [phi >= weights[0] - gamma*weights[0]]
            constraints += [phi >= -weights[0] + gamma*weights[0]]

    alp = cp.Problem(objective, constraints)
    print('Approximate Linear Program for single Tree')
    print('number of constraints: {0:d}'.format(len(constraints)))
    tic = time.clock()
    alp.solve(solver=cp.ECOS)
    toc = time.clock()
    print('completed in {0:0.2f}s = {1:0.2f}m'.format(toc-tic, (toc-tic)/60))
    print('problem status: {0}'.format(alp.status))
    print('error: {0:0.2f}'.format(phi.value))
    print('weight(s): ')
    print(weights.value)

    return weights.value


def solve_tree_Qw_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    """
    Solve a linear program to approximate the state-action function for a single Tree.

    :param alpha: fire spread parameter in Tree dynamics.
    :param beta: fire persistence parameter in Tree dynamics.
    :param delta_beta: control effectiveness parameter in Tree dynamics.
    :param gamma: discount factor.

    :return: weights of basis function parameterization.
    """

    # setup and definitions
    healthy, on_fire, burnt = 0, 1, 2
    active = on_fire

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

            # reward and expectations are based on the number of healthy neighbors (hi)
            # and number of on fire neighbors (fi)
            # iterate through the possible values of these quantities to implement the constraints
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

                        # implement constraints based on Bellman operator approximations
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
            # constraints simplify for the burnt state
            constraints += [phi >= weights[0] - gamma*weights[0]]
            constraints += [phi >= -weights[0] + gamma*weights[0]]

    alp = cp.Problem(objective, constraints)
    print('Approximate Qw Linear Program for single Tree')
    print('number of constraints: {0:d}'.format(len(constraints)))
    tic = time.clock()
    alp.solve(solver=cp.ECOS)
    toc = time.clock()
    print('completed in {0:0.2f}s = {1:0.2f}m'.format(toc-tic, (toc-tic)/60))
    print('problem status: {0}'.format(alp.status))
    print('error: {0:0.2f}'.format(phi.value))
    print('weight(s): ')
    print(weights.value)

    return weights.value


def new_controller_Vw(simulation, delta_beta=0.54, gamma=0.95, capacity=4):
    """
    Implementation of the policy resulting from the basis approximation of the value function.

    :param simulation: LatticeForest simulator object.
    :param delta_beta: control effectiveness parameter in the Tree dynamics.
    :param gamma: discount factor.
    :param capacity: maximum amount of control effort allowed per time step.

    :return: dictionary describing the action. keys are the positions of Trees and values are (delta_alpha, delta_beta)
    indicating the control effort.
    """

    # weights are stored here as they can be computed offline and used online
    weights = [-24.6317755, 5.18974134, -1.37294744]
    action = []

    # priority for control of fires is based on the how easily fire can spread in the future
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

    # sort by priority and apply capacity constraint
    action = sorted(action, key=lambda x: x[0], reverse=True)[:capacity]
    action = [x[1] for x in action]
    action = {x: (0, delta_beta) if x in action else (0, 0) for x in simulation.group.keys()}

    return action


def new_controller_Qw(simulation, delta_beta=0.54, gamma=0.95, capacity=4):
    """
    Implementation of the policy resulting from the basis approximation of the state-action function.

    :param simulation: LatticeForest simulator object.
    :param delta_beta: control effectiveness parameter in the Tree dynamics.
    :param gamma: discount factor.
    :param capacity: maximum amount of control effort allowed per time step.

    :return: dictionary describing the action. keys are the positions of Trees and values are (delta_alpha, delta_beta)
    indicating the control effort.
    """

    if capacity <= 0:
        return {x: (0, 0) for x in simulation.group.keys()}

    # weights are stored here as they can be computed offline and used online
    weights = [3.43034056, -0.37151703, -1.54798762,  0.27803541]

    action = []

    # priority of fires is based on the number of healthy neighbors they have
    for fire in simulation.fires:
        element = simulation.group[fire]

        number_healthy_neighbors = 0
        for n in element.neighbors:
            neighbor = simulation.group[n]
            if neighbor.is_healthy(neighbor.state):
                number_healthy_neighbors += 1

        element_weight = weights[3]*number_healthy_neighbors
        action.append((element_weight, fire))

    # sort by priority and apply capacity constraint
    action = sorted(action, key=lambda x: x[0], reverse=True)[:capacity]
    action = [x[1] for x in action]
    action = {x: (0, delta_beta) if x in action else (0, 0) for x in simulation.group.keys()}

    return action


def solve_priorwork_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    """
    Solve a linear program to approximate the value function for a single Tree, based on the choice of functions
    from prior work.

    :param alpha: fire spread parameter in Tree dynamics.
    :param beta: fire persistence parameter in Tree dynamics.
    :param delta_beta: control effectiveness parameter in Tree dynamics.
    :param gamma: discount factor.

    :return: weights of basis function parameterization.
    """

    # setup and definitions
    healthy, on_fire, burnt = 0, 1, 2

    tree = Tree(alpha, beta, model='linear')
    number_neighbors = 4

    def reward(tree_state, number_healthy_numbers):
        return (tree_state == healthy) - (tree_state == on_fire)*number_healthy_numbers

    phi = cp.Variable()
    weights = cp.Variable(3)

    objective = cp.Minimize(phi)
    constraints = []

    for state in tree.state_space:

        # expectations of the reward and basis functions are based on the number of healthy neighbors (hi) and the
        # number of on fire neighbors (fi)
        # iterate through the possible values of these quantities to compute the expectations
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

    # solve for weights
    alp = cp.Problem(objective, constraints)
    print('Approximate Linear Program from prior work')
    print('number of constraints: {0:d}'.format(len(constraints)))
    tic = time.clock()
    alp.solve(solver=cp.ECOS)
    toc = time.clock()
    print('completed in {0:0.2f}s = {1:0.2f}m'.format(toc-tic, (toc-tic)/60))
    print('problem status: {0}'.format(alp.status))
    print('error: {0:0.2f}'.format(phi.value))
    print('weight(s): ')
    print(weights.value)

    return weights.value


def prior_controller(simulation, delta_beta=0.54, gamma=0.95, capacity=4):
    """
    Implementation of the policy resulting from the basis approximation from prior work of the value function.

    :param simulation: LatticeForest simulator object.
    :param delta_beta: control effectiveness parameter in the Tree dynamics.
    :param gamma: discount factor.
    :param capacity: maximum amount of control effort allowed per time step.

    :return: dictionary describing the action. keys are the positions of Trees and values are (delta_alpha, delta_beta)
    indicating the control effort.
    """

    # weights are stored here as they can be computed offline and used online
    weights = [-25.89014413, -31.92832099, -30.78014474]
    action = []

    # fire priority for control is random, all fires have equal value
    for fire in simulation.fires:
        value = gamma*delta_beta*(weights[2]-weights[1])
        action.append((value, fire))

    # sort by priority and apply capacity constraint
    action = sorted(action, key=lambda x: x[0], reverse=True)[:capacity]
    action = [x[1] for x in action]
    action = {x: (0, delta_beta) if x in action else (0, 0) for x in simulation.group.keys()}

    return action


def run_simulation(simulation, method='prior'):
    """
    Helper function to run a single simulation and report the number of Trees that are healthy, on fire, and burnt.

    :param simulation: LatticeForest simulator object.
    :param method: string indicating the choice of controller - 'prior', 'new_v', or 'new_q'

    :return: list containing the number of Trees that are [healthy, on fire, burnt].
    """

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
    # compute weights for basis function approximation
    # prior_weights = solve_priorwork_alp()
    # new_weights = solve_tree_Vw_alp()
    # new_weights = solve_tree_Qw_alp()

    # simulation and policy ('prior', 'new_v', or 'new_q') setup
    sim = LatticeForest(50, tree_model='linear')
    control_method = 'new_v'

    # run many simulations and report statistics
    stats_batch = []
    for seed in range(100):
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        stats = run_simulation(sim, method=control_method)
        stats_batch.append(stats[0]/np.sum(stats))

        if (seed+1) % 100 == 0:
            print('completed %d simulations' % (seed+1))

    print('method: {0}'.format(control_method))
    print('mean healthy trees [percent]: {0:0.2f}'.format(np.mean(stats_batch)*100))
    print('median healthy trees [percent]: {0:0.2f}'.format(np.median(stats_batch)*100))
