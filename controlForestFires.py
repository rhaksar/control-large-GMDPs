from collections import defaultdict
from cvxpy import *
import numpy as np
import os
import sys
import time

sys.path.append(os.getcwd() + '/simulators')
from fires.ForestElements import Tree
from fires.LatticeForest import LatticeForest


def solve_tree_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    healthy = 0
    on_fire = 1
    burnt = 2

    active = 1

    tree = Tree(alpha, beta, model='linear')
    number_neighbors = 4

    def reward(tree_state, number_healthy_numbers):
        return (tree_state == healthy) - (tree_state == on_fire)*number_healthy_numbers

    phi = Variable()
    weights = Variable(3)

    objective = Minimize(phi)
    constraints = []

    for state in tree.state_space:

        if state in [healthy, on_fire]:
            # for j in range(3**number_neighbors):
                # xj = np.base_repr(j, base=3).zfill(number_neighbors)

            for hi in range(number_neighbors+1):
                for fi in range(number_neighbors+1-hi):
                    # bi = 4 - hi - fi
                    # hi = xj.count(str(healthy))
                    # fi = xj.count(str(on_fire))

                    basis = weights[0] + weights[1]*(state == healthy) + weights[2]*(state == on_fire)*hi

                    neighbors_of_hi = 3*hi - 2*(hi == 3) - 4*(hi == 4)
                    for k in range(2**neighbors_of_hi):
                        xk = np.base_repr(k, base=2).zfill(2*number_neighbors)

                        fj = {i: None for i in range(number_neighbors)}
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

    alp = Problem(objective, constraints)
    print('Approximate Linear Program for single Tree')
    print('number of constraints: %d' % len(constraints))
    tic = time.clock()
    alp.solve(solver=ECOS)
    toc = time.clock()
    print('completed in %0.2fs = %0.2fm' % (toc-tic, (toc-tic)/60))
    print('problem status: %s' % alp.status)
    print('error: %0.2f' % phi.value)
    print('weight(s): ')
    print(weights.value)

    return weights.value


def solve_priorwork_alp(alpha=0.2, beta=0.9, delta_beta=0.54, gamma=0.95):
    healthy = 0
    on_fire = 1
    burnt = 2

    tree = Tree(alpha, beta, model='linear')
    number_neighbors = 4

    def reward(tree_state, number_healthy_numbers):
        return (tree_state == healthy) - (tree_state == on_fire)*number_healthy_numbers

    phi = Variable()
    weights = Variable(4)

    objective = Minimize(phi)
    constraints = []

    for state in tree.state_space:
        for hi in range(number_neighbors+1):
            for fi in range(number_neighbors+1-hi):

                basis = weights[0] + weights[1]*(state == healthy) + weights[2]*(state == on_fire) \
                        + weights[3]*(state == burnt)

                for apply_control in [False, True]:
                    control = (0, delta_beta) if apply_control else (0, 0)

                    expected_basis = weights[0] + weights[1]*tree.dynamics((state, fi, healthy), control) \
                                     + weights[2]*tree.dynamics((state, fi, on_fire), control) \
                                     + weights[3]*tree.dynamics((state, fi, burnt), control)

                    constraints += [phi >= basis - reward(state, hi) - gamma*expected_basis]
                    constraints += [phi >= -basis + reward(state, hi) + gamma*expected_basis]

    alp = Problem(objective, constraints)
    print('Approximate Linear Program from prior work')
    print('number of constraints: %d' % len(constraints))
    tic = time.clock()
    alp.solve(solver=ECOS)
    toc = time.clock()
    print('completed in %0.2fs = %0.2fm' % (toc-tic, (toc-tic)/60))
    print('problem status: %s' % alp.status)
    print('error: %0.2f' % phi.value)
    print('weight(s): ')
    print(weights.value)

    return weights.value


if __name__ == '__main__':
    new_weights = solve_tree_alp()

    # prior_weights = solve_priorwork_alp()

    # weights = [-24.63177549, 5.18974134, -1.37294744]
    # counter = 0
    # for hi in range(5):
    #     for fi in range(5-hi):
    #         bi = 4 - hi - fi
    #         print(hi, fi, bi, hi+fi+bi)
    #         counter += 1
    #
    # print(counter)
