# control-large-GMDPs

A repository to support the paper [Controlling Large, Graph-based MDPs with Global Control Capacity Constraints: An Approximate LP Solution](https://ieeexplore.ieee.org/document/8618745).

Paper citation:
```
@InProceedings{8618745, 
    author={R. N. Haksar and M. Schwager}, 
    booktitle={2018 IEEE Conference on Decision and Control (CDC)}, 
    title={Controlling Large, Graph-based MDPs with Global Control Capacity Constraints: An Approximate LP Solution}, 
    year={2018}, 
    pages={35-42}, 
    doi={10.1109/CDC.2018.8618745}, 
    ISSN={2576-2370}, 
    month={Dec},}
```

### Requirements
- Written for Python 3.5
- Requires `numpy` and `cvxpy` packages
- Requires the [simulators](https://github.com/rhaksar/simulators) repository: clone the repository into the root level of this repository 

### Files
- `controlEpidemic.py`: Solves an approximate linear program to estimate the Q-function and implements the resulting controller.
- `controlForestFires.py`: Solves an apprxoimate linaer program to estimate the value function and implements the resulting controller. Also implements a method from literature. 
