# control-large-GMDPs

A repository to support the paper [Controlling Large, Graph-based MDPs with Global Control Capacity Constraints: An Approximate LP Solution](https://msl.stanford.edu/sites/g/files/sbiybj8446/f/haksar_cdc2018.pdf).

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
- Developed with Python 3.6
- Requires [`cvxpy`](https://www.cvxpy.org/) and `numpy` packages
- Requires the [simulators](https://github.com/rhaksar/simulators) repository 

### Files
- `alp_forestfires_details.pdf`: Notes on implementing the approximate linear program for the forest fires example. 
- `controlEpidemic.py`: Solves an approximate linear program to estimate the Q-function and implements the resulting controller.
- `controlForestFires.py`: Solves an approximate linear program to estimate the value function and implements the resulting controller. Also implements a method from literature. 
