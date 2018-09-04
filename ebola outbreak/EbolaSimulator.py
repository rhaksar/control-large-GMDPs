import copy
import numpy as np

class EbolaSimulator(object):
  """
  A simulation of the 2013 Ebola outbreak in West Africa, involving
  Sierra Leone, Liberia, and Guinea. 
  """
  def __init__(self, graph, init, parameters, rng=None):
    """
    Initializes a simulation object.
    """
    self.alpha = parameters['alpha']
    self.beta = parameters['beta']
    #self.dbeta = parameters['dbeta']
    #self.eta = parameters['eta']

    self.graph = graph

    self.states = {}
    for k in self.graph.keys():
      self.states[k] = {}
      if k in init.keys():
        self.states[k]['status'] = 'infected'
        self.states[k]['value'] = init[k]
      else:
        self.states[k]['status'] = 'susceptible'
        self.states[k]['value'] = 0    
    
    if rng is not None:
      np.random.seed(rng)

    self.iter = 1
    self.end = False

  def step(self, action):
    """
    Updates a simulation object.
    """

    # check if simulation has already terminated
    if self.end:
      print('process has terminated')

    terminate = True

    update = {}
    for k in self.graph.keys():
      update[k] = {}

      # check if location will contain ebola
      if self.states[k]['status'] == 'susceptible':
        Ni = 0
        for edge in self.graph[k]['edges']:
          if self.states[edge]['status'] == 'infected':
            Ni += 1

        #prob = 1 - (self.alpha**Ni)
        prob = self.alpha*Ni
        if np.random.rand() <= prob:
          update[k]['status'] = 'infected'
          update[k]['value'] = 1
          terminate = False
        else:
          update[k]['status'] = 'susceptible'
          update[k]['value'] = 0

      # check if location stabilizes 
      elif self.states[k]['status'] == 'infected':
        if k in action:
          #Ni = 0
          #for edge in self.graph[k]['edges']:
          #  if self.states[edge]['status'] == 'infected':
          #    Ni += 1
          #prob = self.beta - self.eta*Ni

          prob = self.beta
          if np.random.rand() <= prob:
            update[k]['status'] = 'removed'
            update[k]['value'] = copy.copy(self.states[k]['value'])
          else:
            update[k]['status'] = 'infected'
            update[k]['value'] = copy.copy(self.states[k]['value']) + 1
            terminate = False

        else:
          update[k]['status'] = 'infected'
          update[k]['value'] = copy.copy(self.states[k]['value']) + 1
          terminate = False

      # absorbing states do not change
      else:
        update[k] = copy.copy(self.states[k])

    # update states
    for k in self.graph.keys():
      self.states[k] = copy.copy(update[k])

    self.iter += 1
    # if self.iter == 122:
    if terminate:
      self.end = True
