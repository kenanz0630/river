# Ride-hail vehicle routing (RIVER) problem

This repository contains the codes and input data for solving the RIVER problem proposed by Zhang and Nie (2021)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3974957

Here is an example of running the codes for street-hailing service on a simple network:

```ruby
from river import *
import numpy as np

# initiate network
net = 'inputs/net_simp.csv'
net_param = 'inputs/net_simp_param.csv'
mod = Market(net, net_param, 'shail')

% initiate input variables
N = len(mod.network.nodes)
T = 4

fleet = 200
demand = 100 * np.ones((T, N))
alpha = 1./(N-1) * (np.ones((T, N, N) - np.eye(N).reshape(1, N, N))  # equally go to other zones
fare = mod.network.tau - np.eye(N)
gamma = 1.

# solve equilibrium
ls_gap, __ = mod.equilibrium(fleet, demand, alpha, fare)
```
