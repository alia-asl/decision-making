from pymonntorch import Behavior, SynapseGroup
import torch
import random
from typing import Literal

class DeltaBehavior(Behavior):
    DEFAULTS = {'density': {'fix_prob': 0.1, 'fix_count': 10}}
    def __init__(self, con_mode:Literal['full', 'fix_prob', 'fix_count']='full', density=None, w_mean=50, w_mu=5, rescale:bool=False, **kwargs):
        """
        # Parameters
        -----
        `con_mode`: `full`, `fix_prob`, or `fix_count`\\
        The connection type between the two Neuron Groups
        `density`: float in range (0, 1)\\
        The synapses density\\
        Value depends on `con_mode`:\\
        If =`full`, ignored\\
        If =`fix_prob`, it must be float (the prob)\\
        If =`fix_count`, it must be int (the number of pre-synaptic neurons)
        `rescale`: bool\\
        whether to rescale synaptic weights or not
        """
        if density == None:
            if con_mode == 'fix_prob':
                density = self.DEFAULTS['density']['fix_prob']
            elif con_mode == 'fix_count': 
                density = self.DEFAULTS['density']['fix_count']
            
        super().__init__(con_mode=con_mode, density=density, w_mean=w_mean, w_mu=w_mu, rescale=rescale, **kwargs)
    def initialize(self, syn:SynapseGroup):
        self.init_W(syn)
        
    def init_W(self, syn:SynapseGroup):
        con_mode = self.parameter('con_mode', 'full')
        if con_mode in ['full', 'fix_prob', 'fix_count']:
            getattr(self, "init_" + con_mode)(syn=syn)
        else:
            raise ValueError(f'The connection mode {con_mode} is not defined')
        
    def init_full(self, syn:SynapseGroup):
        rescale = self.parameter('rescale', None)
        density = self.parameter('density', None) # dummy call, because of the warning
        w_mean = self.parameter('w_mean', None)
        w_mu   = self.parameter('w_mu', None)
        N = syn.src.size
        if rescale:
            syn.W = syn.matrix(f'normal(mean={w_mean / N}, std={w_mu / N})')
        else:
            syn.W = syn.matrix(f'normal(mean={w_mean}, std={w_mu})')
        
    
    def init_fix_prob(self, syn:SynapseGroup):
        rescale = self.parameter('rescale', None)
        density = self.parameter('density', None)
        if density > 1 or density < 0:
            raise ValueError("If `con_mode` is `fix_prob`, then it must be a probability")
        density  = self.parameter('density', None)
        w_mean = self.parameter('w_mean', None)
        w_mu   = self.parameter('w_mu', None)
        N = syn.src.size
        is_connected = syn.matrix('uniform') <= density
        syn.W = syn.matrix(0)
        if rescale:
            syn.W[is_connected] = syn.matrix(f'normal(mean={w_mean / (N * density)}, std={w_mu / N})')[is_connected]
        else:
            syn.W[is_connected] = syn.matrix(f'normal(mean={w_mean}, std={w_mu})')[is_connected]
        
        

    
    def init_fix_count(self, syn:SynapseGroup):
        rescale = self.parameter('rescale', None)
        density = self.parameter('density', None)
        density = int(density)
        w_mean = self.parameter('w_mean', None)
        w_mu   = self.parameter('w_mu', None)
        if density > syn.src.size:
            raise ValueError("If `con_mode` is `fix_count`, it must be less than the source neurons size")
        syn.W = syn.matrix(0)
        for i in range(syn.dst.size):
            pre_neurons = random.sample(range(syn.src.size), density)
            if rescale:
                syn.W[pre_neurons, i] = torch.normal(w_mean/density, w_mu/density, (len(pre_neurons), ))
            else:
                syn.W[pre_neurons, i] = torch.normal(w_mean, w_mu, (len(pre_neurons), ))

        
    def forward(self, syn:SynapseGroup):
        spikes = syn.src.spike.float()
        output = spikes @ syn.W
        if 'exc' in syn.tags:
            syn.dst.inp += output
        if 'inh' in syn.tags:
            syn.dst.inp -= 0.5 * output
            
class ConductanceBehavior(DeltaBehavior):
    DEFAULTS = {'density': {'fix_prob': 0.1, 'fix_count': 10}, 'g1': 5}
    def __init__(self, 
    con_mode:Literal['full', 'fix_prob', 'fix_count']='full', 
    density=None, 
    w_mean=50, w_mu=5, rescale=False,
    g0=0.0, g1 = 1, tau=10):
        """
        # Parameters
        ----
        `g0`: float\\
            the initial value of g
        `g1`: float\\
            It is used to calculate: Delta G = `g0` + `g1` * exp(-t/ `tau`)\\
            
        `tau`: float\\
            the time constant of conductance decay
        `density`: float in range (0, 1)\\
        the synapses density
        """
        if density == None:
            if con_mode == 'fix_prob':
                density = self.DEFAULTS['density']['fix_prob']
            elif con_mode == 'fix_count': 
                density = self.DEFAULTS['density']['fix_count']
        
        super().__init__(con_mode=con_mode, density=density, w_mean=w_mean, w_mu=w_mu, rescale=rescale, g0=g0, g1=g1, tau=tau)
    
    def initialize(self, syn:SynapseGroup):
        self.init_W(syn)
        # print("Synaptic weights:")
        # print(syn.W)
        self.g0 = self.parameter('g0', None)
        self.g1 = self.parameter('g1', None)
        syn.g = syn.src.vector(self.g0)
        self.last_spike_t = syn.src.vector(1000)
        self.alpha = self.parameter('alpha', None)
        self.tau = self.parameter('tau', None)
    
    def forward(self, syn:SynapseGroup):
        spikes = syn.src.spike
        self.last_spike_t[spikes.bool()] = 0
        spikes = spikes.float()
        syn.g = self.g0 + self.g1 * torch.exp(-self.last_spike_t/self.tau)
        output = (spikes * syn.g) @ syn.W
        if 'exc' in syn.tags:
            syn.dst.inp += output
        if 'inh' in syn.tags:
            syn.dst.inp -= output
        self.last_spike_t += 1
        
