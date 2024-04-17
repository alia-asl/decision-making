from pymonntorch import *
from neuralBehaviors import SetdtBehavior, LIFBehavior, InputBehavior
from synapseBehaviors import DeltaBehavior, ConductanceBehavior
from inputs import step_input
from typing import Literal

def exc_and_inh(con_mode:Literal['full', 'fix_prob', 'fix_count']='full', w_mean_exc=100, w_mean_inh=200, synBehavior=DeltaBehavior):
    lif_params1 = {'tau_m': 10, 'a': -0.5, 'tau_w':  10, 'b':  20, 'R': 3, 'variation': 0.2}
    lif_params2 = {'tau_m': 10, 'a': -0.5, 'tau_w':  10, 'b':  20, 'R': 3, 'variation': 0.2}

    net = Network(behavior={1: SetdtBehavior()}, settings={'synapse_mode': 'SxD'})
    ng_exc = NeuronGroup(800, behavior={
            1: LIFBehavior(**lif_params1),
            2: InputBehavior(step_input, **{'interval0': 20, 'interval1': 5000, 'amp': 10}),
            9: Recorder(variables=['inp', 'voltage']),
            10: EventRecorder(variables=['spike']),
        }, net=net, tag='pop_exc')

    ng_inh = NeuronGroup(200, behavior={
            1: LIFBehavior(**lif_params2),
            2: InputBehavior(step_input, **{'interval0': 20, 'interval1': 5000, 'amp': 10}),
            9: Recorder(variables=['inp', 'voltage']),
            10: EventRecorder(variables=['spike']),
        }, net=net, tag='pop_inh')

    syn_exc_inh = SynapseGroup(ng_exc, ng_inh, net, 'exc', {3:synBehavior(con_mode=con_mode, w_mean=w_mean_exc, rescale=True)})
    syn_inh_exc = SynapseGroup(ng_inh, ng_exc, net, 'inh', {3:synBehavior(con_mode=con_mode, w_mean=w_mean_inh, rescale=True)})

    net.initialize(info=False)
    net.simulate_iterations(200, measure_block_time=False)

    return {'exc': {'vol': ng_exc['voltage', 0], 'inp': ng_exc['inp', 0], 'spike': ng_exc['spike', 0]},
            'inh': {'vol': ng_inh['voltage', 0], 'inp': ng_inh['inp', 0], 'spike': ng_inh['spike', 0]},
            }

def decision_making(
        options:int=2, con_mode:Literal['full', 'fix_prob', 'fix_count']='full', 
        w_mean_exc=100, w_mean_inh=200, neurons_var=0.2, input_var=1, synBehavior=DeltaBehavior,
        amp=10):
    lif_params1 = {'adaptive': True, 'a': 0.5, 'b': 20, 'tau_m': 10, 'tau_w':  10, 'R': 3, 'variation': neurons_var}
    lif_params2 = {'adaptive': True, 'tau_m': 10, 'tau_w':  10, 'R': 3, 'variation': neurons_var}

    net = Network(behavior={1: SetdtBehavior()}, settings={'synapse_mode': 'SxD'})
    options_ngs = []
    for op in range(options):
        options_ngs.append(
        NeuronGroup(800, behavior={
                1: LIFBehavior(**lif_params1),
                2: InputBehavior(input_var, func=step_input, **{'interval0': 200, 'interval1': 5000, 'amp': amp}),
                9: Recorder(variables=['inp', 'voltage']),
                10: EventRecorder(variables=['spike']),
            }, net=net, tag=f'pop_{op}'))
        

    ng_inh = NeuronGroup(200, behavior={
            1: LIFBehavior(**lif_params2),
            2: InputBehavior(input_var, func=step_input, **{'interval0': 20, 'interval1': 5000, 'amp': 0}),
            9: Recorder(variables=['inp', 'voltage']),
            10: EventRecorder(variables=['spike']),
        }, net=net, tag='pop_inh')

    options_syns = []
    for op in range(options):
        options_syns.append(
            SynapseGroup(options_ngs[op], ng_inh, net, 'exc', {3:synBehavior(con_mode=con_mode, w_mean=w_mean_exc, rescale=True)})
        )
        options_syns.append(
            SynapseGroup(ng_inh, options_ngs[op], net, 'inh', {3:synBehavior(con_mode=con_mode, w_mean=w_mean_inh, rescale=True)})
        )
        

    
    net.initialize(info=False)
    net.simulate_iterations(700, measure_block_time=False)
    return {xx: [ng_inh[xx, 0]] + [options_ngs[i][xx, 0] 
                                         for i in range(options)]
                                         for xx in ['inp', 'voltage', 'spike']}
