import recirq.time_crystals
import cirq
import numpy as np

def test_DTCTask():
    np.random.seed(5)
    qubit_locations = [(3, 9), (3, 8), (3, 7), (4, 7), (4, 8), (5, 8), (5, 7), (5, 6), (6, 6), (6, 5), (7, 5), (8, 5),
                          (8, 4), (8, 3), (7, 3), (6, 3)]

    qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
    num_qubits = len(qubits)
    g = 0.94
    instances = 36
    initial_state = np.random.choice(2, num_qubits)
    local_fields = np.random.uniform(-1.0, 1.0, (instances, num_qubits))
    thetas = np.zeros((instances, num_qubits - 1))
    zetas = np.zeros((instances, num_qubits - 1))
    chis = np.zeros((instances, num_qubits - 1))
    gammas = -np.random.uniform(0.5*np.pi, 1.5*np.pi, (instances, num_qubits - 1))
    phis = -2*gammas
    args = ['qubits', 'g', 'initial_state', 'local_fields', 'thetas', 'zetas', 'chis', 'gammas', 'phis']
    default_resolvers = recirq.time_crystals.DTCTask().param_resolvers()
    for arg in args:
        kwargs = {}
        for name in args: 
            kwargs[name] = None if name is arg else locals()[name]
        dtctask = recirq.time_crystals.DTCTask(disorder_instances=instances, **kwargs)
        param_resolvers = dtctask.param_resolvers()
