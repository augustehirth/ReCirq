# -*- coding: utf-8 -*-

from typing import Sequence, Tuple, List

import cirq
import numpy as np
import sympy as sp

import shutil

def symbolic_dtc_circuit_list(
        qubits: Sequence[cirq.Qid],
        cycles: int
        ) -> List[cirq.Circuit]:

    """ Create a list of symbolically parameterized dtc circuits, with increasing cycles
    Args: 
    - qubits: an ordered sequence of the available qubits, which are connected in a chain
    - cycles: the maximum number of cycles, and the total number of resulting circuits
    Returns: 
    - list of circuits with `0, 1, 2, ... cycles` many cycles
    """
     
    num_qubits = len(qubits)

    # Symbol for g
    g_value = sp.Symbol('g')

    # Symbols for random variance and initial state, one per qubit
    local_fields = sp.symbols('local_field:' + str(num_qubits))
    initial_state = sp.symbols('initial_state:' + str(num_qubits))

    # Symbols used for PhasedFsimGate, one for every qubit pair in the chain
    thetas = sp.symbols('theta:' + str(num_qubits - 1))
    zetas = sp.symbols('zeta:' + str(num_qubits - 1))
    chis = sp.symbols('chi:' + str(num_qubits - 1))
    gammas = sp.symbols('gamma:' + str(num_qubits - 1))
    phis = sp.symbols('phi:' + str(num_qubits - 1))

    # Initial moment of Y gates, conditioned on initial state
    initial_operations = cirq.Moment([cirq.Y(qubit) ** initial_state[index] for index, qubit in enumerate(qubits)])

    # First component of U cycle, a moment of ZX gates. 
    sequence_operations = []
    for index, qubit in enumerate(qubits):
        sequence_operations.append(cirq.PhasedXZGate(
                x_exponent=g_value, axis_phase_exponent=0.0,
                z_exponent=local_fields[index])(qubit))

    # Begin U cycle
    u_cycle = [cirq.Moment(sequence_operations)]

    # Second and third components of U cycle, a chain of 2-qubit PhasedFSim gates
    #   The first component is all the 2-qubit PhasedFSim gates starting on even qubits
    #   The second component is the 2-qubit gates starting on odd qubits
    operation_list, other_operation_list = [],[]
    previous_qubit, previous_index = None, None
    for index, qubit in enumerate(qubits):
        if previous_qubit is None:
            previous_qubit, previous_index = qubit, index
            continue

        # Add an fsim gate 
        coupling_gate = cirq.ops.PhasedFSimGate(
            theta=thetas[previous_index], 
            zeta=zetas[previous_index],
            chi=chis[previous_index],
            gamma=gammas[previous_index],
            phi=phis[previous_index]
        )
        operation_list.append(coupling_gate.on(previous_qubit, qubit))
        
        # Swap the operation lists we're adding to, to avoid two-qubit gate overlap
        previous_qubit, previous_index = qubit, index
        operation_list, other_operation_list = other_operation_list, operation_list

    # Add the two components into the U cycle
    u_cycle.append(cirq.Moment(operation_list))
    u_cycle.append(cirq.Moment(other_operation_list))

    # Prepare a list of circuits, with n=0,1,2,3.... cycles
    circuit_list = []
    total_circuit = cirq.Circuit(initial_operations)
    circuit_list.append(total_circuit.copy())
    for c in range(cycles):
        for m in u_cycle:
            total_circuit.append(m)
        circuit_list.append(total_circuit.copy())

    return circuit_list

def dtc_param_resolver_sweep(gs: Sequence[float] = None,
                       initial_states: Sequence[Sequence[int]] = None, 
                       local_fields: Sequence[Sequence[float]] = None, 
                       thetas: Sequence[Sequence[float]] = None, 
                       zetas: Sequence[Sequence[float]] = None, 
                       chis: Sequence[Sequence[float]] = None, 
                       gammas: Sequence[Sequence[float]] = None, 
                       phis: Sequence[Sequence[float]] = None,
                       ) -> cirq.Sweepable:
    """ Create a `cirq.Sweepable` sequence of `cirq.ParamResolver`s, for the parameters of dtc circuits
    Args: 
    - initial_states: list of initial states (list of [0,1] ints) for the circuit
    - local_fields: list of local fields (list of floats) that model random fluctuations
    - thetas: list of list of thetas for each two-qubit fsim gate in the chain
    - zetas: list of list of zetas for each two-qubit fsim gate in the chain
    - chis: list of list of chis for each two-qubit fsim gate in the chain
    - gammas: list of list of gammas for each two-qubit fsim gate in the chain
    - phis: list of list of phis for each two-qubit fsim gate in the chain
    Returns: 
    - A `cirq.Sweepable` which zips together param resolvers for the options for each individual parameter, into a sequence of parameter sweeps. 
    """

    components = []

    # gs are the only parameter that is not qubit-dependent
    if gs is not None and len(gs):
        components.append(cirq.Points('g', gs))

    # The remaining parameters have a separate symbol for each qubit
    labels = ['initial_state', 'local_field', 'theta', 'zeta', 'chi', 'gamma', 'phi']
    parameters = [initial_states, local_fields, thetas, zetas, chis, gammas, phis]

    for label, parameter in zip(labels, parameters): 
        sweep = []
        # if the parameter is not supplied, don't add it to the param resolvers
        if parameter is None: continue

        # for each set of options for the parameter
        for options in parameter: 
            # create a dictionary/param resolver to match each option in the parameter to each symbol name, for each qubit index
            component = {label + str(qubit_index):option for qubit_index, option in enumerate(options)}
            sweep.append(cirq.ParamResolver(component))

        # include sweep over options (list of param resolvers) as a component
        components.append(cirq.ListSweep(sweep))

    # return zip over all separate list sweeps of each parameter's options    
    return cirq.Zip(*components)

def simulate_dtc_circuit_list(circuit_list: Sequence[cirq.Circuit], param_resolver: cirq.ParamResolver, qubit_order: Sequence[cirq.Qid]) -> np.ndarray: 
    """ Simulate a dtc circuit list for a particular param_resolver
    Args: 
    - circuit_list: a DTC circuit list; each element is a circuit with increasingly many cycles
    - param_resolver: a `cirq.ParamResolver`
    - qubit_order: an ordered sequence of qubits defining their order in a chain
    Returns: 
    - a `np.ndarray` of shape (number of cycles, 2**number of qubits) representing the probability of measuring each bit string, for each circuit in the list
    """

    # prepare simulator
    simulator = cirq.Simulator()

    # record lengths of circuits in list
    circuit_positions = [len(c) - 1 for c in circuit_list]

    # only simulate one circuit, the last one
    circuit = circuit_list[-1]
    
    # use simulate_moment_steps to recover all of the state vectors necessary, while only simulating the circuit list once
    probabilities = []
    for k, step in enumerate(simulator.simulate_moment_steps(circuit=circuit, param_resolver=param_resolver, qubit_order=qubit_order)):
        # add the state vector if the number of moments simulated so far is equal to the length of a circuit
        if k in circuit_positions:
            probabilities.append(np.abs(step.state_vector()) ** 2)

    return np.asarray(probabilities)

def simulate_dtc_circuit_list_sweep(circuit_list: Sequence[cirq.Circuit], param_resolvers: Sequence[cirq.ParamResolver], qubit_order: Sequence[cirq.Qid]) -> List[Tuple[cirq.ParamResolver, np.ndarray]]:
  """ Simulate a dtc circuit list over a sweep of param_resolvers
  Args: 
  - circuit_list: a DTC circuit list; each element is a circuit with increasingly many cycles
  - param_resolvers: a list of `cirq.ParamResolver`s to sweep over
  - qubit_order: an ordered sequence of qubits defining their order in a chain
  Generates, for each param_resolver: 
  - `np.ndarray`s of shape (number of cycles, 2**number of qubits) representing the probability of measuring each bit string, for each circuit in the list
  """

  # iterate over param resolvers and simulate for each
  for param_resolver in param_resolvers:
    probabilities = simulate_dtc_circuit_list(circuit_list, param_resolver, qubit_order)
    yield probabilities

def get_polarizations(probabilities: np.ndarray, num_qubits: int, cycles_axis: int = -2, probabilities_axis: int = -1, initial_states: np.ndarray = None) -> np.ndarray: 
    """get polarizations (likelihood of zero) from matrix of probabilities
    Args: 
    - probabilities: `np.ndarray` of shape (:, cycles, probabilities) representing probability to measure each bit string
    - num_qubits: the number of qubits in the circuit the probabilities were generated from
    - cycles_axis: the axis that represents the dtc cycles (if not in -2 indexed axis)
    - probabilities_axis: the axis that represents the probabilities for each bit string (if not in -1 indexed axis)
    - initial_state: `np.ndarray` of shape (:, qubits) representing the initial state for each dtc circuit list
    Returns: 
    - `np.ndarray` of shape (:, cycles, qubits) that represents each qubit's polarization
    """

    # prepare list of polarizations for each qubit
    polarizations = []
    for qubit_index in range(num_qubits):
        # select all indices in range(2**num_qubits) for which the associated element of the statevector for which qubit_index is zero
        shift_by = num_qubits - qubit_index - 1
        state_vector_indices = [i for i in range(2 ** num_qubits) if not (i >> shift_by) % 2]

        # sum over all amplitudes for qubit states for which qubit_index is zero, and rescale them to [-1,1]
        polarization = 2.0 * np.sum(probabilities.take(indices=state_vector_indices, axis=probabilities_axis), axis=probabilities_axis) - 1.0
        polarizations.append(polarization)

    # turn polarizations list into an array, and move the new, leftmost axis for qubits to probabilities_axis
    polarizations = np.moveaxis(np.asarray(polarizations), 0, probabilities_axis)

    # flip polarizations according to the associated initial_state, if provided
    # this means that the polarization of a qubit is relative to it's initial state
    if initial_states is not None:
        initial_states = 1 - 2.0 * np.expand_dims(initial_states, axis=cycles_axis)
        polarizations = initial_states * polarizations

    return polarizations

def filter_if_latex_not_available(label: str) -> str:
    """ return if latex is available in path, and filter label if it isn't
    Args: 
    - label: a tex-formatted string
    Returns: 
    - label, but conditionally in plaintext (backslashes removed)
    """

    if shutil.which('latex') is not None:
        return label
    else:
        # only currently used for labels with backslashes, otherwise this would be more comprehensive
        return label.replace("\\", "")

