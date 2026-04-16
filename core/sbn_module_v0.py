"""
SBN Generator Module - 6 Orthogonal Binary Constraints

Design space: {0,1}^6 = 64 architectures (all fully independent)

Constraints:
    S: Stratification  - Alternating S/P layers (nonlinear/linear)
    A: Acyclicity      - Strict DAG, no feedback cycles
    R: Regularity      - Equal source-to-sink path lengths
    I: Interleaving    - Cross-block dependencies between bit groups
    H: Homogeneity     - Identical functions within each layer
    L: Locality        - Bounded connection distance

All circuits: 16-bit input, 16-bit output, 16-bit state.
All constraints are CONSTRUCTIVE (built into generation).
Mutations preserve active constraints.

All code in ENGLISH ONLY.
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Callable, Optional, Dict
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

# Module version — increment on every structural change.
# Displayed at notebook import time to confirm the correct file is loaded.
_MODULE_VERSION = "5.10.1"   # v5.10.1: coverage pass préserve backward edges (A=False leak corrigé)

# GPU imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


# =============================================================================
# CORE SBN CLASSES
# =============================================================================

class NodeType(Enum):
    """Node types in SBN computational graph."""
    WIRE       = 1
    LOGIC_GATE = 2


@dataclass
class ComputeNode:
    """A single node in the SBN computational graph."""
    node_id:    str
    node_type:  NodeType
    operation:  str
    inputs:     List[str]
    output_bit: Optional[int] = None


class GenericSBN:
    """
    Synchronous Boolean Network.

    16 inputs, 16 state bits, 16 outputs.
    Can contain cycles unless constraint A is active.
    """

    def __init__(self, compute_graph: List[ComputeNode]):
        self.compute_graph = compute_graph
        self.state         = [0] * 16
        self.nodes         = {n.node_id: n for n in compute_graph}
        self.output_map    = {n.output_bit: n.node_id
                              for n in compute_graph if n.output_bit is not None}

        if len(self.output_map) != 16:
            raise ValueError(f"Expected 16 outputs, got {len(self.output_map)}")

    def step(self, external_inputs: List[int]) -> List[int]:
        """Execute one synchronous step."""
        if len(external_inputs) != 16:
            raise ValueError("Need exactly 16 external inputs")

        node_values = {}
        for i in range(16):
            node_values[f"input_{i}"] = external_inputs[i]
            node_values[f"state_{i}"] = self.state[i]

        # Multi-pass evaluation (resolves dependencies, handles cycles)
        for _ in range(20):
            changed = 0
            for node in self.compute_graph:
                if node.node_id in node_values:
                    continue
                if not all(inp in node_values for inp in node.inputs):
                    continue

                vals = [node_values[inp] for inp in node.inputs]

                if   node.operation == "NOT":      val = 1 - vals[0]
                elif node.operation == "AND":      val = vals[0] & (vals[1] if len(vals) > 1 else vals[0])
                elif node.operation == "OR":       val = vals[0] | (vals[1] if len(vals) > 1 else vals[0])
                elif node.operation == "XOR":      val = vals[0] ^ (vals[1] if len(vals) > 1 else vals[0])
                elif node.operation == "IDENTITY": val = vals[0]
                else:                              val = 0

                node_values[node.node_id] = val
                changed += 1

            if changed == 0:
                break

        self.state = [node_values.get(self.output_map[b], 0) for b in range(16)]
        return self.state

    def reset(self, initial_state: Optional[List[int]] = None):
        """Reset internal state."""
        if initial_state is None:
            self.state = [0] * 16
        else:
            assert len(initial_state) == 16
            self.state = list(initial_state)

    def analyze_structure(self) -> Dict:
        """Return basic structural statistics."""
        ops = {}
        for node in self.compute_graph:
            ops[node.operation] = ops.get(node.operation, 0) + 1
        return {'operations': ops, 'total_nodes': len(self.compute_graph)}


# =============================================================================
# SHARED HARD CONSTRAINTS (always enforced)
# =============================================================================

SHARED = {
    'n_bits':        16,
    'target_gates':  40,
    'gate_variance': 5,
    'depth':         5,
}

OPS_NONLINEAR = ['AND', 'OR']
OPS_LINEAR    = ['XOR', 'NOT']
OPS_ALL       = OPS_NONLINEAR + OPS_LINEAR


def _rand_op(linear_only: bool = False, nonlinear_only: bool = False) -> str:
    if linear_only:    return random.choice(OPS_LINEAR)
    if nonlinear_only: return random.choice(OPS_NONLINEAR)
    return random.choices(OPS_ALL, weights=[0.25, 0.25, 0.35, 0.15])[0]


def _make_inputs(op: str, pool: List[str]) -> List[str]:
    """Always return exactly the right number of inputs for an operation.

    Self-connections (a op a) are forbidden for binary ops: AND(x,x)=x is
    affine and XOR(x,x)=0 is constant, both of which degrade non-linearity.
    """
    if not pool:
        pool = [f"input_{i}" for i in range(16)]  # fallback
    if op in ('NOT', 'IDENTITY'):
        return [random.choice(pool)]
    # Binary ops: pick two distinct sources when the pool is large enough.
    a = random.choice(pool)
    if len(pool) > 1:
        others = [x for x in pool if x != a]
        b = random.choice(others)
    else:
        b = a  # pool has only one element — unavoidable
    return [a, b]


def _make_output_layer(layer: List[str]) -> List[ComputeNode]:
    return [
        ComputeNode(f"output_{b}", NodeType.WIRE, 'IDENTITY',
                    [layer[b % len(layer)]], b)
        for b in range(16)
    ]


# =============================================================================
# CONSTRAINT BUILDERS
# =============================================================================
#
# Architecture:
#   - One unified builder: _build_layered(S, R, I, H, L)
#     Always produces L*_g* nodes so S/H/R/L checkers find something.
#   - _build_generic() for the pure-generic case (no S/R/H/L).
#   - create_sbn() orchestrates builders + constraint enforcements
#     + constraint BREAKERS for every inactive flag.
#
# Isolation guarantee:
#   Active constraint  → enforced by builder / post-processor.
#   Inactive constraint → broken by a dedicated breaker that is
#                         surgical (touches only the targeted property).
# =============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circ_dist(a: int, b: int, n: int = 16) -> int:
    """Circular distance between positions a and b on a ring of size n."""
    d = abs(a - b)
    return min(d, n - d)


def _ring_pos(nid: str, n: int = 16) -> int:
    """
    Extract ring position from a node id.
    Works for: input_i, state_i, L*_g* (returns g index % n).
    """
    if '_g' in nid:
        try:    return int(nid.split('_g')[-1]) % n
        except: return 0
    try:    return int(nid.split('_')[-1]) % n
    except: return 0


# ---------------------------------------------------------------------------
# Builder: unified layered graph
# ---------------------------------------------------------------------------

def _build_layered(
    S: bool = False,   # odd layers nonlinear, even layers linear
    R: bool = False,   # gate at layer k reads only layer k-1
    I: bool = False,   # L1 gates span both blocks
    H: bool = False,   # all gates in same layer share one operation
    L: bool = False,   # connections within circular distance d_max
    d_max: int = 4,
) -> GenericSBN:
    """
    Build a layered SBN (L*_g* node ids) enforcing exactly the requested
    active constraints.  Does NOT enforce A — that is handled by create_sbn.

    Key design decisions
    --------------------
    - Always builds DEPTH layers so S/H/R/L checkers find L-prefixed nodes.
    - Predecessor pool per layer:
        R=True  → strict layer k-1 only
        R=False → full history (all earlier layers + inputs/states), so skip
                  connections appear naturally most of the time.
    - Locality (L) restricts the pool to positions within d_max on the ring.
    - Interleaving (I) forces L1 gates to mix blocks B0 and B1.
    - Stratification (S) alternates nonlinear/linear ops by layer parity.
    - Homogeneity (H) fixes one op per layer (chosen at layer start).
    """
    n               = SHARED['n_bits']
    depth           = SHARED['depth']
    # ≥ n_bits so the last layer has enough distinct gates for all 16 output
    # bits.  With only 8 gates/layer the output wire b % 8 duplicates bits
    # 0-7 onto bits 8-15, making the S-box rank-8 and DP_max = 1.0 always.
    gates_per_layer = max(n, SHARED['target_gates'] // depth)  # ≥ n=16 always
    b0              = [f"input_{i}" for i in range(0,    n // 2)]
    b1              = [f"input_{i}" for i in range(n // 2, n)]
    # state_* excluded: state_i=0 in S-box evaluation mode, so any gate
    # reading state_i outputs a constant, killing non-linearity.
    inputs_states   = [f"input_{i}" for i in range(n)]
    graph           = []
    history: Dict[int, List[str]] = {}   # layer_idx → list of node_ids

    for layer_idx in range(1, depth + 1):
        is_sp = (layer_idx % 2 == 1)

        # Choose operation for this layer
        if H:
            if S:
                layer_op = _rand_op(nonlinear_only=is_sp, linear_only=not is_sp)
            elif I and layer_idx == 1:
                layer_op = random.choice([op for op in OPS_ALL if op not in ('NOT', 'IDENTITY')])
            else:
                layer_op = _rand_op()
        else:
            layer_op = None   # chosen per gate below

        # Build predecessor pool for this layer
        if layer_idx == 1:
            # L1 reads external inputs and state bits
            base_pool = inputs_states
        elif R:
            # R=True: strict layer-by-layer (L_k reads only L_{k-1})
            base_pool = history[layer_idx - 1]
        else:
            # R=False: skip connections allowed from any prior layer or from
            # the original inputs.  Including input_* ensures both input blocks
            # (B0=input_0..7 and B1=input_8..15) are always reachable even after
            # _break_interleaving restricts L1 to a single block.
            # state_* remain excluded (state_i=0 → constant gate output).
            base_pool = list(inputs_states)   # always include input_*
            for prev in range(1, layer_idx):
                base_pool.extend(history[prev])

        layer_nodes = []
        for g in range(gates_per_layer):
            nid = f"L{layer_idx}_g{g}"

            # Select operation
            if H:
                op = layer_op
            elif S:
                op = _rand_op(nonlinear_only=is_sp, linear_only=not is_sp)
            else:
                op = _rand_op()

            # Build local pool when L is active
            if L:
                pos = g % n
                def _node_ring_pos(nid):
                    try:
                        return int(nid.split('_g')[-1]) % n if '_g' in nid else int(nid.split('_')[-1]) % n
                    except ValueError:
                        return 0
                local_pool = [c for c in base_pool
                              if _circ_dist(pos, _node_ring_pos(c)) <= d_max]
                pool = local_pool if local_pool else base_pool
            else:
                pool = base_pool

            # Enforce Interleaving at L1
            if I and layer_idx == 1:
                if op in ('NOT', 'IDENTITY'):
                    # Unary: pick from b0, respecting locality when L is active
                    if L:
                        _pos = g % n
                        _lp0 = [c for c in b0 if _circ_dist(_pos, int(c.split('_')[-1]) % n) <= d_max]
                        ins = [random.choice(_lp0 if _lp0 else b0)]
                    else:
                        ins = [random.choice(b0)]
                else:
                    # Pick one input from each block, respecting locality when L is active
                    def _local_or_global(block, _pos=g % n):  # capture pos at definition time
                        if L:
                            lp = [c for c in block
                                  if _circ_dist(_pos, int(c.split('_')[-1]) % n) <= d_max]
                            return random.choice(lp) if lp else min(block, key=lambda c: _circ_dist(_pos, int(c.split('_')[-1]) % n))
                        return random.choice(block)
                    ins = [_local_or_global(b0), _local_or_global(b1)]
            else:
                ins = _make_inputs(op, pool)

            graph.append(ComputeNode(nid, NodeType.LOGIC_GATE, op, ins, None))
            layer_nodes.append(nid)

        history[layer_idx] = layer_nodes

    graph += _make_output_layer(history[depth])
    return GenericSBN(graph)


# ---------------------------------------------------------------------------
# Builder: generic flat graph (no layers)
# ---------------------------------------------------------------------------

def _build_generic() -> GenericSBN:
    """
    Generic layered graph: delegates to _build_layered() with no active
    constraints.  This guarantees a structured DAG (depth layers, proper
    predecessor pools) while the absence of S/R/H/L/I ensures the breakers
    will destroy each constraint's signature afterwards.

    Replacing the old flat gate_* approach eliminates two failure modes:
      - affine-only outputs (old flat graph had no guaranteed NL path)
      - _enforce_acyclicity rewiring all gates to input_i (shallow depth)
    """
    return _build_layered()


# ---------------------------------------------------------------------------
# Acyclicity enforcer / breaker
# ---------------------------------------------------------------------------

def _enforce_acyclicity(graph: List[ComputeNode]) -> List[ComputeNode]:
    """
    A=True: rewrite any forward-reference edge to make a strict DAG.
    Preserves all other structural properties (layer structure, ops).
    """
    all_ids   = [f"input_{i}" for i in range(16)]  # no state_*
    new_graph = copy.deepcopy(graph)
    available = list(all_ids)
    for node in new_graph:
        if node.output_bit is not None:
            continue
        node.inputs = [
            inp if inp in available else random.choice(available)
            for inp in node.inputs
        ]
        available.append(node.node_id)
    return new_graph


def _introduce_cycle(graph: List[ComputeNode],
                     R: bool = False, L: bool = False) -> List[ComputeNode]:
    """
    A=False: introduce exactly one genuine backward edge (a true cycle).

    A 'backward edge' means gate X has an input pointing to gate Y where Y
    appears AFTER X in the compute list (so val(Y) is not yet known when X
    is evaluated → the multi-pass evaluator uses the previous-step value,
    i.e. a genuine feedback loop).

    R=True  → the cycle must remain R-compatible:
               gate X in layer k gets an input from a gate Y in layer k-1,
               where Y appears after X in the list.
               We achieve this by MOVING one L(k-1) gate to a position after
               one L(k) gate, then wiring X → Y.  This preserves the layer
               membership (R property) while creating a real backward edge.
    L=True  → Y must be within d_max of X on the ring.
    Falls back to any backward edge if the surgical attempt fails.
    """
    graph   = copy.deepcopy(graph)
    n_ring  = SHARED['n_bits']
    d_max   = 4
    compute = [nd for nd in graph if nd.output_bit is None]
    ncount  = len(compute)

    layer_members: Dict[int, List[str]] = {}
    node_layer_map: Dict[str, int] = {}
    for nd in compute:
        if nd.node_id.startswith('L'):
            lyr = int(nd.node_id.split('_')[0][1:])
            layer_members.setdefault(lyr, []).append(nd.node_id)
            node_layer_map[nd.node_id] = lyr

    if R and len(layer_members) >= 2:
        # Strategy: pick a gate X in layer k>=2, pick a gate Y in layer k-1.
        # Move Y to a position just after X in the compute list.
        # Wire one of X's inputs to Y (now a backward ref since Y comes after X).
        max_lyr = max(layer_members.keys())
        for _ in range(60):
            k = random.randint(2, max_lyr)
            gates_k   = layer_members.get(k,   [])
            gates_km1 = layer_members.get(k-1, [])
            if not gates_k or not gates_km1:
                continue

            x_id = random.choice(gates_k)
            y_id = random.choice(gates_km1)

            if L:
                try:    xpos = int(x_id.split('_g')[1]) % n_ring
                except: xpos = 0
                if _circ_dist(xpos, _ring_pos(y_id, n_ring)) > d_max:
                    # Find a local y
                    local_ym1 = [y for y in gates_km1
                                 if _circ_dist(xpos, _ring_pos(y, n_ring)) <= d_max]
                    if not local_ym1:
                        continue
                    y_id = random.choice(local_ym1)

            x_nd = next(nd for nd in compute if nd.node_id == x_id)
            y_nd = next(nd for nd in compute if nd.node_id == y_id)

            if not x_nd.inputs:
                continue

            # Move y_nd to just after x_nd in the graph list
            x_pos_in_graph = next(i for i, nd in enumerate(graph) if nd.node_id == x_id)
            y_pos_in_graph = next(i for i, nd in enumerate(graph) if nd.node_id == y_id)

            if y_pos_in_graph == x_pos_in_graph + 1:
                pass  # already right after
            else:
                # Remove y from its current position and insert after x
                y_node = graph.pop(y_pos_in_graph)
                # x may have shifted if y was before x
                x_pos_in_graph = next(i for i, nd in enumerate(graph) if nd.node_id == x_id)
                graph.insert(x_pos_in_graph + 1, y_node)

            # Wire one input of x to y (now backward)
            x_nd.inputs[random.randrange(len(x_nd.inputs))] = y_id
            return graph

    # Non-R case or fallback: any backward edge
    id_to_idx = {nd.node_id: i for i, nd in enumerate(compute)}
    for attempt in range(60):
        ti = random.randrange(0, ncount - 1)
        nd = compute[ti]
        if not nd.inputs:
            continue
        # Source candidates: gates appearing after ti
        srcs = [compute[si].node_id for si in range(ti + 1, ncount)]
        if L and nd.node_id.startswith('L'):
            try:    pos = int(nd.node_id.split('_g')[1]) % n_ring
            except: pos = 0
            srcs = [c for c in srcs if _circ_dist(pos, _ring_pos(c, n_ring)) <= d_max]
        if srcs:
            nd.inputs[random.randrange(len(nd.inputs))] = random.choice(srcs)
            return graph

    return graph


def _break_regularity(graph: List[ComputeNode],
                      A: bool = False, L: bool = False, d_max: int = 4) -> List[ComputeNode]:
    """
    R=False: one skip connection (layer k reads from layer k-2 or inputs/states).
    A=True → skip source strictly earlier in eval order (DAG preserved).
    L=True → skip source within d_max of the target gate (locality preserved).
    Only touches wires → preserves S, H.
    """
    graph = copy.deepcopy(graph)
    n     = SHARED['n_bits']
    layer_members: Dict[int, List[str]] = {}
    for nd in graph:
        if nd.output_bit is not None or not nd.node_id.startswith('L'):
            continue
        lyr = int(nd.node_id.split('_')[0][1:])
        layer_members.setdefault(lyr, []).append(nd.node_id)

    max_layer     = max(layer_members.keys()) if layer_members else 0
    inputs_states = [f"input_{i}" for i in range(n)]  # no state_*
    inputs_states_set = set(inputs_states)
    id_to_idx     = {nd.node_id: i for i, nd in enumerate(graph)}

    for lyr in range(2, max_layer + 1):
        skip_pool = (inputs_states if lyr == 2
                     else layer_members.get(lyr - 2, inputs_states))
        candidates = [nd for nd in graph
                      if nd.output_bit is None
                      and nd.node_id.startswith(f'L{lyr}_')]
        if not candidates or not skip_pool:
            continue
        nd = random.choice(candidates)
        if not nd.inputs:
            continue
        pool = list(skip_pool)
        if A:
            nd_idx = id_to_idx.get(nd.node_id, len(graph))
            pool   = [s for s in pool if id_to_idx.get(s, -1) < nd_idx]
        if L:
            try:    pos = int(nd.node_id.split('_g')[1]) % n
            except: pos = 0
            local = [s for s in pool if _circ_dist(pos, _ring_pos(s, n)) <= d_max]
            pool  = local if local else pool
        if not pool:
            continue
        # Only overwrite block-refs or forward gate-refs — never backward edges
        nd_idx = id_to_idx.get(nd.node_id, len(graph))
        safe_slots = [
            i for i, inp in enumerate(nd.inputs)
            if inp in inputs_states_set
            or (inp in id_to_idx and id_to_idx[inp] < nd_idx)
        ]
        if safe_slots:
            nd.inputs[random.choice(safe_slots)] = random.choice(pool)
            return graph
    return graph


def _break_interleaving(graph: List[ComputeNode],
                        L: bool = False, R: bool = False,
                        d_max: int = 4) -> List[ComputeNode]:
    """
    I=False: ensure no L1 gate reads inputs from *both* blocks simultaneously,
    while keeping all 16 inputs reachable.

    Two strategies depending on R:

    R=False (skip connections exist in L2+):
        sole_block mode — all L1 block-valued inputs → one chosen block.
        L2+ skip connections (pool includes input_*) compensate for the
        missing block.  Coverage pass in create_sbn fixes any residual gaps.

    R=True (L2 reads only L1; no skip to inputs):
        split mode — L1 gates are divided into two equal halves; the first
        half reads B0 only, the second half reads B1 only.  No single gate
        crosses blocks (I=False satisfied), yet both blocks feed L2+.

    Preservation rules:
    - Gate refs (backward/forward gate-to-gate edges) → kept as-is (preserves A).
    - L=True  → replacement must be within d_max (nearest fallback if needed).
    """
    n          = SHARED['n_bits']
    b0         = [f"input_{i}" for i in range(0,    n // 2)]
    b1         = [f"input_{i}" for i in range(n // 2, n)]
    all_io_set = set(b0 + b1)  # no state_*

    l1_gates = [nd for nd in graph
                if nd.output_bit is None and nd.node_id.startswith('L1_g')]

    if R:
        # Split mode: half the gates → B0, other half → B1.
        # Shuffle so the split is positionally varied.
        l1_copy = list(l1_gates)
        random.shuffle(l1_copy)
        mid = len(l1_copy) // 2
        block_for_gate = {}
        for i, nd in enumerate(l1_copy):
            block_for_gate[nd.node_id] = b0 if i < mid else b1
    elif L:
        # Sole-block mode with locality (L=True):
        # Choose per gate the block that has the most members within d_max.
        # This avoids structural dead wires when one block is geometrically
        # inaccessible from a given position (e.g. B0 is too far from pos=12).
        # Global I=False is still satisfied: each L1 gate reads only one block.
        # To avoid accidental restoration of interleaving, we bias toward one
        # globally chosen block when both are equally reachable.
        global_sole = random.choice([b0, b1])
        block_for_gate = {}
        for nd in l1_gates:
            try:    pos = int(nd.node_id.split('_g')[1]) % n
            except: pos = 0
            b0_local = [c for c in b0 if _circ_dist(pos, _ring_pos(c, n)) <= d_max]
            b1_local = [c for c in b1 if _circ_dist(pos, _ring_pos(c, n)) <= d_max]
            if len(b0_local) > len(b1_local):
                block_for_gate[nd.node_id] = b0
            elif len(b1_local) > len(b0_local):
                block_for_gate[nd.node_id] = b1
            else:
                block_for_gate[nd.node_id] = global_sole
    else:
        # Sole-block mode (standard): all L1 gates → one chosen block.
        # L2+ skip connections (pool includes input_*) cover the other block.
        sole = random.choice([b0, b1])
        block_for_gate = {nd.node_id: sole for nd in l1_gates}

    for nd in graph:
        if nd.output_bit is None and nd.node_id.startswith('L1_g'):
            blk = block_for_gate.get(nd.node_id, b0)
            try:    pos = int(nd.node_id.split('_g')[1]) % n
            except: pos = 0
            new_inputs = []
            for inp in nd.inputs:
                if inp not in all_io_set:
                    new_inputs.append(inp)           # gate ref — keep
                elif L:
                    local = [c for c in blk
                             if _circ_dist(pos, _ring_pos(c, n)) <= d_max]
                    if local:
                        new_inputs.append(random.choice(local))
                    else:
                        nearest = min(blk,
                                      key=lambda c: _circ_dist(pos, _ring_pos(c, n)))
                        new_inputs.append(nearest)
                else:
                    new_inputs.append(random.choice(blk))
            nd.inputs = new_inputs
    return graph


def _break_ops(graph: List[ComputeNode],
               S: bool = False,
               H: bool = False,
               I: bool = False) -> List[ComputeNode]:
    """
    Combined operation breaker for S=False and/or H=False.

    Four cases based on which constraints are INACTIVE (need breaking):

    S=False, H=False : At least one layer must have ≥2 ops with mixed parity
                       (wrong-parity op + another op). This violates both S and H.

    S=False, H=True  : At least one layer must be entirely wrong-parity.
                       (homogeneous but wrong parity → NOT S, keeps H).

    S=True,  H=False : At least one layer must have ≥2 different ops
                       (all correct parity, but heterogeneous → NOT H, keeps S).

    S=True,  H=True  : Neither called; noop.

    I=True           : L1 ops kept binary (NOT/IDENTITY excluded) so
                       cross-block wiring survives.

    Only touches operations, never connectivity → preserves A, R, I, L.
    """
    if S and H:          # both active → nothing to break here
        return graph

    graph  = copy.deepcopy(graph)
    layers: Dict[int, List[ComputeNode]] = {}
    for nd in graph:
        if nd.output_bit is not None or not nd.node_id.startswith('L'):
            continue
        lyr = int(nd.node_id.split('_')[0][1:])
        layers.setdefault(lyr, []).append(nd)

    if not S and not H:
        # Need: mixed ops AND at least one wrong-parity op in some layer.
        # Strategy: in any layer, set gate[0] to wrong-parity op,
        # gate[1] to a DIFFERENT op (any parity) → violates both S and H.
        layer_list = list(layers.items())
        random.shuffle(layer_list)
        for lyr, nodes in layer_list:
            if len(nodes) < 2:
                continue
            is_sp = (lyr % 2 == 1)
            wrong_class = OPS_LINEAR    if is_sp else OPS_NONLINEAR
            right_class = OPS_NONLINEAR if is_sp else OPS_LINEAR
            # Protect I: skip L1 if I active and ops need to be binary
            if I and lyr == 1:
                binary = [op for op in OPS_ALL if op not in ('NOT', 'IDENTITY')]
                wrong_binary = [op for op in wrong_class if op not in ('NOT', 'IDENTITY')]
                if not wrong_binary:
                    continue
                nodes[0].operation = random.choice(wrong_binary)
                other = [op for op in binary if op != nodes[0].operation]
                if other:
                    nodes[1].operation = random.choice(other)
            else:
                nodes[0].operation = random.choice(wrong_class)
                other = [op for op in OPS_ALL if op != nodes[0].operation]
                if other:
                    nodes[1].operation = random.choice(other)
            return graph

    elif not S and H:
        # Need: entire layer uses wrong-parity op (homogeneous & wrong parity).
        layer_list = list(layers.items())
        random.shuffle(layer_list)
        for lyr, nodes in layer_list:
            if not nodes:
                continue
            is_sp = (lyr % 2 == 1)
            wrong_class = OPS_LINEAR if is_sp else OPS_NONLINEAR
            # Protect I: skip L1 if I active and all ops need to be binary
            if I and lyr == 1:
                binary_wrong = [op for op in wrong_class if op not in ('NOT', 'IDENTITY')]
                if not binary_wrong:
                    continue
                chosen = random.choice(binary_wrong)
            else:
                chosen = random.choice(wrong_class)
            for nd in nodes:
                nd.operation = chosen
            return graph

    else:  # S=True, H=False
        # Need: at least one layer with ≥2 different ops, all correct parity.
        layer_list = list(layers.items())
        random.shuffle(layer_list)
        for lyr, nodes in layer_list:
            if len(nodes) < 2:
                continue
            is_sp  = (lyr % 2 == 1)
            right_class = OPS_NONLINEAR if is_sp else OPS_LINEAR
            # Protect I: keep binary ops for L1
            if I and lyr == 1:
                right_binary = [op for op in right_class if op not in ('NOT', 'IDENTITY')]
                if len(right_binary) < 2:
                    continue
                ops_in_use = {nd.operation for nd in nodes}
                if len(ops_in_use) == 1 and ops_in_use.issubset(set(right_binary)):
                    alt = [op for op in right_binary if op not in ops_in_use]
                    if alt:
                        nodes[1].operation = random.choice(alt)
                        return graph
            else:
                ops_in_use = {nd.operation for nd in nodes}
                if len(ops_in_use) == 1 and ops_in_use.issubset(set(right_class)):
                    alt = [op for op in right_class if op not in ops_in_use]
                    if alt:
                        nodes[1].operation = random.choice(alt)
                        return graph
    return graph


def _break_locality(graph: List[ComputeNode],
                    R: bool = False, I: bool = True,
                    d_max: int = 4) -> List[ComputeNode]:
    """
    L=False: inject one connection that exceeds d_max on the ring.

    R=True  → only target L1 nodes (their valid predecessors are inputs/states).
    A-safe  → never overwrites a backward edge (a gate ref that appears *after*
              the current node in list order).  Only block-refs (input_*/state_*)
              or forward gate refs are eligible for overwriting.
    I=False → restrict the far candidate pool to the dominant block already
              present in L1, so the distant injection does not accidentally
              create cross-block coverage (which would restore interleaving).
              Applied only when R=True targets L1 nodes.
    """
    n          = SHARED['n_bits']
    graph      = copy.deepcopy(graph)
    b0         = [f"input_{i}" for i in range(0,     n // 2)]
    b1         = [f"input_{i}" for i in range(n // 2, n)]
    all_io     = b0 + b1  # no state_*
    all_io_set = set(all_io)

    # Build position map to detect backward edges
    pos_map = {nd.node_id: i for i, nd in enumerate(graph)}

    l_nodes = ([nd for nd in graph
                if nd.output_bit is None and nd.node_id.startswith('L1_')] if R
               else [nd for nd in graph
                     if nd.output_bit is None and nd.node_id.startswith('L')])

    # I=False: restrict far candidates to the block already in each L1 gate,
    # so the distant injection does not accidentally create cross-block coverage.
    # R=True (split mode): each gate's block is determined individually.
    # R=False (sole-block mode): all L1 gates share one block; use dominant.
    _use_per_gate_block = (not I and R)
    if not I and not R:
        b0_set = set(b0); b1_set = set(b1)
        l1_nodes = [nd for nd in graph
                    if nd.output_bit is None and nd.node_id.startswith('L1_')]
        b0_cnt = sum(1 for nd in l1_nodes for inp in nd.inputs if inp in b0_set)
        b1_cnt = sum(1 for nd in l1_nodes for inp in nd.inputs if inp in b1_set)
        dominant_block = b0 if b0_cnt >= b1_cnt else b1
        far_pool = dominant_block  # state_* excluded
    else:
        far_pool = all_io  # default; overridden per gate when _use_per_gate_block

    random.shuffle(l_nodes)
    for nd in l_nodes:
        try:   gpos = int(nd.node_id.split('_g')[1]) % n
        except: continue
        # Per-gate block restriction for R+I_absent (split mode)
        if _use_per_gate_block:
            b0_set = set(b0); b1_set = set(b1)
            gate_inps = [x for x in nd.inputs if x in all_io_set]
            b0_c = sum(1 for x in gate_inps if x in b0_set)
            b1_c = sum(1 for x in gate_inps if x in b1_set)
            gate_block = b0 if b0_c >= b1_c else b1
            eff_pool = gate_block
        else:
            eff_pool = far_pool
        far = [c for c in eff_pool if _circ_dist(gpos, _ring_pos(c, n)) > d_max]
        if not far:
            continue
        nd_idx = pos_map[nd.node_id]
        # Safe slot: block ref OR forward gate ref (not a backward edge)
        safe_slots = [
            i for i, inp in enumerate(nd.inputs)
            if inp in all_io_set                              # block ref
            or (inp in pos_map and pos_map[inp] < nd_idx)    # forward gate ref
        ]
        if safe_slots:
            nd.inputs[random.choice(safe_slots)] = random.choice(far)
            return graph
    return graph


# =============================================================================


# =============================================================================
# MAIN GENERATOR - ALL 64 COMBINATIONS
# =============================================================================

def create_sbn(
    S: bool = False,
    A: bool = False,
    R: bool = False,
    I: bool = False,
    H: bool = False,
    L: bool = False,
) -> GenericSBN:
    """
    Generate a SBN satisfying EXACTLY the specified constraints and
    VIOLATING every inactive constraint.

    Routing:
      Always uses _build_layered (L*_g* nodes) regardless of which flags
      are active.  _build_generic is kept as a private helper but is no
      longer called by create_sbn.

    Breakers (inactive flags only, surgical):
      A=False → _introduce_cycle(R, L)
      R=False → _break_regularity(A, L)
      S|H     → _break_ops(S, H, I)      when at least one is inactive
      I=False → _break_interleaving(L)
      L=False → _break_locality(R, I)

    All breakers run unconditionally on the layered skeleton because the
    graph always has L*_g* nodes.  The old needs_layers guard is gone.
    """
    # 1. Build layered skeleton enforcing every *active* constraint.
    sbn = _build_layered(S=S, R=R, I=I, H=H, L=L)

    # 2. Enforce acyclicity (post-pass rewires backward edges).
    if A:
        sbn = GenericSBN(_enforce_acyclicity(sbn.compute_graph))

    # 3. Break every *inactive* constraint — order matters:
    #    cycle → regularity → ops → interleaving → locality
    #    (each breaker is surgical and must not undo earlier ones).

    if not A:
        graph = _introduce_cycle(sbn.compute_graph, R=R, L=L)
        try:   sbn = GenericSBN(graph)
        except Exception: pass

    if not R:
        graph = _break_regularity(sbn.compute_graph, A=A, L=L)
        try:   sbn = GenericSBN(graph)
        except Exception: pass

    if not S or not H:
        graph = _break_ops(sbn.compute_graph, S=S, H=H, I=I)
        try:   sbn = GenericSBN(graph)
        except Exception: pass

    # I before L: _break_interleaving restricts to one block; then
    # _break_locality injects a distant connection that cannot be
    # accidentally removed by a later interleaving fix.
    if not I:
        graph = _break_interleaving(list(sbn.compute_graph), L=L, R=R)
        try:   sbn = GenericSBN(graph)
        except Exception: pass

    if not L:
        graph = _break_locality(sbn.compute_graph, R=R, I=I)
        try:   sbn = GenericSBN(graph)
        except Exception: pass

    # ── Input coverage pass (post-breakers) ──────────────────────────────────
    # After all breakers, some input_b may be structurally unreachable.
    # _break_interleaving is the main culprit (L1 restricted to one block or
    # split into two non-crossing halves).
    #
    # Injection rules:
    #   • R=True  → skip L1 entirely (split-mode partition must not be broken);
    #               inject into L2+ where R does not constrain input sources.
    #   • L=True  → only inject into gates within d_max ring distance.
    #   • I=False → for L1 injections: gate must not already read the opposite
    #               block (preserves non-cross-block property of L1).
    #
    # Cost: O(V+E) dead-wire scan + O(dead × gates) ≈ 0.05 ms.
    _needed: set = set()
    _q: list = []
    for _nd in sbn.compute_graph:
        if _nd.output_bit is not None:
            for _inp in _nd.inputs:
                if _inp not in _needed:
                    _needed.add(_inp); _q.append(_inp)
    while _q:
        _nid = _q.pop()
        _nd  = sbn.nodes.get(_nid)
        if _nd is None or _nd.output_bit is not None:
            continue
        for _inp in _nd.inputs:
            if _inp not in _needed:
                _needed.add(_inp); _q.append(_inp)
    _missing = [f"input_{_b}" for _b in range(SHARED['n_bits'])
                if f"input_{_b}" not in _needed]

    if _missing:
        _n  = SHARED['n_bits']
        _d  = SHARED.get('d_max', 4)
        _b0 = set(f"input_{i}" for i in range(_n // 2))
        _b1 = set(f"input_{i}" for i in range(_n // 2, _n))
        _l1 = [_nd for _nd in sbn.compute_graph
               if _nd.output_bit is None and _nd.node_id.startswith('L1_g')]
        _injected = False
        for _inp_name in _missing:
            _inp_bit = int(_inp_name.split('_')[1])
            _inp_blk = _b0 if _inp_name in _b0 else _b1

            # ── L1 candidates ─────────────────────────────────────────────
            _cands: list = []
            for _nd in _l1:
                if L:
                    try:    _gpos = int(_nd.node_id.split('_g')[1]) % _n
                    except: _gpos = 0
                    if min(abs(_gpos - _inp_bit), _n - abs(_gpos - _inp_bit)) > _d:
                        continue
                if not I:
                    # For I=False: gate must read ONLY from the same block as the
                    # missing input (preserves non-cross-block in L1).
                    # For R=True (split mode): gate's block = whichever block its
                    # current inputs belong to.
                    _other = _b1 if _inp_blk is _b0 else _b0
                    if any(_x in _other
                           for _x in _nd.inputs if _x.startswith('input_')):
                        continue  # would create cross-block wiring in L1
                _cands.append(_nd)

            # ── L2+ fallback (only when R=False) ──────────────────────────
            # R=True: injecting into L2+ would violate Regularity (L2 must
            # read only L1 outputs, not raw inputs).  Stay in L1 only.
            if not _cands and not R:
                for _layer in range(2, SHARED.get('depth', 5) + 1):
                    _lp_ok = []
                    for _nd in sbn.compute_graph:
                        if _nd.output_bit is not None: continue
                        if not _nd.node_id.startswith(f'L{_layer}_g'): continue
                        if L:
                            try:    _gpos = int(_nd.node_id.split('_g')[1]) % _n
                            except: _gpos = 0
                            if min(abs(_gpos - _inp_bit),
                                   _n - abs(_gpos - _inp_bit)) > _d:
                                continue
                        _lp_ok.append(_nd)
                    if _lp_ok:
                        _cands = _lp_ok
                        break

            if not _cands:
                continue  # genuinely incompatible — leave dead

            _tgt = random.choice(_cands)
            # Prefer overwriting a forward edge (input_* or earlier gate) to
            # avoid destroying backward edges that implement A=False cycles.
            # A 'safe' slot: input_* (always forward) or a gate id that appears
            # BEFORE _tgt in the compute list (forward reference → safe to replace).
            _tgt_pos = next((i for i, _nd2 in enumerate(sbn.compute_graph)
                             if _nd2.node_id == _tgt.node_id), len(sbn.compute_graph))
            _all_nids = {_nd2.node_id: i for i, _nd2 in enumerate(sbn.compute_graph)}
            _safe_slots = [
                _si for _si, _sinp in enumerate(_tgt.inputs)
                if _sinp.startswith('input_') or _sinp.startswith('state_')
                or (_sinp in _all_nids and _all_nids[_sinp] < _tgt_pos)
            ]
            if _safe_slots:
                _tgt.inputs[random.choice(_safe_slots)] = _inp_name
            elif _tgt.operation in ('NOT', 'IDENTITY'):
                _tgt.inputs[0] = _inp_name
            else:
                _tgt.inputs[-1] = _inp_name
            _injected = True

        if _injected:
            try:
                sbn = GenericSBN(sbn.compute_graph)
            except Exception:
                pass

    # ── Structural dead-wire retry ────────────────────────────────────────────
    # If the coverage pass still left structural dead wires, regenerate the
    # entire SBN (build + breakers + coverage pass) up to MAX_RETRIES times.
    # The coverage pass above is already included in each retry since we call
    # the full pipeline inline.  No recursive call to create_sbn (avoids stack
    # overflow).  Mean retries needed: ~2–4.  MAX_RETRIES=40 gives P(success)
    # ≈ 1-(1-p)^40.  With p≈0.05 (Generic), P≈87%; with p≈0.40 (L), P≈>99%.
    _MAX_RETRIES = 40

    def _struct_dead(sbn_c: GenericSBN) -> bool:
        """True iff any input_b is unreachable from all outputs."""
        _needed: set = set(); _q: list = []
        for _nd in sbn_c.compute_graph:
            if _nd.output_bit is not None:
                for _inp in _nd.inputs:
                    if _inp not in _needed: _needed.add(_inp); _q.append(_inp)
        while _q:
            _nid = _q.pop(); _nd = sbn_c.nodes.get(_nid)
            if _nd is None or _nd.output_bit is not None: continue
            for _inp in _nd.inputs:
                if _inp not in _needed: _needed.add(_inp); _q.append(_inp)
        return any(f"input_{_b}" not in _needed for _b in range(SHARED['n_bits']))

    if _struct_dead(sbn):
        for _retry in range(_MAX_RETRIES):
            # Rebuild full pipeline without calling create_sbn (no recursion).
            _s2 = _build_layered(S=S, R=R, I=I, H=H, L=L)
            if A:
                _s2 = GenericSBN(_enforce_acyclicity(_s2.compute_graph))
            if not A:
                _g = _introduce_cycle(_s2.compute_graph, R=R, L=L)
                try: _s2 = GenericSBN(_g)
                except Exception: pass
            if not R:
                _g = _break_regularity(_s2.compute_graph, A=A, L=L)
                try: _s2 = GenericSBN(_g)
                except Exception: pass
            if not S or not H:
                _g = _break_ops(_s2.compute_graph, S=S, H=H, I=I)
                try: _s2 = GenericSBN(_g)
                except Exception: pass
            if not I:
                _g = _break_interleaving(list(_s2.compute_graph), L=L, R=R)
                try: _s2 = GenericSBN(_g)
                except Exception: pass
            if not L:
                _g = _break_locality(_s2.compute_graph, R=R, I=I)
                try: _s2 = GenericSBN(_g)
                except Exception: pass
            # Re-apply coverage pass on candidate
            _needed2: set = set(); _q2: list = []
            for _nd2 in _s2.compute_graph:
                if _nd2.output_bit is not None:
                    for _i2 in _nd2.inputs:
                        if _i2 not in _needed2: _needed2.add(_i2); _q2.append(_i2)
            while _q2:
                _nid2 = _q2.pop(); _nd2 = _s2.nodes.get(_nid2)
                if _nd2 is None or _nd2.output_bit is not None: continue
                for _i2 in _nd2.inputs:
                    if _i2 not in _needed2: _needed2.add(_i2); _q2.append(_i2)
            _miss2 = [f"input_{_b2}" for _b2 in range(SHARED['n_bits'])
                      if f"input_{_b2}" not in _needed2]
            if _miss2:
                _n2  = SHARED['n_bits']
                _d2  = SHARED.get('d_max', 4)
                _b0s = set(f"input_{_ii}" for _ii in range(_n2 // 2))
                _b1s = set(f"input_{_ii}" for _ii in range(_n2 // 2, _n2))
                for _in2 in _miss2:
                    _ib2 = int(_in2.split('_')[1])
                    _ibk = _b0s if _in2 in _b0s else _b1s
                    _cc: list = []
                    for _ly2 in range(1, SHARED.get('depth', 5) + 1):
                        _pfx = f'L{_ly2}_g'
                        for _nd2 in _s2.compute_graph:
                            if _nd2.output_bit is not None: continue
                            if not _nd2.node_id.startswith(_pfx): continue
                            if L:
                                try: _gp2 = int(_nd2.node_id.split('_g')[1]) % _n2
                                except: _gp2 = 0
                                if min(abs(_gp2-_ib2), _n2-abs(_gp2-_ib2)) > _d2:
                                    continue
                            if not I and _ly2 == 1:
                                _oth2 = _b1s if _ibk is _b0s else _b0s
                                if any(_xx in _oth2 for _xx in _nd2.inputs
                                       if _xx.startswith('input_')):
                                    continue
                            _cc.append(_nd2)
                        if _cc:
                            break
                    if not _cc:
                        continue
                    _tgt2 = random.choice(_cc)
                    # Prefer forward-edge slots to preserve backward edges (A=False).
                    _tp2  = next((i for i, _nd3 in enumerate(_s2.compute_graph)
                                  if _nd3.node_id == _tgt2.node_id),
                                 len(_s2.compute_graph))
                    _nmap = {_nd3.node_id: i
                             for i, _nd3 in enumerate(_s2.compute_graph)}
                    _sf2  = [_si2 for _si2, _sinp2 in enumerate(_tgt2.inputs)
                             if _sinp2.startswith('input_') or _sinp2.startswith('state_')
                             or (_sinp2 in _nmap and _nmap[_sinp2] < _tp2)]
                    if _sf2:
                        _tgt2.inputs[random.choice(_sf2)] = _in2
                    elif _tgt2.operation in ('NOT', 'IDENTITY'):
                        _tgt2.inputs[0] = _in2
                    else:
                        _tgt2.inputs[-1] = _in2
                try: _s2 = GenericSBN(_s2.compute_graph)
                except Exception: pass
            # Accept if no structural dead wires remain AND (A=False → has cycle).
            _has_bw = True
            if not A:
                _seen_r: set = set(f"input_{_bi}" for _bi in range(SHARED['n_bits']))
                _has_bw = False
                for _ndr in _s2.compute_graph:
                    if _ndr.output_bit is not None: continue
                    for _inpr in _ndr.inputs:
                        if _inpr not in _seen_r:
                            _has_bw = True; break
                    if _has_bw: break
                    _seen_r.add(_ndr.node_id)
            if not _struct_dead(_s2) and _has_bw:
                sbn = _s2
                break
        # If MAX_RETRIES exhausted, return the last attempt (may still have
        # algebraic cancellations, but the GA will improve it).

    return sbn

# CONSTRAINT-PRESERVING MUTATION
# =============================================================================

def mutate_sbn(
    sbn:            GenericSBN,
    constraints:    Optional[Dict] = None,
    gate_mut_rate:  float = 0.15,
    wire_mut_rate:  float = 0.15,
) -> GenericSBN:
    """
    Mutate a SBN while preserving active constraints and violating inactive ones.

    Two mutation channels (applied before re-enforcement):
      1. Operation mutation (gate_mut_rate per gate): respects S and H parity.
      2. Wiring mutation   (wire_mut_rate per gate): rewires one input to a new
         source from the valid predecessor pool (respects R structure).

    Post-mutation pipeline mirrors create_sbn:
      Enforce active → break inactive.
    """
    if constraints is None:
        constraints = {}

    A = constraints.get('A', False)
    S = constraints.get('S', False)
    H = constraints.get('H', False)
    L = constraints.get('L', False)
    I = constraints.get('I', False)
    R = constraints.get('R', False)

    needs_layers = True  # graph is always layered (L*_g* nodes)

    new_graph     = copy.deepcopy(sbn.compute_graph)
    layer_op: Dict[int, str] = {}
    inputs_states = [f"input_{i}" for i in range(16)]  # no state_*

    # Layer membership for predecessor-pool-aware wiring
    layer_members: Dict[int, List[str]] = {}
    for node in new_graph:
        if node.output_bit is None and node.node_id.startswith('L'):
            lyr = int(node.node_id.split('_')[0][1:])
            layer_members.setdefault(lyr, []).append(node.node_id)

    dag_available: List[str] = list(inputs_states)

    for node in new_graph:
        if node.output_bit is not None:
            continue

        is_L_node = node.node_id.startswith('L')

        # Predecessor pool (respects R: strict layer k-1)
        if is_L_node:
            lyr       = int(node.node_id.split('_')[0][1:])
            pred_pool = (inputs_states if lyr == 1
                         else layer_members.get(lyr - 1, inputs_states))
        else:
            pred_pool = list(dag_available)
            dag_available.append(node.node_id)

        # Locality filter
        if L and is_L_node:
            n_bits = SHARED['n_bits']
            try:    pos = int(node.node_id.split('_g')[1]) % n_bits
            except: pos = 0
            local = [c for c in pred_pool
                     if _circ_dist(pos, _ring_pos(c, n_bits)) <= 4]
            pred_pool = local if local else pred_pool

        # Operation mutation
        if random.random() < gate_mut_rate:
            if S and H and is_L_node:
                # Both active: correct parity, and whole layer shares same op
                lyr   = int(node.node_id.split('_')[0][1:])
                is_sp = (lyr % 2 == 1)
                if lyr not in layer_op:
                    layer_op[lyr] = _rand_op(nonlinear_only=is_sp, linear_only=not is_sp)
                new_op = layer_op[lyr]
            elif S and is_L_node:
                lyr    = int(node.node_id.split('_')[0][1:])
                is_sp  = (lyr % 2 == 1)
                new_op = _rand_op(nonlinear_only=is_sp, linear_only=not is_sp)
            elif H and is_L_node:
                lyr = int(node.node_id.split('_')[0][1:])
                if lyr not in layer_op:
                    layer_op[lyr] = _rand_op()
                new_op = layer_op[lyr]
            else:
                new_op = _rand_op()
            node.operation = new_op

        # Wiring mutation
        if random.random() < wire_mut_rate and pred_pool:
            node.inputs[random.randrange(len(node.inputs))] = random.choice(pred_pool)

    # ── Enforce H: propagate layer_op to ALL nodes in each mutated layer ──────
    # Without this, unmutated nodes in a layer keep their old op while mutated
    # nodes got a new one → H violated after partial mutation.
    if H:
        layers_seen: Dict[int, str] = {}
        for node in new_graph:
            if node.output_bit is not None or not node.node_id.startswith('L'):
                continue
            lyr = int(node.node_id.split('_')[0][1:])
            if lyr in layer_op:
                # This layer had at least one mutation — enforce uniform op
                node.operation = layer_op[lyr]
            else:
                # No mutation in this layer — record op for consistency check
                if lyr not in layers_seen:
                    layers_seen[lyr] = node.operation
                else:
                    # Unmutated nodes must still share the same op (they should already)
                    node.operation = layers_seen[lyr]

    try:
        result = GenericSBN(new_graph)
    except Exception:
        return sbn

    # ── Re-enforce acyclicity ─────────────────────────────────────────────────
    if A:
        result = GenericSBN(_enforce_acyclicity(result.compute_graph))

    # ── Break inactive constraints (same logic as create_sbn) ────────────────
    if not A:
        graph = _introduce_cycle(result.compute_graph, R=R, L=L)
        try:   result = GenericSBN(graph)
        except Exception: pass

    if needs_layers:
        if not R:
            graph = _break_regularity(result.compute_graph, A=A, L=L)
            try:   result = GenericSBN(graph)
            except Exception: pass
        if not S or not H:
            graph = _break_ops(result.compute_graph, S=S, H=H, I=I)
            try:   result = GenericSBN(graph)
            except Exception: pass
        if not I:
            graph = _break_interleaving(list(result.compute_graph), L=L, R=R)
            try:   result = GenericSBN(graph)
            except Exception: pass
        if not L:
            graph = _break_locality(result.compute_graph, R=R, I=I)
            try:   result = GenericSBN(graph)
            except Exception: pass

    # ── Re-enforce I: guarantee cross-block L1 coverage (LAST, after breakers) ─
    # Must run after all breakers so no subsequent step can erase the injected input.
    if I and needs_layers:
        graph    = list(result.compute_graph)
        n        = SHARED['n_bits']
        b0       = [f"input_{i}" for i in range(0,      n // 2)]
        b1       = [f"input_{i}" for i in range(n // 2, n)]
        b0_set   = set(b0); b1_set = set(b1)
        all_io_s = b0_set | b1_set  # state_* excluded
        l1_nodes = [nd for nd in graph
                    if nd.output_bit is None and nd.node_id.startswith('L1_g')]
        has_b0 = any(inp in b0_set for nd in l1_nodes for inp in nd.inputs)
        has_b1 = any(inp in b1_set for nd in l1_nodes for inp in nd.inputs)
        if l1_nodes and (not has_b0 or not has_b1):
            missing_block = b0 if not has_b0 else b1
            missing_set   = set(missing_block)
            random.shuffle(l1_nodes)
            injected = False
            for nd in l1_nodes:
                try:   pos = int(nd.node_id.split('_g')[1]) % n
                except: pos = 0
                if L:
                    candidates = [c for c in missing_block
                                  if _circ_dist(pos, _ring_pos(c, n)) <= 4]
                    if not candidates:
                        candidates = missing_block
                else:
                    candidates = missing_block
                # Overwrite a state_* or same-block input; never a gate/cycle ref
                safe_slots = [i for i, inp in enumerate(nd.inputs)
                              if inp in all_io_s and inp not in missing_set]
                if safe_slots and candidates:
                    nd.inputs[random.choice(safe_slots)] = random.choice(candidates)
                    injected = True
                    break
            if not injected and l1_nodes:
                for nd in l1_nodes:
                    safe_slots = [i for i, inp in enumerate(nd.inputs)
                                  if inp in all_io_s]
                    if safe_slots and missing_block:
                        nd.inputs[random.choice(safe_slots)] = random.choice(missing_block)
                        break
        try:
            result = GenericSBN(graph)
        except Exception:
            pass

    return result


# =============================================================================
# CONSTRAINT HELPERS
# =============================================================================

CONSTRAINT_KEYS  = ['S', 'A', 'R', 'I', 'H', 'L']
CONSTRAINT_NAMES = ['Stratification', 'Acyclicity', 'Regularity',
                    'Interleaving',   'Homogeneity', 'Locality']


def select_constraints(indices: Tuple[int, ...]) -> Dict[str, bool]:
    """Convert active index tuple to constraints dict."""
    return {key: (i in indices) for i, key in enumerate(CONSTRAINT_KEYS)}


def constraints_to_label(constraints: Dict[str, bool]) -> str:
    """Return a short label like 'S+R+H' or 'Generic'."""
    active = [k for k in CONSTRAINT_KEYS if constraints.get(k, False)]
    return '+'.join(active) if active else 'Generic'


def all_64_architectures() -> List[Dict[str, bool]]:
    """Enumerate all 64 constraint combinations."""
    return [
        {key: bool((mask >> i) & 1) for i, key in enumerate(CONSTRAINT_KEYS)}
        for mask in range(64)
    ]



def _sbn_truth_table(sbn: 'GenericSBN') -> 'np.ndarray':
    """
    Module-level truth table (65536 × 16) for a given SBN.

    Identical to GPUAccelerator._truth_table_cpu but callable without a
    GPUAccelerator instance, so create_sbn can use it for the dead-wire check.
    """
    n    = 16
    size = 1 << n
    x_all = np.arange(size, dtype=np.uint32)

    vals: Dict[str, np.ndarray] = {}
    for i in range(n):
        vals[f"input_{i}"] = ((x_all >> i) & 1).astype(np.uint8)
        vals[f"state_{i}"] = np.zeros(size, dtype=np.uint8)

    for _ in range(20):
        changed = False
        for node in sbn.compute_graph:
            if node.output_bit is not None:
                continue
            if node.node_id in vals:
                continue
            if not all(inp in vals for inp in node.inputs):
                continue
            inp = node.inputs
            a   = vals[inp[0]]
            b   = vals[inp[1]] if len(inp) > 1 else a
            if   node.operation == "NOT":      v = (1 - a).astype(np.uint8)
            elif node.operation == "AND":      v = (a & b)
            elif node.operation == "OR":       v = (a | b)
            elif node.operation == "XOR":      v = (a ^ b)
            elif node.operation == "IDENTITY": v = a.copy()
            else:                              v = np.zeros(size, dtype=np.uint8)
            vals[node.node_id] = v
            changed = True
        if not changed:
            break

    table = np.zeros((size, n), dtype=np.uint8)
    for bit in range(n):
        src = sbn.output_map[bit]
        visited: set = set()
        while src in sbn.nodes and src not in visited:
            visited.add(src)
            nd = sbn.nodes[src]
            if nd.operation == 'IDENTITY' and nd.inputs:
                src = nd.inputs[0]
            else:
                break
        table[:, bit] = vals.get(src, np.zeros(size, dtype=np.uint8))

    return table


# =============================================================================
# GPU ACCELERATOR - FITNESS FUNCTIONS
# =============================================================================

class GPUAccelerator:
    """
    Fitness functions with full GPU pipeline.

    Truth table, WHT, ANF, and DDT all computed on GPU via CUDA kernels.
    Each kernel is compiled once and cached by SBN graph hash.

    Fitness functions:
        diffusion_rounds      : CPU only (16 evaluations, already fast)
        linear_resistance     : GPU pipeline: table → WHT → nonlinearity
        algebraic_degree      : GPU pipeline: table → Möbius → max degree
        differential_resistance: GPU pipeline: table → DDT → max probability
    """

    # DDT kernel — global-memory histogram, no shared memory.
    #
    # Two kernels compiled together:
    #   ddt_batch : for each delta_in in a batch, accumulate output-difference
    #               histogram into a private slice of a global counts array.
    #               One CUDA block per delta_in; threads stride over all n inputs.
    #   ddt_max   : parallel reduction to find the maximum over all histogram bins.
    #
    # Why global memory instead of shared memory:
    #   A full histogram needs n=65536 bins × 4 B = 256 KB per block, which far
    #   exceeds the hardware shared-memory limit (typically 48 KB per block).
    #   Using global memory costs some atomicAdd throughput but avoids the
    #   CUDA_ERROR_INVALID_VALUE that killed the shared-memory approach.
    #
    # Memory per batch: batch_size × n × 4 B = 64 × 65536 × 4 = 16 MB.
    # Allocated once per call and zeroed between batches with cp.zeros / cp.fill.
    _CUDA_DDT = r"""
extern "C" __global__
void ddt_batch(const int* __restrict__ out_int,
               int*       __restrict__ counts,
               int delta_in_start,
               int batch_size,
               int n)
{
    int batch_idx = blockIdx.x;
    int delta_in  = delta_in_start + batch_idx;
    if (delta_in >= n || batch_idx >= batch_size) return;

    // Each block writes to its own private slice to avoid inter-block conflicts.
    int* hist = counts + (long long)batch_idx * n;

    for (int x = threadIdx.x; x < n; x += blockDim.x) {
        int dout = out_int[x] ^ out_int[x ^ delta_in];
        atomicAdd(&hist[dout], 1);
    }
}

extern "C" __global__
void ddt_max(const int* __restrict__ counts,
             int*       __restrict__ global_max,
             long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    atomicMax(global_max, counts[i]);
}
"""

    def __init__(self):
        self.available      = GPU_AVAILABLE
        self._kernel_cache: Dict[int, any] = {}   # sbn_hash → compiled eval kernel
        self._ddt_kernel     = None
        self._ddt_max_kernel = None
        self._popcount_np   = np.array(           # popcount lookup (CPU)
            [bin(i).count('1') for i in range(1 << 16)], dtype=np.uint8
        )

        if self.available:
            try:
                # Check GPU memory WITHOUT allocating large arrays
                device = cp.cuda.Device(0)
                free, total = device.mem_info
                print(f"GPU detected: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
                
                if free < 100 * 1024 * 1024:  # Less than 100 MB free
                    raise RuntimeError(f"Insufficient GPU memory: only {free/1e6:.0f} MB free")
                
                # Only compile kernels, don't allocate large arrays yet
                self._compile_static_kernels()
                print("GPU enabled successfully")
            except Exception as e:
                self.available = False
                print(f"GPU unavailable: {e}")
        else:
            print("CPU mode (no GPU detected)")

    # Compiler options for eval_sbn kernel (per-architecture)
    _KERNEL_OPTS = ('--std=c++11',)

    def _make_kernel(self, code: str, name: str):
        """Compile a per-SBN eval kernel with CUDA include path for cuda_fp16.h"""
        # CUDA include path for cuda_fp16.h and other headers
        cuda_include = '/opt/conda/lib/python3.13/site-packages/nvidia/cuda_runtime/include'
        
        options = (
            '--std=c++11',
            f'-I{cuda_include}',  # Critical: makes cuda_fp16.h accessible
        )
        
        try:
            return cp.RawKernel(code, name, options=options, backend='nvrtc')
        except Exception as e:
            print(f"[WARNING] Kernel compilation with include path failed: {e}")
            # Fallback without include path
            try:
                return cp.RawKernel(code, name, options=('--std=c++11',), backend='nvrtc')
            except:
                return cp.RawKernel(code, name)

    def _compile_static_kernels(self):
        """Compile batched DDT kernels at startup. WHT/ANF use NumPy (faster, no headers)."""
        try:
            self._ddt_kernel     = self._make_kernel(self._CUDA_DDT, 'ddt_batch')
            self._ddt_max_kernel = self._make_kernel(self._CUDA_DDT, 'ddt_max')
            print("DDT kernels compiled successfully")
        except Exception as e:
            print(f"WARNING: DDT kernel compilation failed: {type(e).__name__}")
            print(f"  Error: {str(e)[:200]}")
            print(f"  Differential fitness will use CPU fallback (~30s per eval)")
            self._ddt_kernel     = None
            self._ddt_max_kernel = None

    # =========================================================================
    # Public fitness functions
    # =========================================================================

    def diffusion_rounds(self, sbn: GenericSBN, max_rounds: int = 16) -> int:
        """
        CPU. Minimum rounds until all outputs depend on at least one input.
        Lower = better. Ideal = 1.
        """
        for r in range(1, max_rounds + 1):
            covered = np.zeros(16, dtype=bool)
            for bit in range(16):
                inp = [0] * 16
                inp[bit] = 1
                sbn.reset([0] * 16)
                out = sbn.step(inp)
                for j, v in enumerate(out):
                    if v:
                        covered[j] = True
            if covered.all():
                return r
        return max_rounds

    def linear_resistance(self, sbn: GenericSBN) -> float:
        """
        Minimum nonlinearity over 16 output bits. Higher = better. Ideal = 32640.

        GPU: table (CUDA) → WHT (CUDA) → nonlinearity
        CPU: table (Python loop) → WHT (NumPy butterfly)
        """
        if self.available:
            try:
                return self._linear_gpu(sbn)
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
        return self._linear_cpu(sbn)

    def algebraic_degree(self, sbn: GenericSBN) -> int:
        """
        Maximum ANF degree over 16 output bits. Higher = better. Ideal = 16.

        GPU: table (CUDA) → Möbius (CUDA) → max Hamming weight of nonzero indices
        CPU: table (Python loop) → Möbius (NumPy)
        """
        if self.available:
            try:
                return self._algebraic_gpu(sbn)
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
        return self._algebraic_cpu(sbn)

    def differential_resistance(self, sbn: GenericSBN) -> float:
        """
        Maximum differential probability (DP) over the full DDT.

        Standard metric in the literature (Biham & Shamir 1991, NIST FIPS):

            DP_max = max_{Δin ≠ 0, Δout} DDT[Δin][Δout] / 2^n

        where  DDT[Δin][Δout] = #{x ∈ {0,1}^n : F(x) ⊕ F(x ⊕ Δin) = Δout}

        Lower = better. Ideal (APN): 2 / 2^16 = 3.05e-5.
        A linear function scores 1.0 (all outputs identical for fixed Δin).

        GPU path: exact DDT via CUDA atomicAdd kernel (all 65535 Δin).
        CPU path: exact DDT via NumPy vectorized XOR (slower, same result).
        """
        if self.available:
            try:
                result = self._differential_gpu(sbn)
                if not hasattr(self, '_differential_gpu_ok'):
                    print(f"[differential] GPU path active", flush=True)
                    self._differential_gpu_ok = True
                return result
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Print error on FIRST failure only
                if not hasattr(self, '_differential_gpu_failed'):
                    import traceback
                    print(f"[differential] GPU failed — falling back to CPU (117s/eval!)")
                    print(f"  Error: {type(e).__name__}: {e}")
                    print(f"  GPU available: {self.available}")
                    print(f"  DDT kernel:    {self._ddt_kernel}")
                    print(f"  DDT max kernel:{self._ddt_max_kernel}")
                    traceback.print_exc()
                    self._differential_gpu_failed = True
                pass
        if not hasattr(self, '_differential_cpu_warned'):
            print(f"[differential] Using CPU fallback (~30-120s/eval). Check GPU setup.", flush=True)
            self._differential_cpu_warned = True
        return self._differential_cpu(sbn)

    # =========================================================================
    # GPU pipelines
    # =========================================================================

    def _linear_gpu(self, sbn: GenericSBN) -> float:
        """
        Table on GPU → transfer to CPU → vectorized NumPy WHT.

        WHT on (65536, 16) with NumPy reshape butterfly: 0.04s.
        Much faster than the CUDA kernel path and avoids fp16 header issues.
        """
        n     = 1 << 16
        table = cp.asnumpy(self._truth_table_gpu_raw(sbn))  # (65536, 16) uint8

        f = 1 - 2 * table.astype(np.int32)                 # (65536, 16) ±1
        h = 1
        while h < n:
            f_r      = f.reshape(-1, 2 * h, 16)
            lo, hi   = f_r[:, :h, :].copy(), f_r[:, h:, :]
            f_r[:, :h, :] = lo + hi
            f_r[:, h:, :] = lo - hi
            h *= 2

        return float((n - int(np.max(np.abs(f)))) / 2)

    def _algebraic_gpu(self, sbn: GenericSBN) -> int:
        """
        Table on GPU → transfer to CPU → vectorized NumPy Möbius transform.

        ANF on (65536, 16) with NumPy reshape: 0.002s.
        """
        n   = 1 << 16
        anf = cp.asnumpy(self._truth_table_gpu_raw(sbn)).copy()  # (65536,16) uint8

        for i in range(16):
            step  = 1 << i
            anf_r = anf.reshape(-1, 2 * step, 16)
            anf_r[:, step:, :] ^= anf_r[:, :step, :]

        active = anf.any(axis=1)
        nz     = np.where(active)[0]
        return 0 if len(nz) == 0 else int(self._popcount_np[nz].max())

    def _differential_gpu(self, sbn: GenericSBN) -> float:
        """
        Exact DDT on GPU — one sync, max on CPU.

        Redesign (v5.9.5):
        ------------------
        The old design synced Python↔GPU once per batch (1024 times for the
        full DDT).  Each sync costs ~1-2 ms in Python context-switch overhead,
        giving ~2 s of pure overhead per SBN — dominating the ~0.2 s of actual
        GPU work.

        New design mirrors _linear_gpu / _algebraic_gpu:
          1. Launch ALL ddt_batch kernels back-to-back (no sync between them).
          2. Transfer the entire counts buffer to CPU in ONE call (cp.asnumpy).
          3. Find the global max with NumPy — much faster than the ddt_max kernel.

        Memory: batch_size × n × 4 B = 1024 × 65536 × 4 = 256 MB peak.
        This fits on any modern GPU with ≥1 GB VRAM.
        """
        if self._ddt_kernel is None:
            raise RuntimeError("DDT kernel not compiled")

        n          = 1 << 16        # 65536
        batch_size = 512            # delta_in values per kernel launch
        threads    = 256

        # Pack output table once; keep it on GPU for the whole call.
        table       = self._truth_table(sbn)                         # (n, 16) uint8
        powers      = np.array([1 << i for i in range(16)], dtype=np.int32)
        out_int_gpu = cp.asarray((table.astype(np.int32) * powers).sum(axis=1))

        # Allocate full counts buffer on GPU (reused across batches).
        counts_gpu = cp.zeros(batch_size * n, dtype=cp.int32)

        global_max = 0
        for delta_start in range(1, n, batch_size):
            cur = min(batch_size, n - delta_start)

            # Zero only the used slice (fast cp.fill on a view).
            counts_gpu[:cur * n].fill(0)

            # Launch kernel — no sync here, let the GPU queue pipeline.
            self._ddt_kernel(
                (cur,), (threads,),
                (out_int_gpu, counts_gpu, delta_start, cur, n),
            )

            # One sync + transfer per batch; NumPy max is ~0.1 ms.
            cp.cuda.Stream.null.synchronize()
            batch_max = int(cp.asnumpy(counts_gpu[:cur * n]).max())
            if batch_max > global_max:
                global_max = batch_max

        del out_int_gpu, counts_gpu
        return float(global_max) / float(n)

    # =========================================================================
    # CPU fallbacks
    # =========================================================================

    def _linear_cpu(self, sbn: GenericSBN) -> float:
        """
        Vectorized WHT over all 16 output bits at once.

        Mirrors _linear_gpu exactly but uses _truth_table_cpu.
        Matrix butterfly: ~3 ms vs ~3000 ms for the Python-loop _wht_cpu path.
        """
        n     = 1 << 16
        table = self._truth_table_cpu(sbn)          # (65536, 16) uint8
        f     = 1 - 2 * table.astype(np.int32)      # (65536, 16) ±1
        h = 1
        while h < n:
            f_r    = f.reshape(-1, 2 * h, 16)
            lo, hi = f_r[:, :h, :].copy(), f_r[:, h:, :]
            f_r[:, :h, :] = lo + hi
            f_r[:, h:, :] = lo - hi
            h *= 2
        return float((n - int(np.max(np.abs(f)))) / 2)

    def _algebraic_cpu(self, sbn: GenericSBN) -> int:
        """
        Vectorized Möbius / ANF transform over all 16 output bits at once.

        Mirrors _algebraic_gpu exactly but uses _truth_table_cpu.
        Matrix butterfly: ~0.5 ms vs ~1800 ms for the Python-loop _anf_cpu path.
        """
        table = self._truth_table_cpu(sbn).copy()   # (65536, 16) uint8
        for i in range(16):
            step    = 1 << i
            anf_r   = table.reshape(-1, 2 * step, 16)
            anf_r[:, step:, :] ^= anf_r[:, :step, :]
        # Any row where at least one bit is non-zero is an active monomial
        active = table.any(axis=1)                   # (65536,) bool
        nz     = np.where(active)[0]
        return 0 if len(nz) == 0 else int(self._popcount_np[nz].max())

    def _differential_cpu(self, sbn: GenericSBN) -> float:
        """
        Exact DDT on CPU via batched NumPy.

        Instead of looping over all 65535 delta_in values one by one (~38 s),
        we process them in batches of `batch` rows.  For each batch we build
        a (batch × 65536) array of output differences and compute per-row
        max counts with np.bincount applied via a histogram trick.

        Time: ~2 s on a modern CPU (vs 38 s for the serial loop).
        """
        n         = 16
        size      = 1 << n                                      # 65536
        table     = self._truth_table_cpu(sbn)                  # (65536, 16) uint8
        powers    = np.array([1 << i for i in range(n)], dtype=np.int32)
        out_int   = (table.astype(np.int32) * powers).sum(axis=1)  # (65536,) int32
        x_all     = np.arange(size, dtype=np.int32)

        global_max = 0
        batch      = 256   # rows of DDT processed at once; tunable

        for base in range(1, size, batch):
            deltas = np.arange(base, min(base + batch, size), dtype=np.int32)  # (B,)
            # x_xor[b, x] = x ^ delta[b]  →  shape (B, size)
            x_xor     = x_all[None, :] ^ deltas[:, None]               # (B, size)
            delta_out = out_int[None, :] ^ out_int[x_xor]              # (B, size)
            # Flatten and use a single bincount over all B rows, offset per row
            offsets   = (np.arange(len(deltas), dtype=np.int64) * size)[:, None]
            flat      = (delta_out.astype(np.int64) + offsets).ravel()
            counts    = np.bincount(flat, minlength=len(deltas) * size)
            counts    = counts.reshape(len(deltas), size)
            mc        = int(counts.max())
            if mc > global_max:
                global_max = mc

        return float(global_max) / float(size)

    # =========================================================================
    # Truth table (internal)
    # =========================================================================

    def _truth_table(self, sbn: GenericSBN) -> np.ndarray:
        """Public helper: returns truth table as CPU numpy array."""
        if self.available:
            try:
                return cp.asnumpy(self._truth_table_gpu_raw(sbn))
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
        return self._truth_table_cpu(sbn)

    def _truth_table_gpu_raw(self, sbn: GenericSBN):
        """Returns truth table as CuPy array (stays on GPU)."""
        sbn_hash = self._sbn_hash(sbn)
        if sbn_hash not in self._kernel_cache:
            code   = self._compile_eval_kernel(sbn)
            kernel = self._make_kernel(code, 'eval_sbn')
            self._kernel_cache[sbn_hash] = kernel

        kernel  = self._kernel_cache[sbn_hash]
        n       = 1 << 16
        out_gpu = cp.zeros((n, 16), dtype=cp.uint8)
        threads = 256
        blocks  = (n + threads - 1) // threads
        kernel((blocks,), (threads,), (out_gpu, n))
        cp.cuda.Stream.null.synchronize()
        return out_gpu

    def _truth_table_gpu(self, sbn: GenericSBN) -> np.ndarray:
        """Alias kept for backward compatibility."""
        return cp.asnumpy(self._truth_table_gpu_raw(sbn))

    def _truth_table_cpu(self, sbn: GenericSBN) -> np.ndarray:
        """
        Vectorized NumPy truth table: evaluates all 2^16 inputs simultaneously.

        Semantics: F(x) = sbn.step(x) from zero initial state, i.e. state_i = 0.
        This matches the SBN definition as a stateless S-box for cryptographic
        evaluation: one synchronous step from state=0 given external input x.

        For acyclic graphs (A=True) a single topological pass suffices.
        For cyclic graphs, unresolved nodes (whose inputs include a backward
        edge not yet computed) are left at their state_i = 0 initialisation,
        which is identical to what step() returns after 20 passes when a cycle
        fails to converge (the state bits were reset to 0 before the call).
        This ensures CPU and GPU paths produce identical results.
        """
        n    = 16
        size = 1 << n
        x_all = np.arange(size, dtype=np.uint32)

        # state_i = 0 for all x  (S-box semantics: evaluate from zero state)
        vals: Dict[str, np.ndarray] = {}
        for i in range(n):
            vals[f"input_{i}"] = ((x_all >> i) & 1).astype(np.uint8)
            vals[f"state_{i}"] = np.zeros(size, dtype=np.uint8)

        # Multi-pass topological evaluation — resolves nodes in dependency order.
        # For acyclic graphs: converges in exactly 1 pass.
        # For cyclic graphs: unresolvable nodes stay at 0 (state_i fallback).
        for _ in range(20):
            changed = False
            for node in sbn.compute_graph:
                if node.output_bit is not None:
                    continue
                if node.node_id in vals:
                    continue
                if not all(inp in vals for inp in node.inputs):
                    continue
                inp = node.inputs
                a   = vals[inp[0]]
                b   = vals[inp[1]] if len(inp) > 1 else a
                if   node.operation == "NOT":      v = (1 - a).astype(np.uint8)
                elif node.operation == "AND":      v = (a & b)
                elif node.operation == "OR":       v = (a | b)
                elif node.operation == "XOR":      v = (a ^ b)
                elif node.operation == "IDENTITY": v = a.copy()
                else:                              v = np.zeros(size, dtype=np.uint8)
                vals[node.node_id] = v
                changed = True
            if not changed:
                break

        # Assemble output table (65536 × 16)
        table = np.zeros((size, n), dtype=np.uint8)
        for bit in range(n):
            src = sbn.output_map[bit]
            # Follow IDENTITY chain to the real source
            visited: set = set()
            while src in sbn.nodes and src not in visited:
                visited.add(src)
                nd = sbn.nodes[src]
                if nd.operation == 'IDENTITY' and nd.inputs:
                    src = nd.inputs[0]
                else:
                    break
            table[:, bit] = vals.get(src, np.zeros(size, dtype=np.uint8))

        return table

    # =========================================================================
    # Kernel compiler (per-SBN eval kernel)
    # =========================================================================

    # Increment this version whenever _compile_eval_kernel logic changes.
    # It is mixed into every cache key so stale compiled kernels are never reused.
    _KERNEL_VERSION = 3  # v3: state_i=0 (S-box semantics, not state_i=input_i)

    def _sbn_hash(self, sbn: GenericSBN) -> int:
        parts = [f"{n.node_id}|{n.operation}|{'_'.join(n.inputs)}"
                 for n in sbn.compute_graph]
        return hash((self._KERNEL_VERSION, tuple(parts)))

    def _compile_eval_kernel(self, sbn: GenericSBN) -> str:
        """Unroll SBN circuit into a CUDA C kernel (one thread per input)."""
        topo     = self._topological_order(sbn)
        declared = ({f"input_{i}" for i in range(16)} |
                    {f"state_{i}" for i in range(16)})

        def safe(nid):
            return nid.replace(".", "_").replace("-", "_")

        def val(nid):
            """Return the CUDA variable name for a node id.

            If nid is already declared (input_i, state_i, or a computed gate),
            return its variable name.  Otherwise the node is unresolved — most
            likely part of a cycle in the generic (unconstrained) graph.  In the
            CPU path, cycles are handled by initialising state_i = input_i via
            sbn.reset(bits) before sbn.step(bits).  We replicate that behaviour
            here: map gate_N back to state_{N % 16} so unresolved gates get the
            same fallback value as the CPU path, rather than the constant 0 that
            broke differential scoring.
            """
            if nid in declared:
                return safe(nid)
            # Fallback: use state bit corresponding to node index (mirrors CPU reset)
            try:
                idx = int(nid.split('_')[-1]) % 16
            except (ValueError, IndexError):
                idx = 0
            return f"state_{idx}"

        lines = [
            'extern "C" __global__',
            'void eval_sbn(unsigned char* out, int n) {',
            '    int x = blockIdx.x * blockDim.x + threadIdx.x;',
            '    if (x >= n) return;',
            '',
        ]
        for i in range(16):
            lines.append(f'    int input_{i} = (x >> {i}) & 1;')
            lines.append(f'    int state_{{i}} = 0;  // S-box: zero initial state')
        lines.append('')

        for node_id in topo:
            if node_id in declared or node_id.startswith('output_'):
                continue
            vn = safe(node_id)
            if node_id not in sbn.nodes:
                lines.append(f'    int {vn} = 0;')
                declared.add(node_id)
                continue
            node = sbn.nodes[node_id]
            inp  = node.inputs
            a    = val(inp[0]) if inp      else '0'
            b    = val(inp[1]) if len(inp) > 1 else a
            ops  = {'NOT': f'1-{a}', 'AND': f'{a}&{b}', 'OR': f'{a}|{b}',
                    'XOR': f'{a}^{b}', 'IDENTITY': a}
            lines.append(f'    int {vn} = {ops.get(node.operation, "0")};')
            declared.add(node_id)
        lines.append('')

        for bit in range(16):
            src = sbn.output_map[bit]
            visited = set()
            while src in sbn.nodes and src not in visited:
                visited.add(src)
                nd = sbn.nodes[src]
                if nd.operation == 'IDENTITY' and nd.inputs:
                    src = nd.inputs[0]
                else:
                    break
            lines.append(f'    out[x*16+{bit}] = (unsigned char){val(src)};')

        lines.append('}')
        return '\n'.join(lines)

    def _topological_order(self, sbn: GenericSBN) -> List[str]:
        inputs   = ([f'input_{i}' for i in range(16)] +
                    [f'state_{i}' for i in range(16)])
        resolved = set(inputs)
        ordered  = list(inputs)
        pending  = [n for n in sbn.compute_graph if n.output_bit is None]
        for _ in range(len(pending) + 1):
            progress, still = False, []
            for node in pending:
                if all(inp in resolved for inp in node.inputs):
                    ordered.append(node.node_id)
                    resolved.add(node.node_id)
                    progress = True
                else:
                    still.append(node)
            pending = still
            if not progress:
                for node in pending:
                    ordered.append(node.node_id)
                break
        return ordered

    # =========================================================================
    # CPU transform helpers
    # =========================================================================

    def _wht_cpu(self, col: np.ndarray) -> np.ndarray:
        f = 1 - 2 * col.astype(np.int32)
        h = 1
        while h < len(f):
            for i in range(0, len(f), h * 2):
                for j in range(i, i + h):
                    f[j], f[j+h] = f[j]+f[j+h], f[j]-f[j+h]
            h *= 2
        return f

    def _anf_cpu(self, col: np.ndarray) -> np.ndarray:
        anf = col.copy().astype(np.uint8)
        n   = int(np.log2(len(anf)))
        for i in range(n):
            step = 1 << i
            for j in range(0, len(anf), step << 1):
                for k in range(j, j + step):
                    anf[k+step] ^= anf[k]
        return anf

    # Aliases for backward compatibility with RewardFunctions
    def _wht(self, col): return self._wht_cpu(col)
    def _anf(self, col): return self._anf_cpu(col)

# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class RewardFunctions:
    """
    6 reward functions to guide GA evolution.

    Rewards are RELATIVE signals used during selection, not primary fitness.
    They measure how much a mutation improved a cryptographic property.

    All rewards return float in [-1, +1] or [0, 1] as documented.
    """

    def __init__(self, gpu: 'GPUAccelerator'):
        self.gpu = gpu

    # ------------------------------------------------------------------
    # Helper: compute all 6 rewards with shared truth table
    # ------------------------------------------------------------------

    def compute_all_rewards(self,
                            sbn:        GenericSBN,
                            sbn_before: Optional[GenericSBN] = None,
                            ) -> Dict[str, float]:
        """
        Compute all 6 rewards for `sbn`, sharing one truth-table computation
        across R1 (algebraic_degree_delta), R2 (anf_entropy), and R3
        (walsh_flatness).

        Truth table is computed once and reused — saves ~2 redundant
        full table builds compared to calling each reward independently.

        Args:
            sbn:        the SBN to evaluate (the "after" state in the GA).
            sbn_before: optional "before" SBN for R1 and R5.
                        If None, R1 returns 0.0 and R5 returns 0.0.

        Returns:
            Dict with keys: 'algebraic_delta', 'anf_entropy', 'walsh_flatness',
                            'differential_uniformity', 'dependency_growth',
                            'symmetry_breaking'.
        """
        # One truth-table build for sbn (used by R2, R3, and R1 if k=1)
        table_after = self.gpu._truth_table(sbn)

        # R1 also needs a truth table for sbn_before
        if sbn_before is not None:
            table_before = self.gpu._truth_table(sbn_before)
            r1 = self.algebraic_degree_delta(
                sbn_before, sbn,
                table_before=table_before,
                table_after=table_after,
            )
            r5 = self.dependency_growth(sbn_before, sbn)
        else:
            r1 = 0.0
            r5 = 0.0

        r2 = self.anf_entropy(sbn,        table=table_after)
        r3 = self.walsh_flatness(sbn,     table=table_after)
        r4 = self.differential_uniformity(sbn)
        r6 = self.symmetry_breaking(sbn)

        return {
            'algebraic_delta':          r1,
            'anf_entropy':              r2,
            'walsh_flatness':           r3,
            'differential_uniformity':  r4,
            'dependency_growth':        r5,
            'symmetry_breaking':        r6,
        }

    # ------------------------------------------------------------------
    # Reward 1: Algebraic degree delta
    # ------------------------------------------------------------------

    def algebraic_degree_delta(self,
                               sbn_before:    GenericSBN,
                               sbn_after:     GenericSBN,
                               k:             int = 1,
                               table_before:  Optional[np.ndarray] = None,
                               table_after:   Optional[np.ndarray] = None) -> float:
        """
        Reward 1: Change in effective algebraic degree after k iterations.

        Evaluates ANF degree of F^k (k-fold composition) for before and after.

        r = (deg_after - deg_before) / 16   ∈ [-1, +1]

        Positive = degree increased after mutation = better.

        Args:
            table_before: optional precomputed truth table for sbn_before
                          (shape (65536, 16), dtype uint8). Avoids recomputing
                          when the caller already has it (e.g. from anf_entropy
                          or walsh_flatness called on the same SBN).
            table_after:  same for sbn_after.
        """
        def _degree_from_table(table: np.ndarray) -> int:
            max_deg = 0
            for i in range(16):
                anf = self.gpu._anf(table[:, i])
                nz  = np.nonzero(anf)[0]
                if len(nz):
                    max_deg = max(max_deg,
                                  int(np.max([bin(int(x)).count('1') for x in nz])))
            return max_deg

        def _build_table_k(sbn: GenericSBN) -> np.ndarray:
            n     = 16
            size  = 1 << n
            table = np.zeros((size, n), dtype=np.uint8)
            for x in range(size):
                bits  = [(x >> i) & 1 for i in range(n)]
                state = list(bits)
                sbn.reset(state)
                for _ in range(k):
                    state = sbn.step(state)
                table[x] = state
            return table

        # k=1 → reuse caller-supplied table if available; otherwise build it
        if k == 1:
            tb = table_before if table_before is not None else self.gpu._truth_table(sbn_before)
            ta = table_after  if table_after  is not None else self.gpu._truth_table(sbn_after)
        else:
            # k>1 requires composing F k times — build dedicated tables
            tb = _build_table_k(sbn_before)
            ta = _build_table_k(sbn_after)

        deg_before = _degree_from_table(tb)
        deg_after  = _degree_from_table(ta)
        return float(deg_after - deg_before) / 16.0

    # ------------------------------------------------------------------
    # Reward 2: ANF entropy
    # ------------------------------------------------------------------

    def anf_entropy(self,
                    sbn:   GenericSBN,
                    table: Optional[np.ndarray] = None) -> float:
        """
        Reward 2: Entropy of ANF coefficients (monomial density).

        For each output bit i:
            p_i = fraction of nonzero ANF coefficients
            H_i = binary entropy of p_i

        r = mean_i H_i   ∈ [0, 1]

        Ideal = 1 (exactly half the monomials are active).

        Args:
            table: optional precomputed truth table (shape (65536, 16), uint8).
                   Pass it to avoid redundant computation when the same SBN is
                   evaluated by multiple reward functions.
        """
        if table is None:
            table = self.gpu._truth_table(sbn)
        entropies = []
        for i in range(16):
            anf = self.gpu._anf(table[:, i])
            p   = float(np.sum(anf)) / len(anf)
            if p == 0.0 or p == 1.0:
                h = 0.0
            else:
                h = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
            entropies.append(h)
        return float(np.mean(entropies))

    # ------------------------------------------------------------------
    # Reward 3: Walsh spectrum flatness
    # ------------------------------------------------------------------

    def walsh_flatness(self,
                       sbn:   GenericSBN,
                       table: Optional[np.ndarray] = None) -> float:
        """
        Reward 3: Flatness of Walsh-Hadamard spectrum (excluding DC term).

        For each output bit i:
            spectrum_i = |WHT_i[a]|  for a = 1..2^16-1
            flatness_i = 1 - std(spectrum_i) / max(spectrum_i)

        r = mean_i flatness_i   ∈ [0, 1]

        Ideal = 1 (perfectly flat spectrum → bent function).

        Args:
            table: optional precomputed truth table (shape (65536, 16), uint8).
                   Pass it to avoid redundant computation when the same SBN is
                   evaluated by multiple reward functions.
        """
        if table is None:
            table = self.gpu._truth_table(sbn)
        flatnesses = []
        for i in range(16):
            wht      = self.gpu._wht(table[:, i])
            spectrum = np.abs(wht[1:]).astype(np.float64)  # Exclude DC
            mx       = float(np.max(spectrum))
            if mx == 0.0:
                flatnesses.append(1.0)
            else:
                flatnesses.append(1.0 - float(np.std(spectrum)) / mx)
        return float(np.mean(flatnesses))

    # ------------------------------------------------------------------
    # Reward 4: Differential uniformity
    # ------------------------------------------------------------------

    def differential_uniformity(self,
                                 sbn: GenericSBN,
                                 num_deltas: int = 64,
                                 num_samples: int = 512) -> float:
        """
        Reward 4: Uniformity of differential output distribution.

        For each sampled Δin ≠ 0, compute empirical distribution of Δout.
        Lower variance = more uniform = better resistance.

        r = 1 - mean_variance / 0.25   ∈ [0, 1]

        Ideal = 1 (perfectly uniform differential distribution).

        Note: reduced num_deltas/num_samples for speed (~2s vs ~300s).
        """
        n         = 16
        variances = []

        for _ in range(num_deltas):
            delta_in      = random.randint(1, (1 << n) - 1)
            delta_in_bits = [(delta_in >> i) & 1 for i in range(n)]
            counts: Dict[int, int] = defaultdict(int)

            for _ in range(num_samples):
                x      = random.randint(0, (1 << n) - 1)
                x_bits = [(x >> i) & 1 for i in range(n)]
                xd_bits = [a ^ b for a, b in zip(x_bits, delta_in_bits)]

                sbn.reset(x_bits)
                out_x   = sbn.step(x_bits)
                sbn.reset(xd_bits)
                out_xd  = sbn.step(xd_bits)

                delta_out = sum((out_x[i] ^ out_xd[i]) << i for i in range(n))
                counts[delta_out] += 1

            probs = np.array(list(counts.values()), dtype=np.float64) / num_samples
            variances.append(float(np.var(probs)))

        mean_var = float(np.mean(variances))
        return float(np.clip(1.0 - mean_var / 0.25, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Reward 5: Dependency growth
    # ------------------------------------------------------------------

    def dependency_growth(self,
                          sbn_before: GenericSBN,
                          sbn_after:  GenericSBN) -> float:
        """
        Reward 5: Increase in dependency matrix density after mutation.

        density = number of 1s in D(16×16) / 256

        r = density_after - density_before   ∈ [-1, +1]

        Positive = more dependencies after mutation = better diffusion.
        """
        def _density(sbn: GenericSBN) -> float:
            D = np.zeros((16, 16), dtype=int)
            for bit in range(16):
                output_id = sbn.output_map[bit]
                visited   = set()
                deps      = set()

                def _find_deps(nid: str):
                    if nid in visited:
                        return
                    visited.add(nid)
                    if nid.startswith('state_'):
                        try:
                            deps.add(int(nid.split('_')[1]))
                        except ValueError:
                            pass
                        return
                    if nid.startswith('input_'):
                        return
                    if nid in sbn.nodes:
                        for inp in sbn.nodes[nid].inputs:
                            _find_deps(inp)

                _find_deps(output_id)
                for j in deps:
                    D[bit, j] = 1
            return float(np.sum(D)) / 256.0

        return float(_density(sbn_after) - _density(sbn_before))

    # ------------------------------------------------------------------
    # Reward 6: Symmetry breaking
    # ------------------------------------------------------------------

    def symmetry_breaking(self, sbn: GenericSBN, width: int = 8) -> float:
        """
        Reward 6: Fraction of structurally unique gates.

        Each gate's iterative signature is computed (operation + sorted
        child signatures). Leaves (inputs/states) are normalized by their
        relative role within groups of `width` bits to detect cross-group
        symmetries.

        Cycle-safe: nodes involved in cycles receive a sentinel signature
        ('CYCLE',) instead of causing infinite recursion.

        penalty = duplicate_gates / total_gates
        r = 1 - penalty   ∈ [0, 1]

        Ideal = 1 (no duplicate subgraphs → fully broken symmetry).
        """
        cache:   Dict[str, tuple] = {}
        IN_PROGRESS = object()   # sentinel: node is on the current DFS path

        def _leaf_role(nid: str) -> str:
            parts = nid.split('_')
            kind  = parts[0]
            idx   = int(parts[1])
            return f"{kind}_{idx % width}"

        def _sig_iterative(root: str) -> tuple:
            """
            Iterative post-order DFS.
            Uses an explicit stack of (node_id, iterator_over_inputs) pairs.
            Cycle detection: mark nodes as IN_PROGRESS before pushing children;
            if a child is already IN_PROGRESS, assign ('CYCLE',) immediately.
            """
            if root in cache:
                return cache[root]

            # stack entries: (nid, inputs_iter_or_None)
            # None means the node is a leaf / already handled
            stack: list = [(root, None)]
            in_progress: set = set()

            while stack:
                nid, it = stack[-1]

                # ── Leaf: resolve immediately ─────────────────────────────
                if nid.startswith('input_') or nid.startswith('state_'):
                    cache[nid] = (_leaf_role(nid),)
                    stack.pop()
                    continue

                if nid not in sbn.nodes:
                    cache[nid] = (nid,)
                    stack.pop()
                    continue

                # ── Already resolved ──────────────────────────────────────
                if nid in cache and cache[nid] is not IN_PROGRESS:
                    stack.pop()
                    continue

                node = sbn.nodes[nid]

                # ── First visit: initialize iterator ──────────────────────
                if it is None:
                    cache[nid] = IN_PROGRESS          # mark as in-progress
                    in_progress.add(nid)
                    stack[-1] = (nid, iter(node.inputs))
                    continue

                # ── Subsequent visit: advance iterator ────────────────────
                try:
                    child = next(it)
                    if child in in_progress or (child in cache and cache[child] is IN_PROGRESS):
                        # Cycle detected — assign sentinel and keep going
                        cache[child] = ('CYCLE',)
                    elif child not in cache:
                        stack.append((child, None))
                except StopIteration:
                    # All children resolved → compute this node's signature
                    child_sigs = tuple(
                        sorted(
                            cache.get(inp, ('UNRESOLVED',))
                            for inp in node.inputs
                        )
                    )
                    cache[nid] = (node.operation, child_sigs)
                    in_progress.discard(nid)
                    stack.pop()

            return cache.get(root, ('UNRESOLVED',))

        gate_sigs = [
            _sig_iterative(n.node_id)
            for n in sbn.compute_graph
            if n.node_type == NodeType.LOGIC_GATE
        ]

        if not gate_sigs:
            return 1.0

        sig_counts: Dict[tuple, int] = defaultdict(int)
        for s in gate_sigs:
            sig_counts[s] += 1

        duplicates = sum(c - 1 for c in sig_counts.values())
        return float(1.0 - duplicates / len(gate_sigs))


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

def run_genetic_algorithm(
    constraints:       Dict[str, bool],
    fitness_fn:        Callable,
    maximize:          bool,
    population_size:   int,
    num_generations:   int,
    mutation_rate:     float,
    progress_callback: Optional[Callable] = None,
) -> Tuple:
    """Run GA over the constrained SBN space."""
    population  = [create_sbn(**constraints) for _ in range(population_size)]
    elite_count = max(1, population_size // 5)
    best_sbn    = None
    best_score  = -float('inf') if maximize else float('inf')
    history     = []

    for gen in range(num_generations):
        scores = [fitness_fn(sbn) for sbn in population]
        idx    = np.argmax(scores) if maximize else np.argmin(scores)

        if ((maximize and scores[idx] > best_score) or
                (not maximize and scores[idx] < best_score)):
            best_score = scores[idx]
            best_sbn   = population[idx]

        history.append(best_score)

        if progress_callback:
            progress_callback(gen, num_generations, best_score)

        ranked  = sorted(zip(population, scores),
                         key=lambda x: x[1], reverse=maximize)
        parents = [sbn for sbn, _ in ranked[:elite_count]]

        new_pop = list(parents)
        while len(new_pop) < population_size:
            parent = random.choice(parents)
            child  = parent
            for _ in range(max(1, int(np.random.poisson(mutation_rate)))):
                child = mutate_sbn(child, constraints=constraints)
            new_pop.append(child)

        population = new_pop[:population_size]

    return best_sbn, best_score, history
