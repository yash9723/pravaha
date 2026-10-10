"""Runnable check, no pytest needed: python test_qaoa_logistics.py

1. Each QUBO's lowest-energy bitstring is a valid plan whose cost matches a brute force written
   straight from the problem statement (catches weak penalties and bit-order bugs). Always true
   for the exact encodings; for knapsack's approximate one, on at least 85 of 100 instances.
2. The Qiskit Hamiltonian matches the QUBO energies up to a constant.
3. QAOA, end to end, finds the optimal truck load and beats random guessing at it.
"""
import itertools

import numpy as np

from qaoa_logistics import BUILDERS, VEHICLES, bits_table, path_len, qubo_to_ising, solve


def brute_force(name, data, size):
    if name == "tsp":
        return min(path_len(data["d"], [0, *p, 0]) for p in itertools.permutations(range(1, size)))
    if name == "vrp":
        return min(
            sum(path_len(data["d"], [0, *r, 0]) for r in np.split(np.array(p), cuts))
            for p in itertools.permutations(range(1, size + 1))
            for cuts in itertools.combinations(range(1, size), VEHICLES - 1)
        )
    if name == "knapsack":
        w, v, W = data["w"], data["v"], data["W"]
        return -max(v @ x for x in itertools.product([0, 1], repeat=size) if w @ x <= W)
    f, c = data["f"], data["c"]  # facility: open set S, everyone goes to their nearest open depot
    return min(f[list(S)].sum() + c[:, list(S)].min(1).sum()
               for r in range(1, len(f) + 1) for S in itertools.combinations(range(len(f)), r))


for name, (build, size) in BUILDERS.items():
    seeds = range(100 if name == "knapsack" else 5)
    exact = 0
    for seed in seeds:
        prob = build(size, np.random.default_rng(seed))
        B = bits_table(prob.n)
        E_obj, E_pen = prob.obj.energies(B), prob.pen.energies(B)
        valid = prob.valid(B) if prob.valid else E_pen < 1e-6
        k = np.argmin(E_obj + E_pen)
        exact += valid[k] and np.isclose(E_obj[k], brute_force(name, prob.data, size))
        diag = qubo_to_ising(prob.obj.Q + prob.pen.Q).to_matrix(sparse=True).diagonal().real
        shift = diag - (E_obj + E_pen)
        assert np.allclose(shift, shift[0]), (name, seed, "Hamiltonian != QUBO")
    # knapsack's slack-free penalty is approximate by design (about 90% measured); the others are exact
    need = 85 if name == "knapsack" else len(seeds)
    assert exact >= need, (name, f"QUBO optimum == brute force on only {exact}/{len(seeds)}")
    print(f"ok  {name}: QUBO optimum == brute force on {exact}/{len(seeds)}, Hamiltonian == QUBO")

res = solve(BUILDERS["knapsack"][0](6, np.random.default_rng(7)), seed=7)
p_opt = res.is_opt @ res.probs
assert res.ratio == 1.0, res.ratio
assert p_opt > 2 * res.is_opt.mean(), "QAOA should beat random guessing"
print(f"ok  knapsack: QAOA found the optimal load, P(optimal)={p_opt:.3f} vs random {res.is_opt.mean():.3f}")
