"""Quantum logistics optimization with QAOA (Qiskit).

Each logistics problem is written as a QUBO (minimize x.Q.x over bits x), solved by one
shared QAOA engine, and checked against the exact optimum from exhaustive search.

  tsp       one vehicle: shortest delivery route through every stop
  vrp       two vehicles share the customers: shortest total distance
  knapsack  load the most valuable parcels onto a truck with a weight limit
  facility  choose which depots to open and which depot serves each customer

usage: python qaoa_logistics.py {tsp,vrp,knapsack,facility,all} [--size N] [--reps 2] [--seed 7]
"""
import argparse
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from qiskit import transpile
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

MAX_QUBITS = 20  # the statevector and the energy table both hold 2^n entries
SIM = AerSimulator(method="statevector", max_parallel_threads=1)  # at <= 20 qubits threads cost more than they save
VEHICLES = 2
DEPOTS = 3


# ------------------------------------------------------------------ QUBO + QAOA engine


class Qubo:
    """Energy x.Q.x + offset over bits x. Q is upper-triangular; x_i^2 = x_i, so the diagonal is linear."""

    def __init__(self, n):
        self.Q, self.offset = np.zeros((n, n)), 0.0

    def add(self, i, j, w):
        self.Q[min(i, j), max(i, j)] += w

    def square(self, terms, target, A):
        """Add A * (sum(c * x_i for i, c in terms) - target) ** 2."""
        for (i, ci), (j, cj) in itertools.combinations(terms, 2):
            self.add(i, j, 2 * A * ci * cj)
        for i, ci in terms:
            self.add(i, i, A * (ci * ci - 2 * target * ci))
        self.offset += A * target**2

    def energies(self, B):
        return ((B @ self.Q) * B).sum(1) + self.offset


@dataclass
class Problem:
    title: str
    obj: Qubo  # the cost to minimize
    pen: Qubo  # constraint penalties; exact ones are 0 when satisfied, >= A (more than any saving) when broken
    describe: Callable  # bits -> one-line summary of the plan
    draw: Callable  # (matplotlib axes, bits) -> None
    valid: Callable = None  # bits table -> mask of real solutions; default: no penalty paid
    data: dict = field(default_factory=dict)  # raw instance, for the brute-force test
    decode: Callable = None  # bits -> structured plan, for the dashboard

    @property
    def n(self):
        return len(self.obj.Q)


@dataclass
class Result:
    found: int | None  # basis state of the best valid sample, None if no sample was valid
    opt: int  # basis state of the exact optimum
    probs: np.ndarray  # final QAOA distribution over all 2^n basis states
    valid: np.ndarray
    is_opt: np.ndarray
    ratio: float  # approximation ratio of `found` (1.0 = optimal)


def bits_table(n):
    """Row k holds the bits of basis state k; column i is qubit i (Qiskit's little-endian order)."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)


def bits(k, n):
    return (k >> np.arange(n)) & 1


def qubo_to_ising(Q):
    """Cost Hamiltonian from x_i = (1 - Z_i) / 2; equals the QUBO up to a constant."""
    off = np.triu(Q, 1)
    terms = [("Z", [i], -Q[i, i] / 2 - (off[i].sum() + off[:, i].sum()) / 4) for i in range(len(Q))]
    terms += [("ZZ", [int(i), int(j)], off[i, j] / 4) for i, j in zip(*np.nonzero(off))]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=len(Q)).simplify()


def cvar(p, E, alpha):
    """Mean energy of the best `alpha` share of outcomes (CVaR, Barkoutsos et al. 2020). E sorted ascending."""
    k = min(np.searchsorted(np.cumsum(p), alpha), len(p) - 1)
    w = p[: k + 1].copy()
    w[-1] = alpha - w[:-1].sum()
    return w @ E[: k + 1] / alpha


def solve(prob, reps=2, restarts=3, seed=0, alpha=0.25, shots=1024):
    B = bits_table(prob.n)
    E_obj, E_pen = prob.obj.energies(B), prob.pen.energies(B)
    E = E_obj + E_pen
    valid = prob.valid(B) if prob.valid else E_pen < 1e-6
    is_opt = valid & np.isclose(E_obj, E_obj[valid].min())
    order = np.argsort(E)
    E_sorted = E[order]

    Q = prob.obj.Q + prob.pen.Q
    H = qubo_to_ising(Q / np.abs(Q).max())  # unit scale keeps good gamma angles near O(1)
    circuit = transpile(qaoa_ansatz(H, reps=reps), basis_gates=["h", "rx", "rz", "rzz", "cx"])
    circuit.save_probabilities()

    # ponytail: exact statevector simulation caps this near 20 qubits. For real hardware, transpile for the
    # backend, measure instead of save_probabilities, and swap probs() for qiskit-ibm-runtime SamplerV2 counts.
    def probs(theta):
        return np.asarray(SIM.run(circuit.assign_parameters(theta)).result().data(0)["probabilities"])

    rng = np.random.default_rng(seed)
    runs = [
        minimize(lambda t: cvar(probs(t)[order], E_sorted, alpha), rng.uniform(0, np.pi, circuit.num_parameters),
                 method="COBYLA", options={"maxiter": 200})
        for _ in range(restarts)
    ]
    p = probs(min(runs, key=lambda r: r.fun).x)

    samples = rng.choice(len(p), size=shots, p=p / p.sum())
    good = samples[valid[samples]]
    found = int(good[np.argmin(E_obj[good])]) if len(good) else None
    opt = int(np.flatnonzero(is_opt)[0])
    c, c_opt = (E_obj[found] if found is not None else np.nan), E_obj[opt]
    ratio = 1.0 if np.isclose(c, c_opt) else (c / c_opt if c_opt < 0 else c_opt / c)
    return Result(found, opt, p, valid, is_opt, float(np.nan_to_num(ratio)))


def plot(prob, res, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    if res.found is None:
        left.text(0.5, 0.5, "no valid solution sampled", ha="center")
    else:
        b = bits(res.found, prob.n)
        prob.draw(left, b)
        left.set_title("QAOA: " + prob.describe(b), fontsize=9)
    top = np.argsort(res.probs)[::-1][:10]
    colors = ["tab:green" if res.is_opt[k] else "tab:blue" if res.valid[k] else "lightgray" for k in top]
    right.bar(range(len(top)), res.probs[top], color=colors)
    right.set_xticks(range(len(top)), ["".join(map(str, bits(k, prob.n))) for k in top],
                     rotation=60, fontsize=7, family="monospace")
    right.set_ylabel("probability")
    right.set_title(f"Most likely QAOA outcomes (bits x0..x{prob.n - 1})", fontsize=9)
    right.legend(handles=[Patch(color="tab:green", label="optimal"), Patch(color="tab:blue", label="valid"),
                          Patch(color="lightgray", label="breaks a constraint")])
    fig.suptitle(prob.title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ logistics problems


def path_len(d, route):
    return sum(d[a, b] for a, b in zip(route, route[1:]))


def distances(pts):
    return np.linalg.norm(pts[:, None] - pts[None], axis=-1).round(1)


def draw_routes(ax, pts, routes):
    for route, color in zip(routes, ["tab:orange", "tab:purple", "tab:green", "tab:brown"]):
        for a, b in zip(route, route[1:]):
            ax.annotate("", pts[b], pts[a], arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.scatter(*pts[1:].T, s=80, zorder=3, label="stop")
    ax.scatter(*pts[0], s=160, marker="s", c="tab:red", zorder=3, label="depot")
    for k, xy in enumerate(pts):
        ax.annotate(str(k), xy, xytext=(6, 6), textcoords="offset points")
    ax.set_aspect("equal")
    ax.legend(loc="best")


def build_tsp(size, rng):
    """One vehicle leaves depot 0 and visits stops 1..size-1. x[i,t] = 1: stop i is visited t-th."""
    if size < 3:
        raise ValueError("tsp needs --size >= 3 (the depot plus at least 2 stops)")
    pts = rng.uniform(0, 10, (size, 2))
    d = distances(pts)
    m = size - 1  # stops to order; fixing the depot first saves 2m - 1 qubits

    def var(i, t):
        return (i - 1) * m + t

    obj, pen = Qubo(m * m), Qubo(m * m)
    for i in range(1, size):
        obj.add(var(i, 0), var(i, 0), d[0, i])  # depot -> first stop
        obj.add(var(i, m - 1), var(i, m - 1), d[i, 0])  # last stop -> depot
        for j in range(1, size):
            for t in range(m - 1):
                if i != j:
                    obj.add(var(i, t), var(j, t + 1), d[i, j])
    # Costs are >= 0, so a penalty above any one route's length means breaking a rule never pays.
    A = path_len(d, [*range(size), 0]) + 1
    for i in range(1, size):
        pen.square([(var(i, t), 1) for t in range(m)], 1, A)  # each stop visited once
    for t in range(m):
        pen.square([(var(i, t), 1) for i in range(1, size)], 1, A)  # one stop per time slot

    def route(b):
        return [0, *(i for t in range(m) for i in range(1, size) if b[var(i, t)]), 0]

    return Problem(
        "TSP: one-vehicle delivery route", obj, pen,
        describe=lambda b: f"route {'->'.join(map(str, route(b)))}  length {path_len(d, route(b)):.1f}",
        draw=lambda ax, b: draw_routes(ax, pts, [route(b)]),
        data={"d": d},
    )


def build_vrp(size, rng):
    """VEHICLES vehicles leave depot 0 and together serve customers 1..size. x[i,j] = 1: a vehicle drives i -> j."""
    if not 2 <= size <= 3:
        raise ValueError("vrp supports --size 2..3: this edge formulation has no subtour elimination, which "
                         "becomes necessary from 4 customers (MTZ constraints, at a steep qubit cost)")
    N = size + 1
    pts = rng.uniform(0, 10, (N, 2))
    d = distances(pts)
    edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    var = {e: k for k, e in enumerate(edges)}
    obj, pen = Qubo(len(edges)), Qubo(len(edges))
    for k, (i, j) in enumerate(edges):
        obj.add(k, k, d[i, j])
    # Penalty above a known plan's length: vehicle 1 serves customer 1, vehicle 2 serves the rest.
    A = path_len(d, [0, 1, 0]) + path_len(d, [0, *range(2, N), 0]) + 1
    for v in range(N):
        need = VEHICLES if v == 0 else 1  # every vehicle leaves and returns; each customer: in once, out once
        pen.square([(var[v, j], 1) for j in range(N) if j != v], need, A)
        pen.square([(var[j, v], 1) for j in range(N) if j != v], need, A)

    def routes(b):
        nxt = {i: j for (i, j), k in var.items() if b[k] and i}
        out = []
        for j in [j for (i, j), k in var.items() if b[k] and i == 0]:
            r = [0, j]
            while r[-1] != 0:
                r.append(nxt[r[-1]])
            out.append(r)
        return out

    def describe(b):
        rs = routes(b)
        return f"routes {' | '.join('->'.join(map(str, r)) for r in rs)}  total {sum(path_len(d, r) for r in rs):.1f}"

    return Problem(f"VRP: {VEHICLES} vehicles share {size} customers", obj, pen, describe,
                   draw=lambda ax, b: draw_routes(ax, pts, routes(b)), data={"d": d})


def build_knapsack(size, rng):
    """Load `size` parcels (weight w, value v) onto a truck of capacity W, maximizing value.

    The rule sum(w x) <= W uses unbalanced penalization (Montanez-Barrera et al. 2023): with
    spare room h = W - sum(w x), add lam * (h^2 - h). Overloading (h < 0) costs a lot, while a
    little spare room costs almost nothing. Unlike the textbook slack-variable encoding, it
    needs no extra qubits (4 fewer here), and QAOA finds the best load 2-5x more often than a
    random guess (the slack encoding managed about 1x). The trade-off: it's approximate, so in
    about 1 instance in 10 the QUBO's lowest state isn't the best load. Reported answers stay
    correct, because the readout scores samples on their real weight and value.
    """
    if size < 2:
        raise ValueError("knapsack needs --size >= 2")
    w, v = rng.integers(1, 6, size), rng.integers(1, 10, size)
    W = int(w.sum()) // 2
    lam = (v / w).mean()  # value per unit weight, so the penalty scales with the values
    obj, pen = Qubo(size), Qubo(size)
    for i in range(size):
        obj.add(i, i, -v[i])
        pen.add(i, i, lam * w[i])  # -lam * h = lam * sum(w x) - lam * W
    pen.offset -= lam * W
    pen.square([(i, w[i]) for i in range(size)], W, lam)  # +lam * h^2

    def describe(b):
        x = b[:size].astype(bool)
        return f"parcels {np.flatnonzero(x).tolist()}  weight {w[x].sum()}/{W}  value {v[x].sum()}"

    def draw(ax, b):
        x = b[:size].astype(bool)
        for row, chosen in ((1, x), (0, ~x)):
            left = 0
            for i in np.flatnonzero(chosen):
                ax.barh(row, w[i], left=left, color="tab:green" if row else "lightgray", edgecolor="k")
                ax.text(left + w[i] / 2, row, f"P{i}\nv={v[i]}", ha="center", va="center", fontsize=8)
                left += w[i]
        ax.axvline(W, color="tab:red", ls="--", label=f"capacity {W}")
        ax.set_yticks([0, 1], ["left behind", "on truck"])
        ax.set_xlabel("weight")
        ax.legend()

    return Problem(f"Knapsack: load {size} parcels onto one truck", obj, pen, describe, draw,
                   valid=lambda B: B @ w <= W,  # the penalty isn't 0 on valid loads, so check weight directly
                   data={"w": w, "v": v, "W": W})


def facility_qubo(f, c):
    """Open depots (fixed cost f_j) and serve each customer i from an open one (cost c_ij).
    Qubit j: depot j opens. Qubit M + i*M + j: customer i is served by depot j.
    Returns obj, pen and plan(bits) -> (open depots, the depot serving each customer)."""
    M, size = len(f), len(c)

    def x(i, j):
        return M + i * M + j

    obj, pen = Qubo(M * (size + 1)), Qubo(M * (size + 1))
    for j in range(M):
        obj.add(j, j, f[j])
        for i in range(size):
            obj.add(x(i, j), x(i, j), c[i, j])
    A = f[0] + c[:, 0].sum() + 1  # above the cost of "open depot 0, send everyone there"; costs >= 0
    for i in range(size):
        pen.square([(x(i, j), 1) for j in range(M)], 1, A)  # served exactly once...
        for j in range(M):  # ...and only by an open depot: A * x_ij * (1 - y_j)
            pen.add(x(i, j), x(i, j), A)
            pen.add(x(i, j), j, -A)

    def plan(b):
        return [j for j in range(M) if b[j]], [next(j for j in range(M) if b[x(i, j)]) for i in range(size)]

    return obj, pen, plan


def build_facility(size, rng):
    """Open some of DEPOTS candidate depots (fixed cost f) and serve each of `size` customers from an
    open one (cost = distance)."""
    if size < 1:
        raise ValueError("facility needs --size >= 1")
    depots, cust = rng.uniform(0, 10, (DEPOTS, 2)), rng.uniform(0, 10, (size, 2))
    c = np.linalg.norm(cust[:, None] - depots[None], axis=-1).round(1)
    f = rng.uniform(3, 8, DEPOTS).round(1)
    obj, pen, plan = facility_qubo(f, c)

    def describe(b):
        opened, serve = plan(b)
        cost = f[opened].sum() + sum(c[i, j] for i, j in enumerate(serve))
        return f"open depots {opened}  customer->depot {serve}  cost {cost:.1f}"

    def draw(ax, b):
        opened, serve = plan(b)
        for i, j in enumerate(serve):
            ax.plot(*zip(cust[i], depots[j]), color="tab:gray", zorder=1)
        ax.scatter(*cust.T, s=80, zorder=3, label="customer")
        ax.scatter(*depots.T, s=220, marker="s", zorder=3, edgecolors="tab:red", linewidths=2,
                   facecolors=["tab:red" if j in opened else "none" for j in range(DEPOTS)],
                   label="depot (filled = open)")
        for j, xy in enumerate(depots):
            ax.annotate(f"D{j} (open cost {f[j]})", xy, xytext=(8, 8), textcoords="offset points", fontsize=8)
        for i, xy in enumerate(cust):
            ax.annotate(f"C{i}", xy, xytext=(6, 6), textcoords="offset points")
        ax.set_aspect("equal")
        ax.legend(loc="best")

    return Problem(f"Facility location: {DEPOTS} candidate depots, {size} customers", obj, pen, describe, draw,
                   data={"f": f, "c": c})


BUILDERS = {"tsp": (build_tsp, 4), "vrp": (build_vrp, 3), "knapsack": (build_knapsack, 6),
            "facility": (build_facility, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", choices=[*BUILDERS, "all"])
    ap.add_argument("--size", type=int, help="stops / customers / parcels / customers (ignored with 'all')")
    ap.add_argument("--reps", type=int, default=2, help="QAOA layers p (default 2)")
    ap.add_argument("--restarts", type=int, default=3, help="optimizer restarts (default 3)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    for name in BUILDERS if a.problem == "all" else [a.problem]:
        build, size = BUILDERS[name]
        if a.size and a.problem != "all":
            size = a.size
        try:
            prob = build(size, np.random.default_rng(a.seed))
        except ValueError as e:
            raise SystemExit(f"{name}: {e}")
        if prob.n > MAX_QUBITS:
            raise SystemExit(f"{name} --size {size} needs {prob.n} qubits; exact simulation is capped at "
                             f"{MAX_QUBITS}. Use a smaller --size.")
        print(f"\n== {prob.title} | {prob.n} qubits, p={a.reps} ==")
        t0 = time.perf_counter()
        res = solve(prob, a.reps, a.restarts, a.seed)
        print("QAOA   :", prob.describe(bits(res.found, prob.n)) if res.found is not None else "no valid sample")
        print("Optimal:", prob.describe(bits(res.opt, prob.n)), " (exhaustive search)")
        print(f"P(optimal)={res.is_opt @ res.probs:.3f} (random guess: {res.is_opt.mean():.2g})  "
              f"P(valid)={res.valid @ res.probs:.3f} (random: {res.valid.mean():.3f})  "
              f"approx ratio={res.ratio:.3f}  [{time.perf_counter() - t0:.1f}s]")
        plot(prob, res, f"{name}.png")
        print(f"saved {name}.png")


if __name__ == "__main__":
    main()
