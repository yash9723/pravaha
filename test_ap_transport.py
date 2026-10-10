"""Runnable check, no pytest needed: python test_ap_transport.py

For every AP scheduling problem, on several hubs, depots and seeds: the QUBO's lowest-energy state
is a valid plan, and the plan as the dashboard decodes it costs exactly what a brute force written
straight from the problem statement finds. Then QAOA, end to end, finds the optimal plans.
"""
import itertools

import numpy as np

import ap_transport as ap
from qaoa_logistics import MAX_QUBITS, bits, bits_table, solve


def qubo_best(prob):
    B = bits_table(prob.n)
    E_obj, E_pen = prob.obj.energies(B), prob.pen.energies(B)
    k = int(np.argmin(E_obj + E_pen))
    assert E_pen[k] < 1e-6, (prob.title, "QUBO optimum breaks a constraint")
    return prob.decode(bits(k, prob.n))


def brute_transfer(d):
    free, step = d["free"], d["step"]
    best = float("inf")
    for s in itertools.permutations(range(d["slots"]), len(free)):  # distinct slots: one route per slot
        t = dict(zip(free, (x * step for x in s))) | d["fixed"]
        best = min(best, sum(p * ap.wait_min(t[a], t[b]) for (a, b), p in d["w"].items()))
    return best


def brute_buses(d):
    trips, depot, n = d["trips"], d["depot"], len(d["trips"])
    options = [[None] + [j for j in range(n) if j != i and ap.can_follow(trips[i], trips[j])] for i in range(n)]
    best = float("inf")
    for nxt in itertools.product(*options):  # each trip names the trip its bus runs next
        used = [j for j in nxt if j is not None]
        if len(used) != len(set(used)):
            continue
        heads = set(range(n)) - set(used)
        km = 0.0
        for h in heads:
            chain = [h]
            while nxt[chain[-1]] is not None:
                chain.append(nxt[chain[-1]])
            km += ap.empty_km(depot, trips[chain[0]].frm) + ap.empty_km(trips[chain[-1]].to, depot)
            km += sum(ap.empty_km(trips[a].to, trips[b].frm) for a, b in zip(chain, chain[1:]))
        best = min(best, len(heads) * d["bus_cost"] + km)
    return best


def brute_crews(d):
    n, duties, cost = len(d["trips"]), d["duties"], d["cost"]
    return min(sum(cost[k] for k in pick)
               for r in range(1, n + 1) for pick in itertools.combinations(range(len(duties)), r)
               if sorted(t for k in pick for t in duties[k]) == list(range(n)))  # every trip exactly once


def brute_depots(d):
    f, c = d["f"], d["c"]
    return min(f[list(S)].sum() + c[:, list(S)].min(1).sum()  # every town goes to its nearest open depot
               for r in range(1, len(f) + 1) for S in itertools.combinations(range(len(f)), r))


def main():
    cases = [
        *[(ap.build_transfer(hub, ap.routes_through(hub)[:R], slots, seed), brute_transfer, "pax_min")
          for hub, R, slots, seed in [("Vijayawada", 4, 4, 0), ("Vijayawada", 4, 4, 1), ("Guntur", 3, 4, 0),
                                      ("Tirupati", 3, 5, 0)]],
        *[(ap.build_transfer(hub, free, 4, 0, fixed, count), brute_transfer, "pax_min")  # routes timed elsewhere
          for hub, free, fixed, count in [
              ("Guntur", ("GNT-TNL", "AMR-GNT"), (("NH16-S", 7.5), ("VJA-KNL", 41.2)), True),
              ("Vijayawada", ("VJA-KNL", "VJA-TNL", "AMR-GNT"), (("NH16-N", 0.0), ("NH16-S", 30.0)), False),
              ("Kurnool", ("KNL-HUP",), (("VJA-KNL", 52.7), ("TPT-KNL", 3.3)), True)]],
        # what-if inputs: a festival surge at a hub, dearer diesel for buses and depots
        (ap.build_transfer("Tirupati", ap.routes_through("Tirupati"), 4, 0, surge=3), brute_transfer, "pax_min"),
        (ap.build_buses("Vijayawada", 8, 2, fuel=1.4), brute_buses, None),
        (ap.build_depots(["Vijayawada", "Gudivada"], 0, 0, ("Guntur", "Tenali", "Amaravati"), 1.4), brute_depots,
         "cost"),
        *[(ap.build_buses(depot, 8, seed), brute_buses, None)
          for depot, seed in [("Vijayawada", 0), ("Vijayawada", 2), ("Kurnool", 0), ("Rajahmundry", 0)]],
        *[(ap.build_crews(depot, 8, seed), brute_crews, "paid_h")
          for depot, seed in [("Vijayawada", 0), ("Guntur", 4), ("Kurnool", 0), ("Visakhapatnam", 1)]],
        *[(ap.build_depots(depots, towns, seed), brute_depots, "cost")
          for depots, towns, seed in [(["Vijayawada", "Guntur", "Eluru"], 4, 0),
                                      (["Vijayawada", "Guntur", "Eluru"], 4, 1),
                                      (["Kurnool", "Kadapa", "Anantapur"], 4, 0),
                                      (["Tirupati", "Nellore", "Kadapa", "Chittoor"], 3, 0)]],
    ]
    for prob, brute, key in cases:
        plan = qubo_best(prob)
        got = plan["buses"] * prob.data["bus_cost"] + plan["empty_km"] if key is None else plan[key]
        want = brute(prob.data)
        assert np.isclose(got, want), (prob.title, got, want)
        print(f"ok  {prob.title} ({prob.n} qubits): QUBO optimum == brute force ({want:,.1f})")

    for prob in (ap.build_transfer("Vijayawada", ap.routes_through("Vijayawada")[:4]), ap.build_buses("Vijayawada"),
                 ap.build_crews("Vijayawada"), ap.build_depots(["Vijayawada", "Guntur", "Eluru"])):
        res = solve(prob, seed=0)
        p_opt = res.is_opt @ res.probs
        assert res.ratio == 1.0, (prob.title, res.ratio)
        print(f"ok  QAOA finds the optimum: {prob.title}, P(optimal)={p_opt:.3f} vs random {res.is_opt.mean():.2g}")

    for seed, scenario in itertools.product(range(3), ap.SCENARIOS):  # every piece fits (or is reported skipped)
        pieces = ap.statewide_pieces(seed, scenario)
        for kind, label, args in pieces:
            try:
                assert ap.BUILDERS[kind](*args).n <= MAX_QUBITS, (kind, label, seed, scenario)
            except ValueError:
                pass
        for closed in ap.SCENARIOS[scenario].get("closed", ()):  # nothing is based at a closed depot
            assert all(label != closed and closed not in args[0] for kind, label, args in pieces), (scenario, closed)
    pieces = ap.statewide_pieces(0)
    small = [p for p in ap.statewide_pieces(0) if p[0] == "buses"][:3]
    out = list(ap.solve_statewide(pieces=small, hubs=False, workers=2))  # parallel, or the in-process fallback
    assert sorted(o[0][1] for o in out) == sorted(p[1] for p in small) and all(o[1] for o in out)
    print(f"ok  statewide: {len(pieces)} independent pieces within {MAX_QUBITS} qubits; parallel solving returns all")

    # Hub chain: every piece is solved exactly, and a route's fixed time at a later hub is its time at the
    # hub where it was timed, moved along the road (route_clock).
    timed_at, chain = {}, ap.solve_hub_chain(0)
    for (kind, label, args), res, err, _ in chain:
        prob = ap.build_transfer(*args)
        assert np.isclose(qubo_best(prob)["pax_min"], brute_transfer(prob.data)), label
        arrive = prob.decode(bits(res.found if res.found is not None else res.opt, prob.n))["arrive"]
        hub, free, _, _, fixed, *_ = args
        for r, t in fixed:
            g, t_g = timed_at[r]
            c = ap.route_clock(r)
            assert np.isclose(t, round((t_g + c[hub] - c[g]) % ap.CYCLE_MIN, 1)), (label, r)
        timed_at |= {r: (hub, arrive[r]) for r in free}
    assert len(chain) == len(ap.hub_schedule()) and set(timed_at) >= {r for h in ap.STATIONS if len(
        ap.routes_through(h)) >= 3 for r in ap.routes_through(h)}
    print(f"ok  hub chain: {len(chain)} coordinated pieces, each exact; fixed times follow the roads")


if __name__ == "__main__":  # worker processes re-import this module; only the parent runs the checks
    main()
