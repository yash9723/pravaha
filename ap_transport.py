"""Quantum-assisted public transport scheduling for Andhra Pradesh.

A statewide bus schedule has thousands of trips; QAOA on a simulator handles about 20 qubits.
So the system is hybrid: each scheduling decision is cut into small pieces (one hub, one
depot's batch of trips, one region), each piece becomes a QUBO that QAOA solves
(qaoa_logistics.solve), and every answer is checked against exhaustive search.

  build_transfer  hub transfer timing: when each route reaches a hub, so connections are short
  build_buses     bus-to-trip assignment: chain a depot's trips into as few bus duties as possible
  build_crews     crew duty selection: cover every trip with exactly one crew duty, least paid time
  build_depots    depot allocation: which depots to run and which depot houses each town's buses

solve_statewide runs the whole state: hubs as a chain (solve_hub_chain), where a route timed at
one hub is fixed at the next by the road between them, and every other piece in parallel.

Data: data/stations.csv (approximate city coordinates) and data/routes.csv (illustrative
corridors along real highways). Demand, trip times and costs are synthetic; the constants below
name every assumption. To use APSRTC data, replace the CSVs and these constants.
"""
import csv
import itertools
import logging
import os
import sys
import time
import types
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qaoa_logistics import MAX_QUBITS, Problem, Qubo, bits, facility_qubo, solve

DATA = Path(__file__).parent / "data"
log = logging.getLogger(__name__)

# Illustrative assumptions -- replace with APSRTC figures.
SPEED_KMPH = 45  # average express bus speed, stops included
CIRCUITY = 1.3  # road km per straight-line km
CYCLE_MIN = 60  # hub timetables repeat every hour
DWELL_MIN = 10  # a bus stands this long at the hub
MIN_TRANSFER_MIN = 5  # walk between platforms
MAX_TRIP_KM = 160  # depot trips serve towns within this road distance
MIN_LAYOVER_MIN = 10  # turnaround between two trips of one bus or crew
MAX_LAYOVER_MIN = 240  # a bus idles at most this long between trips
BUS_COST_KM = 200  # one more bus in service costs as much as this many empty km
SIGN_ON_OFF_H = 0.5  # paid crew time to sign on and off
MAX_DUTY_H = 9  # longest crew duty, travel to and from the depot included
MAX_DUTIES = 16  # candidate duties offered to QAOA (one qubit each)
COST_PER_KM = 45  # Rs per bus-km of empty running


def load_network():
    with open(DATA / "stations.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stations = {r["name"]: (float(r["lat"]), float(r["lon"])) for r in rows}
    regions = {r["name"]: r.get("region") or "Andhra Pradesh" for r in rows}  # the region column is optional
    with open(DATA / "routes.csv", newline="", encoding="utf-8") as f:
        routes = {r["route_id"]: {"name": r["name"], "stops": r["stops"].split("|")} for r in csv.DictReader(f)}
    for rid, r in routes.items():
        unknown = [s for s in r["stops"] if s not in stations]
        if unknown:
            raise ValueError(f"routes.csv: route {rid} has stops missing from stations.csv: {unknown}")
    return stations, regions, routes


STATIONS, REGION, ROUTES = load_network()


def road_km(a, b):
    """Road distance estimate: great-circle distance times CIRCUITY."""
    (la1, lo1), (la2, lo2) = np.radians(STATIONS[a]), np.radians(STATIONS[b])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(h)) * CIRCUITY)


def drive_min(a, b):
    return road_km(a, b) / SPEED_KMPH * 60


def routes_through(station):
    return [rid for rid, r in ROUTES.items() if station in r["stops"]]


def clock(minutes):
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


# ------------------------------------------------------------------ hub transfer timing


def wait_min(t_from, t_to):
    """Minutes from stepping off a bus that reaches the hub at t_from to the departure of a bus that
    reaches it at t_to (minutes past the hour; buses leave DWELL_MIN after arriving)."""
    return (t_to - t_from + DWELL_MIN - MIN_TRANSFER_MIN) % CYCLE_MIN + MIN_TRANSFER_MIN


def transfer_demand(hub, a, b, seed):
    """Synthetic passengers/hour changing from route a to route b at hub, the same in every piece."""
    return int(np.random.default_rng([seed, zlib.crc32(f"{hub}|{a}|{b}".encode())]).integers(5, 60))


def build_transfer(hub, route_ids, slots=4, seed=0, fixed=(), count_fixed=True, surge=1):
    """Choose when each route in route_ids reaches `hub` within the hourly cycle so that connecting
    passengers wait least. x[r, s] = 1: route r arrives in slot s, i.e. s * CYCLE/slots past the
    hour, and leaves DWELL_MIN later. These routes share one platform group: one route per slot.

    fixed = ((route, minute), ...) are routes already timed at another hub. They cost no qubits, but
    every transfer to or from them counts. count_fixed: also count transfers between two fixed
    routes (true for a hub's first piece, so each transfer is counted exactly once statewide).
    surge multiplies every transfer's passengers (a festival, say)."""
    free, fixed = list(route_ids), dict(fixed)
    R = len(free)
    if R < 1 or R + len(fixed) < 2:
        raise ValueError("pick at least 2 routes through the hub")
    if slots < R:
        raise ValueError("need at least as many time slots as routes (one route per slot)")
    step = CYCLE_MIN / slots
    idx = {r: i for i, r in enumerate(free)}
    w = {(a, b): round(transfer_demand(hub, a, b, seed) * surge)
         for a, b in itertools.permutations(free + list(fixed), 2) if a in idx or b in idx or count_fixed}

    def var(r, s):
        return idx[r] * slots + s

    obj, pen = Qubo(R * slots), Qubo(R * slots)
    for (a, b), p in w.items():
        if a in fixed and b in fixed:  # already decided: a constant
            obj.offset += p * wait_min(fixed[a], fixed[b])
        elif b in fixed:
            for s in range(slots):
                obj.add(var(a, s), var(a, s), p * wait_min(s * step, fixed[b]))
        elif a in fixed:
            for s in range(slots):
                obj.add(var(b, s), var(b, s), p * wait_min(fixed[a], s * step))
        else:
            for sa, sb in itertools.product(range(slots), repeat=2):
                obj.add(var(a, sa), var(b, sb), p * wait_min(sa * step, sb * step))
    guess = np.zeros(R * slots)
    guess[[i * slots + i for i in range(R)]] = 1
    A = obj.energies(guess[None])[0] + 1  # waits are >= 0, so beating any valid timetable's total never pays
    for r in free:
        pen.square([(var(r, s), 1) for s in range(slots)], 1, A)  # each route gets one slot
    for s in range(slots):
        for r1, r2 in itertools.combinations(free, 2):
            pen.add(var(r1, s), var(r2, s), A)  # one route per slot

    def decode(bits_):
        arrive = {r: next(s for s in range(slots) if bits_[var(r, s)]) * step for r in free} | fixed
        transfers = [(a, b, p, wait_min(arrive[a], arrive[b])) for (a, b), p in w.items()]
        pax_min, pax = sum(p * m for *_, p, m in transfers), sum(w.values())
        return {"arrive": arrive, "fixed": set(fixed), "transfers": transfers, "pax_min": pax_min, "pax": pax,
                "avg_wait": pax_min / pax}

    def describe(bits_):
        d = decode(bits_)
        times = ", ".join(f"{r} :{int(d['arrive'][r]):02d}" for r in free)
        return f"{times} | avg transfer wait {d['avg_wait']:.1f} min"

    return Problem(f"Hub transfer timing at {hub}", obj, pen, describe, None,
                   data={"w": w, "free": free, "fixed": fixed, "slots": slots, "step": step}, decode=decode)


def route_clock(rid):
    """Minutes from a route's first stop to reaching each stop: driving plus DWELL_MIN per stop."""
    stops, t = ROUTES[rid]["stops"], [0.0]
    for a, b in zip(stops, stops[1:]):
        t.append(t[-1] + drive_min(a, b) + DWELL_MIN)
    return dict(zip(stops, t))


def hub_schedule():
    """The order hubs (stations with 3+ routes) are timed in, busiest first, in platform groups of up
    to HUB_GROUP free routes: [(hub, free routes, fixed routes, first piece at this hub)]. A route
    timed at one hub is fixed at every later one, so later pieces time only what is still free.

    ponytail: a hub whose routes were all timed elsewhere gets no piece, so transfers there aren't
    scored; none of today's hubs hit this."""
    timed, order = set(), []
    for hub in sorted((s for s in STATIONS if len(routes_through(s)) >= 3), key=lambda s: (-len(routes_through(s)), s)):
        through, first = routes_through(hub), True
        while free := [r for r in through if r not in timed][:HUB_GROUP]:
            order.append((hub, tuple(free), tuple(r for r in through if r in timed), first))
            timed.update(free)
            first = False
    return order


def solve_hub_chain(seed=0, reps=2, restarts=1, qseed=0, scenario="Normal day"):
    """Time every hub in hub_schedule order. After each piece, every newly timed route's arrival time
    at each of its other stops follows from the road between them, so later pieces take it as fixed.
    Returns solve_piece results whose args carry those fixed times, so each piece can be rebuilt."""
    known, out = {}, []  # (route, stop) -> arrival minute past the hour
    schedule, surge = hub_schedule(), SCENARIOS[scenario].get("surge", {})
    n_groups, seen = Counter(hub for hub, *_ in schedule), Counter()
    for hub, free, fixed_ids, first in schedule:
        seen[hub] += 1
        label = hub + (f" (platform group {seen[hub]})" if n_groups[hub] > 1 else "")
        args = (hub, free, 4, seed, tuple((r, known[r, hub]) for r in fixed_ids), first, surge.get(hub, 1))
        result = solve_piece(("transfer", label, args), reps, restarts, qseed)
        out.append(result)
        res = result[1]
        if res is None:  # can't happen at today's sizes; time the routes in slot order so the chain goes on
            arrive = {r: i * CYCLE_MIN / 4 for i, r in enumerate(free)}
        else:
            prob = build_transfer(*args)
            arrive = prob.decode(bits(res.found if res.found is not None else res.opt, prob.n))["arrive"]
        for r in free:
            clock_ = route_clock(r)
            for stop, t in clock_.items():
                known[r, stop] = round((arrive[r] + t - clock_[hub]) % CYCLE_MIN, 1)
    return out


# ------------------------------------------------------------------ depot trips (buses and crews)


@dataclass(frozen=True)
class Trip:
    name: str
    frm: str
    to: str
    start: float  # minutes after midnight
    end: float


def depot_towns(depot):
    return sorted({s for r in ROUTES.values() if depot in r["stops"] for s in r["stops"]
                   if s != depot and road_km(depot, s) <= MAX_TRIP_KM})


def depot_trips(depot, n, seed):
    """A synthetic day: `n` trips between `depot` and towns on its routes, leaving 05:30-15:00."""
    towns = depot_towns(depot)
    if not towns:
        raise ValueError(f"no towns within {MAX_TRIP_KM} km of {depot} on its routes")
    rng = np.random.default_rng(seed)
    raw = []
    for k in range(n):
        town = towns[rng.integers(len(towns))]
        a, b = (depot, town) if k % 2 == 0 else (town, depot)
        raw.append((int(rng.integers(22, 61)) * 15, a, b))  # 05:30 .. 15:00 in 15-minute steps
    raw.sort()
    return [Trip(f"T{k + 1}", a, b, s, s + round(drive_min(a, b))) for k, (s, a, b) in enumerate(raw)]


def empty_km(a, b):
    return road_km(a, b) if a != b else 0.0


def can_follow(t1, t2):
    """Can one bus run t2 after t1 (driving empty between them if needed) without idling too long?"""
    return t1.end + drive_min(t1.to, t2.frm) + MIN_LAYOVER_MIN <= t2.start <= t1.end + MAX_LAYOVER_MIN


def build_buses(depot, n_trips=8, seed=0, fuel=1.0):
    """Chain a depot's trips into bus duties. z[i, j] = 1: the bus that runs trip i runs trip j
    next, running empty from i's end to j's start if they differ. Buses used = trips - links.
    Cost = buses * bus_cost + empty km (pull-out from and pull-in to the depot included), where a
    bus is worth BUS_COST_KM empty km at today's fuel price, and fewer when fuel costs `fuel` times more."""
    trips = depot_trips(depot, n_trips, seed)
    n = len(trips)
    arcs = [(i, j) for i, j in itertools.permutations(range(n), 2) if can_follow(trips[i], trips[j])]
    if not arcs:
        raise ValueError("no two of these trips can share a bus; try more trips or another seed")
    bus_cost = BUS_COST_KM / fuel
    pull_out = [empty_km(depot, t.frm) for t in trips]
    pull_in = [empty_km(t.to, depot) for t in trips]
    obj, pen = Qubo(len(arcs)), Qubo(len(arcs))
    obj.offset = n * bus_cost + sum(pull_out) + sum(pull_in)  # no links: every trip is its own bus
    gain = []
    for k, (i, j) in enumerate(arcs):  # a link saves a bus and a pull-in/pull-out, but may run empty
        gain.append(bus_cost + pull_in[i] + pull_out[j] - empty_km(trips[i].to, trips[j].frm))
        obj.add(k, k, -gain[-1])
    # Dropping one link costs at most max(gain), so a clash penalty above that never pays.
    A = max(max(gain), 0) + 1
    for i in range(n):
        for side in (0, 1):  # at most one next trip (side 0) and one previous trip (side 1) per trip
            for k1, k2 in itertools.combinations([k for k, arc in enumerate(arcs) if arc[side] == i], 2):
                pen.add(k1, k2, A)

    def decode(b):
        nxt = {arcs[k][0]: arcs[k][1] for k in range(len(arcs)) if b[k]}
        blocks = []
        for i in sorted(set(range(n)) - set(nxt.values())):
            chain = [i]
            while chain[-1] in nxt:
                chain.append(nxt[chain[-1]])
            blocks.append(chain)
        km = sum(pull_out[c[0]] + pull_in[c[-1]] + sum(empty_km(trips[a].to, trips[b2].frm) for a, b2 in zip(c, c[1:]))
                 for c in blocks)
        return {"blocks": blocks, "buses": len(blocks), "empty_km": km}

    def describe(b):
        d = decode(b)
        return f"{d['buses']} buses, {d['empty_km']:.0f} empty km"

    return Problem(f"Bus-to-trip assignment at {depot} depot", obj, pen, describe, None,
                   data={"trips": trips, "depot": depot, "bus_cost": bus_cost}, decode=decode)


def build_crews(depot, n_trips=8, seed=0):
    """Cover every trip with exactly one crew duty, least paid hours (set partitioning). Candidate
    duties are chains of up to 3 trips that connect station to station; y_d = 1: duty d is worked.
    Paid time runs from leaving the depot (as passengers, if the first trip starts elsewhere) to
    getting back, plus SIGN_ON_OFF_H."""
    trips = depot_trips(depot, n_trips, seed)
    n = len(trips)

    def paid_h(chain):
        first, last = trips[chain[0]], trips[chain[-1]]
        on = first.start - (drive_min(depot, first.frm) if first.frm != depot else 0)
        off = last.end + (drive_min(last.to, depot) if last.to != depot else 0)
        return SIGN_ON_OFF_H + (off - on) / 60

    multi, frontier = [], [[i] for i in range(n)]
    for _ in range(2):  # extend to 2- and 3-trip duties
        frontier = [c + [j] for c in frontier for j in range(n)
                    if trips[c[-1]].to == trips[j].frm and trips[c[-1]].end + MIN_LAYOVER_MIN <= trips[j].start]
        multi += [c for c in frontier if paid_h(c) - SIGN_ON_OFF_H <= MAX_DUTY_H]
    # ponytail: keeps the cheapest-per-trip multi-trip duties that fit the qubit budget; a full system
    # would generate duties by column generation and send QAOA only the promising ones.
    multi = sorted(multi, key=lambda c: paid_h(c) / len(c))[: max(MAX_DUTIES - n, 0)]
    duties = [[i] for i in range(n)] + multi  # single-trip duties keep every instance solvable
    cost = [paid_h(d) for d in duties]
    obj, pen = Qubo(len(duties)), Qubo(len(duties))
    for k, c in enumerate(cost):
        obj.add(k, k, c)
    A = sum(cost[:n]) + 1  # above the all-single-trip plan; costs >= 0
    for i in range(n):
        pen.square([(k, 1) for k, d in enumerate(duties) if i in d], 1, A)  # each trip worked exactly once

    def decode(b):
        chosen = [k for k in range(len(duties)) if b[k]]
        return {"duties": [duties[k] for k in chosen], "hours": [cost[k] for k in chosen],
                "crews": len(chosen), "paid_h": sum(cost[k] for k in chosen)}

    def describe(b):
        d = decode(b)
        return f"{d['crews']} crews, {d['paid_h']:.1f} paid hours"

    return Problem(f"Crew duty selection at {depot} depot", obj, pen, describe, None,
                   data={"trips": trips, "duties": duties, "cost": cost}, decode=decode)


# ------------------------------------------------------------------ depot allocation


def build_depots(depots, n_towns=4, seed=0, towns=None, fuel=1.0):
    """Choose which candidate depots to run and which depot houses the buses for each nearby town's
    services. Daily cost: each open depot's running cost + every bus running empty from its depot to
    its town in the morning and back at night, at `fuel` times today's cost per km. The towns are
    `towns`, else the n_towns nearest."""
    if len(depots) < 2:
        raise ValueError("pick at least 2 candidate depots")
    if towns is None:
        towns = sorted((s for s in STATIONS if s not in depots), key=lambda s: min(road_km(s, d) for d in depots))
        towns = towns[:n_towns]
    towns = list(towns)
    rng = np.random.default_rng(seed)
    buses = rng.integers(4, 13, len(towns))  # buses based at each town's services
    f = rng.integers(10, 31, len(depots)) * 1000.0  # Rs/day to run each depot
    c = np.array([[2 * buses[i] * road_km(t, d) * COST_PER_KM * fuel for d in depots]
                  for i, t in enumerate(towns)]).round()
    obj, pen, plan = facility_qubo(f, c)

    def decode(b):
        opened, serve = plan(b)
        return {"open": [depots[j] for j in opened], "serve": {t: depots[j] for t, j in zip(towns, serve)},
                "cost": f[opened].sum() + sum(c[i, j] for i, j in enumerate(serve))}

    def describe(b):
        d = decode(b)
        return f"open {', '.join(d['open'])} | Rs {d['cost']:,.0f}/day"

    return Problem(f"Depot allocation: {', '.join(depots)}", obj, pen, describe, None,
                   data={"f": f, "c": c, "towns": towns, "buses": buses}, decode=decode)


# ------------------------------------------------------------------ statewide decomposition

BUILDERS = {"transfer": build_transfer, "buses": build_buses, "crews": build_crews, "depots": build_depots}
HUB_GROUP = 4  # routes timed together at a hub: 4 routes x 4 slots = 16 qubits

# What-if days, and how each changes the plan's inputs. "surge": transfers at a hub multiply;
# "closed": depots with no buses or crews based there; "fuel": cost per empty km multiplies.
SCENARIOS = {
    "Normal day": {"about": "The day as the data describes it."},
    "Festival at Tirupati": {"surge": {"Tirupati": 3},
                             "about": "Brahmotsavam pilgrims: three times as many passengers change buses at Tirupati."},
    "Guntur depot closed": {"closed": {"Guntur"},
                            "about": "No buses or crews are based at Guntur; the Krishna-Guntur region plans without it."},
    "Diesel up 40%": {"fuel": 1.4,
                      "about": "Each empty kilometre costs 40% more, so an extra bus becomes the cheaper fix more often."},
}


def statewide_pieces(seed=0, scenario="Normal day"):
    """The independent pieces of the state's daily schedule: [(kind, label, builder args)]. Every
    depot's bus and crew batch, and every region's depot allocation (its best-connected stations are
    the candidate depots). Hubs are timed separately, as a chain: see solve_hub_chain.

    ponytail: buses and crews aren't shared between neighbouring depots, and a town at a region's
    edge can't be served from the next region. Only hub timetables are coordinated across pieces."""
    what = SCENARIOS[scenario]
    closed, fuel = what.get("closed", set()), what.get("fuel", 1.0)
    pieces = []
    for depot in STATIONS:
        if depot_towns(depot) and depot not in closed:
            pieces += [("buses", depot, (depot, 8, seed, fuel)), ("crews", depot, (depot, 8, seed))]
    for region in dict.fromkeys(REGION.values()):
        members = sorted((s for s in STATIONS if REGION[s] == region), key=lambda s: (-len(routes_through(s)), s))
        k = min(3, max(2, len(members) // 2))
        cands = [s for s in members[:k] if s not in closed]
        towns = tuple(s for s in members if s not in cands)  # a closed depot's own town still needs serving
        if len(cands) >= 2 and towns:
            pieces.append(("depots", region, (tuple(cands), 0, seed, towns, fuel)))
    return pieces


def solve_piece(piece, reps=2, restarts=1, seed=0):
    """Build and QAOA-solve one piece -> (piece, Result or None, error or None, seconds)."""
    kind, _, args = piece
    try:
        prob = BUILDERS[kind](*args)
    except ValueError as e:
        return piece, None, str(e), 0.0
    if prob.n > MAX_QUBITS:
        return piece, None, f"{prob.n} qubits, over the {MAX_QUBITS}-qubit simulator limit", 0.0
    t0 = time.perf_counter()
    return piece, solve(prob, reps=reps, restarts=restarts, seed=seed), None, time.perf_counter() - t0


def statewide_size(seed=0, scenario="Normal day"):
    """How many results solve_statewide yields."""
    return len(statewide_pieces(seed, scenario)) + len(hub_schedule())


def solve_statewide(seed=0, reps=2, restarts=1, qseed=0, pieces=None, hubs=True, workers=None, scenario="Normal day"):
    """Solve the statewide plan in parallel processes: the hub chain as one job (its pieces depend on
    each other), every other piece as its own. Yields solve_piece results as they finish. `pieces`
    overrides statewide_pieces(seed, scenario); hubs=False skips the hub chain.
    Processes, not threads: QAOA's optimizer loop is Python-bound, and threads measured only 1.5x."""
    pieces = statewide_pieces(seed, scenario) if pieces is None else pieces
    jobs = ([(solve_hub_chain, (seed, reps, restarts, qseed, scenario))] if hubs else []) + \
           [(solve_piece, (p, reps, restarts, qseed)) for p in pieces]  # the long hub chain starts first
    # Workers inherit this before importing numpy. Left unset, OpenBLAS reserves ~3 GB of commit per
    # process for its thread buffers, and a few workers exhaust Windows' commit limit and crash.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    done = set()
    pool = ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) // 2))
    try:
        # Spawned workers re-import the caller's __main__ (under Streamlit: the whole dashboard) unless it
        # has no file; they only need this module, so hide it while submit starts them.
        main, sys.modules["__main__"] = sys.modules["__main__"], types.ModuleType("__main__")
        try:
            futures = {pool.submit(fn, *args): i for i, (fn, args) in enumerate(jobs)}
        finally:
            sys.modules["__main__"] = main
        for future in as_completed(futures):
            out = future.result()
            yield from out if isinstance(out, list) else [out]
            done.add(futures[future])
    except BrokenProcessPool as e:  # workers forbidden or killed (sandbox, memory): finish the rest in-process
        log.warning("worker processes failed (%s); solving the remaining %d jobs in this process",
                    e, len(jobs) - len(done))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # a stopped run doesn't wait for queued pieces
    for i, (fn, args) in enumerate(jobs):
        if i not in done:
            out = fn(*args)
            yield from out if isinstance(out, list) else [out]
