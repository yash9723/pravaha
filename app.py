"""Pravaha: quantum-assisted bus planning for Andhra Pradesh.

    streamlit run app.py
"""
import time
from collections import Counter

import pandas as pd
import streamlit as st

import ap_transport as ap
import ui

st.set_page_config(page_title="Pravaha · bus planning for Andhra Pradesh", page_icon=":material/route:",
                   layout="wide")
ui.css()
st.logo(str(ui.ROOT / "assets" / "logo.svg"), icon_image=str(ui.ROOT / "assets" / "mark.svg"), size="large")
for key, value in {"reps": 2, "restarts": 3, "qseed": 0, "iseed": 2, "scenario": "Normal day", "runs": {}}.items():
    st.session_state.setdefault(key, value)


# ------------------------------------------------------------------ statewide plan


def overview():
    S = st.session_state
    scenario = S.scenario
    pieces, schedule = ap.statewide_pieces(S.iseed, scenario), ap.hub_schedule()
    total, count = len(pieces) + len(schedule), Counter(kind for kind, *_ in pieces)
    ui.page_header("Andhra Pradesh · statewide daily plan", f"Plan the state's bus day as {total} small quantum problems",
                   "No quantum computer today can schedule a whole state at once. Pravaha cuts the day into pieces of "
                   "at most 20 qubits, solves each one with QAOA on Qiskit's simulator, and checks every answer "
                   "against exhaustive search.")
    key = (scenario, S.iseed, S.reps, S.qseed)  # each scenario's plan is kept, so days can be compared
    saved = S.runs.get(key)
    report = ui.summarize(saved[0]) if saved else None
    if report:
        ui.state_kpis(report, saved[1])
        normal = S.runs.get(("Normal day", *key[1:])) if scenario != "Normal day" else None
        if normal:
            ui.section(f"{scenario}, against a normal day", "QAOA's statewide plan under each, same seeds")
            ui.compare_days(ui.summarize(normal[0]), report, scenario)
        elif scenario != "Normal day":
            st.caption("Plan a normal day too, with the same seeds, to see this scenario's change next to it.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        hubs = list(dict.fromkeys(h for h, *_ in schedule))
        if report:
            ui.section("Where QAOA matched the optimum", "hover a station for its pieces")
            ui.network_map(dots=[{"station": s, "color": ui.GOOD if all(v) else ui.SERIOUS, "radius": 5 + 2 * len(v),
                                  "label": f"{s}: {sum(v)} of {len(v)} pieces optimal"} for s, v in report["at"].items()],
                           labels=False, height=470)
            ui.legend([(ui.GOOD, "every piece at this station optimal", "dot"),
                       (ui.SERIOUS, "at least one piece above the optimum", "dot"),
                       (ui.MUTED, "bus corridor", "line")])
        else:
            ui.section("The network", f"{len(ap.STATIONS)} stations · {len(ap.ROUTES)} corridors · hover for names")
            ui.network_map(dots=[{"station": h, "color": ui.ACCENT, "radius": 9, "label": h} for h in hubs], height=470)
            ui.legend([(ui.ACCENT, "hub, timed in the chain", "dot"), (ui.INK2, "station and depot", "dot"),
                       (ui.MUTED, "bus corridor", "line")])
    with right:
        ui.section("What a run does", "pieces solved in parallel")
        with st.container(border=True, key="card-run"):
            st.selectbox("Scenario", list(ap.SCENARIOS), key="scenario")
            st.caption(ap.SCENARIOS[scenario]["about"])
            st.html(f"""<ol class="steps">
              <li><div><b>Time the hubs, in order</b><span><em>{len(schedule)}</em> pieces. {hubs[0]} first; each
                route's time then carries along the road to the next hub.</span></div></li>
              <li><div><b>Chain each depot's buses</b><span><em>{count['buses']}</em> batches of 8 trips: fewest buses
                and empty kilometres.</span></div></li>
              <li><div><b>Roster each depot's crews</b><span><em>{count['crews']}</em> batches: cover every trip once
                for the fewest paid hours.</span></div></li>
              <li><div><b>Place depots in each region</b><span><em>{count['depots']}</em> regions: which depots run,
                and which towns each serves.</span></div></li>
            </ol>""")
            run = st.button("Plan the whole state" if not report else "Plan again", type="primary",
                            icon=":material/play_arrow:" if not report else ":material/refresh:", width="stretch",
                            key="run-state")
            if report:
                slug = scenario.lower().replace(" ", "-").replace("%", "pc")
                st.download_button("Download this plan", ui.plan_zip(saved[0], report), width="stretch",
                                   file_name=f"pravaha-{slug}-seed{S.iseed}.zip", mime="application/zip",
                                   icon=":material/download:", on_click="ignore", key="download-plan",
                                   help="CSV files: hub timetables, bus duties, crew rosters, depot allocation and "
                                        "every piece's result.")
            st.caption("Takes one to two minutes with parallel worker processes, longer without. QAOA uses one "
                       "optimizer restart per piece here.")

    if run:
        with st.status(f"Planning the state: {scenario.lower()}...", expanded=True) as status:
            bar, done, t0 = st.progress(0.0), [], time.perf_counter()
            for out in ap.solve_statewide(S.iseed, reps=S.reps, restarts=1, qseed=S.qseed, scenario=scenario):
                done.append(out)
                kind, label, _ = out[0]
                bar.progress(len(done) / total, text=f"{len(done)} of {total} pieces · {ui.KINDS[kind][0]}: {label}")
            status.update(label=f"Planned {total} pieces in {time.perf_counter() - t0:.0f} s", state="complete")
        S.runs[key] = (done, time.perf_counter() - t0)
        st.rerun()

    if not report:
        ui.empty_state("No plan yet", "Run the plan to see statewide totals next to the exhaustive-search optimum, "
                                      "a map of where QAOA matched it, the coordinated hub timetable, and every piece.")
        return
    ui.section("Hub clock", "each route's arrival at each hub, minutes past every hour")
    ui.hub_clock(report["clock"])
    st.caption("The chain times Vijayawada first. Every later hub takes the routes already timed (orange: carried along "
               "the road from the hub named on hover) and times only its remaining routes around them.")
    ui.section("Every piece", f"{len(report['rows'])} pieces")
    ui.pieces_table(report["rows"])


# ------------------------------------------------------------------ drill-downs, one piece each


def hub_page():
    S = st.session_state
    ui.page_header("Drill-down · one hub", "When should each route reach the hub?",
                   "Routes meet at a hub once an hour. QAOA picks each route's arrival slot, one route per platform "
                   "slot, so passengers changing buses wait least. Here a hub is timed on its own; the statewide "
                   "plan also coordinates hubs along the roads.")
    with st.container(border=True, key="card-controls"):
        c = st.columns([1.1, 3, 1.1])
        hub = c[0].selectbox("Hub", ui.HUBS, key="t-hub")
        through = ap.routes_through(hub)
        picked = c[1].multiselect("Routes", through, default=through[:4], key=f"t-routes-{hub}", max_selections=5,
                                  format_func=lambda r: f"{r} · {ap.ROUTES[r]['name']}")
        slots = c[2].selectbox("Arrival slots per hour", [4, 5, 6], key="t-slots",
                               format_func=lambda s: f"{s}, every {60 // s} min")
        routes = tuple(r for r in through if r in picked)  # the hub's order, so a route keeps its colour
        prob, res, plan, secs = ui.run_bar("transfer", (hub, routes, slots, S.iseed))
    if not prob:
        return
    color = {r: ui.SERIES[i] if i < len(ui.SERIES) else ui.INK2 for i, r in enumerate(through)}
    if res:
        ui.scorecard("transfer", prob, res, plan, secs)
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section(f"Routes through {hub}", f"{len(routes)} of {len(through)} routes selected")
        ui.network_map([ui.route_path(r, color[r]) for r in routes],
                       [{"station": hub, "color": ui.INK, "radius": 9, "label": hub}],
                       focus=[s for r in routes for s in ap.ROUTES[r]["stops"]])
        ui.legend([(color[r], f"{r} · {ap.ROUTES[r]['name']}", "line") for r in routes])
    with right:
        if plan:
            ui.section("Timetable at the hub", "minutes past every hour")
            ui.hub_timetable(plan["arrive"], routes, color)
            ui.section("Transfers", "busiest first")
            st.dataframe(pd.DataFrame([{"from": a, "to": b, "passengers / h": p, "wait (min)": round(w)}
                                       for a, b, p, w in plan["transfers"]])
                         .sort_values("passengers / h", ascending=False), hide_index=True, height=250)
        else:
            ui.section("Passengers changing buses", "per hour, from the row's route to the column's")
            ui.demand_heatmap(prob.data["w"], routes)
            ui.empty_state("Not timed yet", "Run QAOA to place each route in an arrival slot and see every "
                                            "transfer's wait.")
    if res:
        ui.outcomes(prob, res)


def trip_table(trips):
    st.dataframe(pd.DataFrame([{"trip": t.name, "from": t.frm, "to": t.to, "departs": ap.clock(t.start),
                                "arrives": ap.clock(t.end)} for t in trips]), hide_index=True)


def depot_controls(prefix):
    with st.container(border=True, key=f"card-{prefix}"):
        c = st.columns([1, 3])
        depot = c[0].selectbox("Depot", ui.DEPOTS, index=ui.DEPOTS.index("Vijayawada"), key=f"{prefix}-depot")
        n_trips = c[1].slider("Trips in this batch", 6, 10, 8, key=f"{prefix}-trips",
                              help="Ten trips can need more than 20 qubits; the Run button says so if it does.")
        return depot, n_trips, ui.run_bar("buses" if prefix == "b" else "crews",
                                          (depot, n_trips, st.session_state.iseed))


def bus_page():
    ui.page_header("Drill-down · one depot", "Which bus runs which trip?",
                   "QAOA links a depot's trips into bus duties, so the depot puts as few buses on the road and runs "
                   "them empty as little as it can. One more bus costs as much as "
                   f"{ap.BUS_COST_KM} empty kilometres.")
    depot, _, (prob, res, plan, secs) = depot_controls("b")
    if not prob:
        return
    trips = prob.data["trips"]
    if res:
        ui.scorecard("buses", prob, res, plan, secs)
    left, right = st.columns([2, 3], gap="large")
    with left:
        ui.section("Trips from this depot", "a synthetic day, 05:30 to 15:00 departures")
        ui.network_map([{"path": [ui.lonlat(t.frm), ui.lonlat(t.to)], "color": ui.rgb(ui.SERIES[0]),
                         "name": f"{t.name} · {t.frm} → {t.to}"} for t in trips],
                       [{"station": depot, "color": ui.INK, "radius": 9, "label": f"{depot} depot"}],
                       focus=[depot] + [t.to for t in trips], height=400)
        ui.legend([(ui.INK, "depot", "dot"), (ui.SERIES[0], "trip", "line")])
    with right:
        if plan:
            ui.section("Bus duties", f"{plan['buses']} buses for {len(trips)} trips · {plan['empty_km']:,.0f} empty km")
            ui.gantt(trips, [(f"Bus {k + 1}", chain) for k, chain in enumerate(plan["blocks"])])
        else:
            ui.section("Trips to cover", f"{len(trips)} trips")
        trip_table(trips)
        if not plan:
            ui.empty_state("No duties yet", "Run QAOA to chain these trips into bus duties.")
    if res:
        ui.outcomes(prob, res)


def crew_page():
    ui.page_header("Drill-down · one depot", "Which crew works which trips?",
                   "From candidate duties (chains of trips one crew can work in a shift), QAOA picks a set that covers "
                   "every trip exactly once for the fewest paid hours. Each candidate duty is one qubit.")
    _, _, (prob, res, plan, secs) = depot_controls("c")
    if not prob:
        return
    trips, duties, cost = prob.data["trips"], prob.data["duties"], prob.data["cost"]
    if res:
        ui.scorecard("crews", prob, res, plan, secs)
    if plan:
        ui.section("Crew roster", f"{plan['crews']} crews · {plan['paid_h']:.1f} paid hours")
        ui.gantt(trips, [(f"Crew {k + 1} · {h:.1f} h", d) for k, (d, h) in enumerate(zip(plan["duties"], plan["hours"]))])
    chosen = plan["duties"] if plan else []
    ui.section("Candidate duties", f"QAOA chooses among {len(duties)}")
    st.dataframe(pd.DataFrame([{"chosen": d in chosen,
                                "duty": " → ".join(f"{trips[i].name} {trips[i].frm}–{trips[i].to}" for i in d),
                                "trips": len(d), "paid hours": round(h, 2)} for d, h in zip(duties, cost)]),
                 hide_index=True, column_config={"chosen": st.column_config.CheckboxColumn(width="small")})
    if not plan:
        ui.empty_state("No roster yet", "Run QAOA to pick the duties that cover every trip once.")
    if res:
        ui.outcomes(prob, res)


def depot_page():
    S = st.session_state
    ui.page_header("Drill-down · one region", "Which depots should run, and which towns should each serve?",
                   "Every open depot has a daily running cost, and every bus runs empty from its depot to its town each "
                   "morning and back each night. QAOA weighs the two.")
    with st.container(border=True, key="card-depots"):
        c = st.columns([3, 1])
        cands = c[0].multiselect("Candidate depots", list(ap.STATIONS), default=["Vijayawada", "Guntur", "Eluru"],
                                 max_selections=4, key="d-depots")
        n_towns = c[1].slider("Towns to serve", 2, 5, 4, key="d-towns")
        prob, res, plan, secs = ui.run_bar("depots", (tuple(cands), n_towns, S.iseed))
    if not prob:
        return
    towns, buses, f, cost = prob.data["towns"], prob.data["buses"], prob.data["f"], prob.data["c"]
    opened = plan["open"] if plan else []
    if res:
        ui.scorecard("depots", prob, res, plan, secs)
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("Depots and towns", "lines join each town to the depot that serves it" if plan else "")
        dots = [{"station": d, "color": ui.ACCENT if d in opened or not plan else ui.MUTED, "radius": 11, "label": d}
                for d in cands] + [{"station": t, "color": ui.INK2, "radius": 6, "label": t} for t in towns]
        paths = [{"path": [ui.lonlat(t), ui.lonlat(d)], "color": ui.rgb(ui.ACCENT), "name": f"{t} served by {d}"}
                 for t, d in plan["serve"].items()] if plan else []
        ui.network_map(paths, dots, focus=list(cands) + towns)
        ui.legend(([(ui.ACCENT, "open depot", "dot"), (ui.MUTED, "closed depot", "dot")] if plan
                   else [(ui.ACCENT, "candidate depot", "dot")]) + [(ui.INK2, "town", "dot")])
    with right:
        ui.section("Depots")
        st.dataframe(pd.DataFrame([{"depot": d, "running cost / day": ui.inr(f[j]),
                                    "status": ("open" if d in opened else "closed") if plan else "candidate"}
                                   for j, d in enumerate(cands)]), hide_index=True)
        ui.section("Towns", "buses based at each town's services")
        st.dataframe(pd.DataFrame([{"town": t, "buses": int(buses[i]), "served by": plan["serve"][t] if plan else "-",
                                    "empty running / day": ui.inr(cost[i, list(cands).index(plan["serve"][t])]
                                                                  if plan else cost[i].min())}
                                   for i, t in enumerate(towns)]), hide_index=True)
        if not plan:
            ui.empty_state("Not allocated yet", "Run QAOA to decide which depots open and which towns each one serves.")
    if res:
        ui.outcomes(prob, res)


# ------------------------------------------------------------------ method


def pipeline(seed):
    """The method as a Graphviz flow, with today's piece counts."""
    count = Counter(kind for kind, *_ in ap.statewide_pieces(seed))
    return f"""digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#fdfdfb", color="#e1e0d9", penwidth=1.2,
        fontname="Geist, Segoe UI, sans-serif", fontsize=11, fontcolor="#1b1b19", margin="0.2,0.12"];
  edge [color="#807e77", arrowsize=0.6, penwidth=1.1];
  day [label="Statewide day\\n{len(ap.STATIONS)} stations · {len(ap.ROUTES)} corridors"];
  hubs [label="Hub chain\\n{len(ap.hub_schedule())} pieces, one after another"];
  depots [label="Depot batches\\n{count['buses'] + count['crews']} pieces, in parallel"];
  regions [label="Regions\\n{count['depots']} pieces, in parallel"];
  qubo [label="QUBO\\nat most 20 qubits"];
  qaoa [label="QAOA\\nQiskit Aer simulator"];
  check [label="Exhaustive check\\nclassical fallback"];
  plan [label="Statewide plan", fillcolor="#e4edf8", color="#1c5cab", fontcolor="#1c5cab"];
  day -> {{hubs depots regions}}; {{hubs depots regions}} -> qubo -> qaoa -> check -> plan;
}}"""


def method_page():
    ui.page_header("How it works", "A statewide schedule, one small quantum problem at a time",
                   "What Pravaha computes, what it assumes, and where its limits are.")
    ui.section("The pipeline")
    st.graphviz_chart(pipeline(st.session_state.iseed), width="stretch")
    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Why pieces")
        ui.prose("<p>A day of statewide bus operations has thousands of trips. QAOA on a simulator handles about "
                 "<b>20 qubits</b>, and today's quantum hardware is not much larger in practice. So Pravaha works the "
                 "way depots already do: hub by hub, depot by depot, region by region.</p><p>Hubs are the exception "
                 "to independence. They are timed <b>in a chain</b>: once Vijayawada fixes a route's arrival time, "
                 "the road fixes its time at Guntur, and Guntur only times its remaining routes.</p>")
        ui.section("From timetable to qubits")
        ui.prose("<p>Each piece becomes a <b>QUBO</b>: a cost over yes/no variables. A qubit might mean "
                 "\"route NH16-S arrives at :15\", \"the bus from trip T3 runs T5 next\", or \"crew duty 7 is "
                 "worked\". Rules such as one slot per route become penalties large enough that breaking a rule "
                 "never pays.</p>")
    with right:
        ui.section("What QAOA does")
        ui.prose("<p>QAOA prepares a quantum state, alternates cost and mixing layers, and tunes their angles with a "
                 "classical optimizer (COBYLA, scoring the best quarter of outcomes). The tuned circuit is sampled "
                 "<b>1,024 times</b>; each sample is a candidate plan, and the best valid one is kept.</p>")
        ui.section("How every answer is checked")
        ui.prose("<p>At these sizes the optimum can still be found by trying every plan, so every piece is checked "
                 "against <b>exhaustive search</b>. The dashboard reports QAOA's plan and the optimum side by side. If "
                 "QAOA samples no valid plan, the classical answer is used and counted as a fallback.</p>")
    ui.section("Assumptions", "illustrative; set real figures in ap_transport.py")
    st.dataframe(pd.DataFrame([
        ("Average bus speed, stops included", f"{ap.SPEED_KMPH} km/h"),
        ("Road distance", f"straight line × {ap.CIRCUITY}"),
        ("Hub timetable cycle", f"{ap.CYCLE_MIN} min"),
        ("Dwell at a hub", f"{ap.DWELL_MIN} min"),
        ("Platform change", f"{ap.MIN_TRANSFER_MIN} min"),
        ("Bus layover between trips", f"{ap.MIN_LAYOVER_MIN} to {ap.MAX_LAYOVER_MIN} min"),
        ("One more bus in service", f"costs as much as {ap.BUS_COST_KM} empty km"),
        ("Crew sign-on and sign-off", f"{ap.SIGN_ON_OFF_H} h paid"),
        ("Longest crew duty", f"{ap.MAX_DUTY_H} h"),
        ("Empty running cost", f"₹{ap.COST_PER_KM} per km"),
    ], columns=["assumption", "value"]), hide_index=True)
    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Data")
        ui.prose("<p><b>data/stations.csv</b>: name, latitude, longitude and region for 25 stations (approximate "
                 "city coordinates). <b>data/routes.csv</b>: route id, name and stops for 15 corridors along real "
                 "highways, illustrative rather than APSRTC's actual routes. Passenger demand, trip times and costs "
                 "are synthetic, drawn from the scenario seed. Replace the CSVs with APSRTC data to plan for real.</p>")
    with right:
        ui.section("Limits")
        ui.prose("<p>At these sizes a classical solver finds the same plans instantly: this shows the architecture, "
                 "not a quantum speed-up. Buses and crews are not shared between neighbouring depots, and each crew "
                 f"batch offers QAOA only its {ap.MAX_DUTIES} cheapest candidate duties. Every piece runs on a "
                 "simulator; running on IBM quantum hardware means replacing one function in the engine.</p>")


st.navigation([
    st.Page(overview, title="Statewide plan", icon=":material/map:", url_path="statewide", default=True),
    st.Page(hub_page, title="Hub timing", icon=":material/schedule:", url_path="hubs"),
    st.Page(bus_page, title="Bus duties", icon=":material/directions_bus:", url_path="buses"),
    st.Page(crew_page, title="Crew rosters", icon=":material/badge:", url_path="crews"),
    st.Page(depot_page, title="Depots", icon=":material/warehouse:", url_path="depots"),
    st.Page(method_page, title="How it works", icon=":material/info:", url_path="method"),
], position="top").run()
