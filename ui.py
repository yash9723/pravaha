"""Building blocks for the Pravaha dashboard: theme, page chrome, maps, charts and result panels."""
import html
import io
import time
import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import ap_transport as ap
from qaoa_logistics import MAX_QUBITS, bits, solve

ROOT = Path(__file__).parent
# Reference categorical palette in its fixed, validated order: an entity keeps its colour everywhere.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT, GOOD, SERIOUS, MUTED, INK, INK2, CARD = "#1c5cab", "#0ca30c", "#ec835a", "#c3c2b7", "#1b1b19", "#52514e", "#fdfdfb"
DAY = pd.Timestamp("2026-01-01")
HUBS = [s for s in ap.STATIONS if len(ap.routes_through(s)) >= 3]
DEPOTS = [s for s in ap.STATIONS if ap.depot_towns(s)]
esc = html.escape


def inr(rupees):
    """Rupees the Indian way: in lakh from one lakh up."""
    return f"₹{rupees / 1e5:.2f} lakh" if rupees >= 1e5 else f"₹{rupees:,.0f}"


# What each kind of piece is called, the unit its plans are scored in, and a plan as a short value.
KINDS = {"transfer": ("Hub timing", "avg transfer wait", lambda d: f"{d['avg_wait']:.1f} min"),
         "buses": ("Bus duties", "buses · empty km", lambda d: f"{d['buses']} · {d['empty_km']:,.0f}"),
         "crews": ("Crew rosters", "crews · paid hours", lambda d: f"{d['crews']} · {d['paid_h']:.1f}"),
         "depots": ("Depots", "cost per day", lambda d: inr(d["cost"]))}


# ------------------------------------------------------------------ page chrome


def css():
    st.html(ROOT / "style.css")


def solver_settings():
    """Settings every page shares; the widgets keep their values in session state."""
    with st.popover("Solver settings", icon=":material/tune:", width="stretch"):
        st.slider("QAOA layers (p)", 1, 3, key="reps",
                  help="More layers can raise the chance of sampling the optimum, at 3-6x the run time.")
        st.slider("Optimizer restarts", 1, 5, key="restarts",
                  help="Random restarts per piece on the drill-down pages. The statewide plan uses one.")
        st.number_input("QAOA seed", 0, 999, key="qseed")
        st.number_input("Scenario seed", 0, 999, key="iseed",
                        help="Passenger demand, trips and costs are synthetic, drawn from this seed.")


def page_header(eyebrow, title, lede):
    left, right = st.columns([5, 1.25], vertical_alignment="top")
    left.html(f'<header class="page-head"><p class="eyebrow">{esc(eyebrow)}</p><h1>{esc(title)}</h1>'
              f'<p class="lede">{esc(lede)}</p></header>')
    with right:
        solver_settings()


def section(title, note=""):
    st.html(f'<div class="section-title"><h2>{esc(title)}</h2><p>{esc(note)}</p></div>')


def chips(items):
    """items: [(text, tone)], tone one of '', 'accent', 'good', 'warn'."""
    st.html('<div class="chips">' + "".join(f'<span class="chip {t}">{esc(x)}</span>' for x, t in items) + "</div>")


def legend(items):
    """items: [(colour, label, 'dot' or 'line')]: a swatch plus words, so colour never works alone."""
    st.html('<div class="legend">' + "".join(f'<span><i class="{shape}" style="background:{c}"></i>{esc(label)}</span>'
                                             for c, label, shape in items) + "</div>")


def empty_state(title, body):
    st.html(f'<div class="empty"><h3>{esc(title)}</h3><p>{esc(body)}</p></div>')


def prose(markup):
    """Trusted, hand-written HTML paragraphs (never user input)."""
    st.html(f'<div class="prose">{markup}</div>')


# ------------------------------------------------------------------ maps


def rgb(hex_color):
    return [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]


def lonlat(station):
    lat, lon = ap.STATIONS[station]
    return [lon, lat]


def route_path(rid, color):
    return {"path": [lonlat(s) for s in ap.ROUTES[rid]["stops"]], "color": rgb(color),
            "name": f"{rid} · {ap.ROUTES[rid]['name']}"}


def network_map(paths=(), dots=(), focus=(), labels=True, height=440):
    """AP corridors in grey, `paths` drawn over them, `dots` (labelled unless labels=False; the label is
    always the tooltip), zoomed to fit the `focus` stations."""
    base = [route_path(rid, MUTED) for rid in ap.ROUTES]
    towns = [{"pos": lonlat(s), "name": s} for s in ap.STATIONS]
    pts, above = [], []
    for d in dots:  # a label goes above its dot, or below when a nearby label already sits above
        pos = lonlat(d["station"])
        near = any(abs(pos[0] - q[0]) < 0.6 and abs(pos[1] - q[1]) < 0.35 for q in above)
        pts.append({"pos": pos, "name": d["label"], "color": rgb(d["color"]), "radius": d["radius"],
                    "offset": [0, d["radius"] + 12] if near else [0, -d["radius"] - 11]})
        if not near:
            above.append(pos)
    px = "'pixels'"  # pydeck reads bare strings as data expressions; quoting makes it the literal unit
    layers = [
        pdk.Layer("PathLayer", base, get_path="path", get_color="color", get_width=2, width_units=px, pickable=True),
        pdk.Layer("PathLayer", list(paths), get_path="path", get_color="color", get_width=5, width_units=px,
                  pickable=True, cap_rounded=True, joint_rounded=True),
        pdk.Layer("ScatterplotLayer", towns, get_position="pos", get_fill_color=rgb(INK2), get_radius=3.5,
                  radius_units=px, pickable=True),
        pdk.Layer("ScatterplotLayer", pts, get_position="pos", get_fill_color="color", get_radius="radius",
                  radius_units=px, stroked=True, get_line_color=rgb(CARD), line_width_min_pixels=2, pickable=True),
        pdk.Layer("TextLayer", pts if labels else [], get_position="pos", get_text="name", get_size=12,
                  get_color=rgb(INK), get_pixel_offset="offset", background=True, get_background_color=rgb(CARD),
                  background_padding=[4, 2], font_family="'Geist, Segoe UI, sans-serif'", font_weight=600),
    ]
    lats, lons = zip(*(ap.STATIONS[s] for s in focus or ap.STATIONS))
    span = max(max(lats) - min(lats), max(lons) - min(lons), 0.3)  # degrees the view must fit
    view = pdk.ViewState(latitude=(max(lats) + min(lats)) / 2, longitude=(max(lons) + min(lons)) / 2,
                         zoom=float(np.clip(np.log2(360 / span) - 0.6, 5, 9)))
    tooltip = {"html": "{name}", "style": {"background": INK, "color": CARD, "fontSize": "12px", "padding": "6px 9px",
                                           "borderRadius": "6px", "fontFamily": "Geist, 'Segoe UI', sans-serif"}}
    # The wheel scrolls the page, not the map: zoom with the +/- buttons, a double-click, or a pinch.
    controls = pdk.View(type="MapView", controller={"scrollZoom": False, "dragRotate": False, "touchRotate": False})
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, views=[controls], map_style=None,
                             tooltip=tooltip), height=height)


# ------------------------------------------------------------------ charts


def gantt(trips, rows):
    """A day timeline: one row per bus or crew, given as (label, trip indices). Trips long enough to hold
    their name are labelled; every trip's details are on hover."""
    df = pd.DataFrame([{"row": label, "trip": trips[i].name, "route": f"{trips[i].frm} → {trips[i].to}",
                        "start": DAY + pd.Timedelta(minutes=trips[i].start),
                        "end": DAY + pd.Timedelta(minutes=trips[i].end), "minutes": trips[i].end - trips[i].start,
                        "time": f"{ap.clock(trips[i].start)}–{ap.clock(trips[i].end)}"}
                       for label, chain in rows for i in chain])
    y = alt.Y("row:N", sort=None, title=None, axis=alt.Axis(labelLimit=180))
    bars = alt.Chart(df).mark_bar(cornerRadius=4, color=SERIES[0], height=alt.RelativeBandSize(0.62)).encode(
        x=alt.X("start:T", axis=alt.Axis(format="%H:%M", title=None, grid=True)), x2="end:T", y=y,
        tooltip=["row", "trip", "route", "time"])
    names = alt.Chart(df[df.minutes >= 40]).mark_text(align="left", dx=6, color=CARD, fontSize=11, fontWeight=600) \
        .encode(x="start:T", y=y, text="trip")
    st.altair_chart((bars + names).properties(height=46 * len(rows) + 36), width="stretch")


def hub_timetable(arrive, routes, color):
    """When each route stands at the hub, as bars across one hour."""
    df = pd.DataFrame([{"route": r, "arrives": arrive[r], "departs": arrive[r] + ap.DWELL_MIN,
                        "at": f":{int(arrive[r]):02d}–:{int(arrive[r] + ap.DWELL_MIN) % 60:02d}"} for r in routes])
    chart = alt.Chart(df).mark_bar(cornerRadius=4, height=alt.RelativeBandSize(0.6)).encode(
        x=alt.X("arrives:Q", scale=alt.Scale(domain=[0, 60]), axis=alt.Axis(values=[0, 15, 30, 45, 60], title=None)),
        x2="departs:Q", y=alt.Y("route:N", sort=list(routes), title=None),
        color=alt.Color("route:N", scale=alt.Scale(domain=list(routes), range=[color[r] for r in routes]), legend=None),
        tooltip=["route", alt.Tooltip("at", title="at the hub")])
    st.altair_chart(chart.properties(height=38 * len(routes) + 36), width="stretch")


def demand_heatmap(w, routes):
    """Passengers per hour changing from each row's route to each column's: one hue, light to dark, with
    the number in every cell (so the colour never has to be read on its own)."""
    df = pd.DataFrame([{"from": a, "to": b, "passengers": w[a, b]} for a in routes for b in routes if (a, b) in w])
    base = alt.Chart(df).encode(
        x=alt.X("to:N", sort=list(routes), title="to", axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("from:N", sort=list(routes), title="from"))
    cells = base.mark_rect(cornerRadius=4, stroke=CARD, strokeWidth=3).encode(
        color=alt.Color("passengers:Q", scale=alt.Scale(domain=[0, 60], range=["#e4edf8", ACCENT]), legend=None),
        tooltip=["from", "to", alt.Tooltip("passengers", title="passengers / h")])
    nums = base.mark_text(fontSize=12, fontWeight=600).encode(
        text="passengers:Q", color=alt.condition(alt.datum.passengers > 34, alt.value(CARD), alt.value(INK)))
    st.altair_chart((cells + nums).properties(height=44 * len(routes) + 50), width="stretch")


def hub_clock(rows):
    """The coordinated timetable: every route's arrival at every hub, minutes past the hour. A time chosen
    at a hub and a time carried there along the road get different colours (and words in the legend)."""
    df = pd.DataFrame(rows)
    lane = {}  # each hub's dots go in the first lane where the previous label (about 9 minutes wide) has ended
    for _, g in df.groupby("hub"):
        ends = []
        for i, m in g.minute.sort_values().items():
            j = next((j for j, end in enumerate(ends) if m >= end), len(ends))
            ends[j:j + 1] = [m + 9]
            lane[i] = str(j)
    df["lane"] = pd.Series(lane)
    order = list(dict.fromkeys(r["hub"] for r in rows))
    base = alt.Chart(df).encode(
        x=alt.X("minute:Q", scale=alt.Scale(domain=[0, 60]),
                axis=alt.Axis(values=[0, 15, 30, 45, 60], title="minutes past every hour")),
        y=alt.Y("hub:N", sort=order, title=None, axis=alt.Axis(labelFontWeight=600, labelPadding=10)),
        yOffset=alt.YOffset("lane:N", scale=alt.Scale(paddingOuter=0.6)))
    dots = base.mark_circle(size=130, opacity=1, stroke=CARD, strokeWidth=2).encode(
        color=alt.Color("how:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["chosen at this hub", "carried along the road"],
                                        range=[SERIES[0], SERIES[1]])),
        tooltip=["hub", "route", alt.Tooltip("at", title="arrives"), "how", "from"])
    names = base.mark_text(align="left", dx=9, fontSize=10.5, color=INK2).encode(text="route")
    st.altair_chart((dots + names).properties(height=86 * len(order) + 50), width="stretch")


def outcomes(prob, res):
    """Where QAOA's shots land: how many are usable plans, and the valid plans it favours most."""
    valid_idx = np.flatnonzero(res.valid)
    top = valid_idx[np.argsort(res.probs[valid_idx])[::-1][:10]]
    df = pd.DataFrame({
        "plan": [f"#{i + 1}" for i in range(len(top))],
        "status": ["optimal" if res.is_opt[k] else "valid" for k in top],
        "detail": [prob.describe(bits(k, prob.n)) for k in top],
        "probability": res.probs[top],
        "bitstring": ["".join(map(str, bits(k, prob.n))) for k in top],
    })
    section("What QAOA measured", f"{res.valid @ res.probs:.0%} of shots were valid plans · random guessing: "
                                  f"{res.valid.mean():.2%}")
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=alt.RelativeBandSize(0.42)).encode(
        x=alt.X("plan:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("probability:Q", axis=alt.Axis(format=".2%", tickCount=4), title=None),
        color=alt.Color("status:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["optimal", "valid"], range=[GOOD, SERIES[0]])),
        tooltip=["plan", "status", "detail", alt.Tooltip("probability:Q", format=".2%"), "bitstring"])
    st.caption("Each shot of the final circuit is one candidate plan; shots that break a rule are discarded. "
               "These are the valid plans QAOA sampled most often.")
    st.altair_chart(chart.properties(height=220), width="stretch")
    with st.expander("Table of these plans"):
        st.dataframe(df, hide_index=True, column_config={"probability": st.column_config.NumberColumn(format="percent")})


# ------------------------------------------------------------------ one piece at a time


@st.cache_data(show_spinner=False, max_entries=64)
def run_qaoa(kind, args, reps, restarts, seed):
    prob = ap.BUILDERS[kind](*args)
    t0 = time.perf_counter()
    res = solve(prob, reps=reps, restarts=restarts, seed=seed)
    return res, time.perf_counter() - t0


def run_bar(kind, args):
    """Build the piece and offer a Run button beside its size. Returns (prob, res, plan, seconds);
    res, plan and seconds stay None until QAOA has run with the current inputs and settings."""
    S = st.session_state
    try:
        prob = ap.BUILDERS[kind](*args)
    except ValueError as e:
        msg = str(e)
        st.warning(msg[0].upper() + msg[1:] + ".", icon=":material/error:")
        return None, None, None, None
    over = prob.n > MAX_QUBITS
    key = (kind, args, S.reps, S.restarts, S.qseed)
    button, facts = st.columns([1, 3], vertical_alignment="center")
    if button.button("Run QAOA", key=f"run-{kind}", type="primary", icon=":material/play_arrow:", disabled=over,
                     width="stretch"):
        S[f"ran-{kind}"] = key
    with facts:
        chips([(f"{prob.n} qubits", "accent"), (f"{2 ** prob.n:,} possible plans", "")]
              + ([(f"over the {MAX_QUBITS}-qubit simulator limit: make it smaller", "warn")] if over else []))
    if S.get(f"ran-{kind}") != key:
        return prob, None, None, None
    with st.spinner("Running QAOA on Qiskit Aer..."):
        res, secs = run_qaoa(*key)
    return prob, res, (prob.decode(bits(res.found, prob.n)) if res.found is not None else None), secs


def scorecard(kind, prob, res, plan, secs):
    """QAOA's plan against the exhaustive-search optimum, and how much better than chance QAOA did."""
    _, unit, fmt = KINDS[kind]
    best = prob.decode(bits(res.opt, prob.n))
    p_opt, p_rand = res.is_opt @ res.probs, res.is_opt.mean()
    c = st.columns(4)
    c[0].metric(f"QAOA plan · {unit}", fmt(plan) if plan else "no valid plan", border=True,
                delta="matches the optimum" if res.ratio == 1 else f"{1 / res.ratio - 1:.1%} above the optimum",
                delta_color="normal" if res.ratio == 1 else "inverse", delta_arrow="off")
    c[1].metric(f"Best possible · {unit}", fmt(best), "exhaustive search", delta_color="off", delta_arrow="off",
                border=True)
    c[2].metric("Chance per shot of the optimum", f"{p_opt:.1%}", f"{p_opt / p_rand:,.0f}× a random guess",
                delta_color="off", delta_arrow="off", border=True)
    c[3].metric("Problem size", f"{prob.n} qubits", f"solved in {secs:.0f} s", delta_color="off", delta_arrow="off",
                border=True)


# ------------------------------------------------------------------ the whole state


def summarize(results):
    """Fold statewide results into totals for QAOA's plan and for the exhaustive optimum, a table row per
    piece, the outcomes at each station, and the coordinated hub clock."""
    keys = ["pax_min", "pax", "buses", "empty_km", "crews", "paid_h", "cost"]
    qaoa, best = dict.fromkeys(keys, 0.0), dict.fromkeys(keys, 0.0)
    rows, at, clock, timed_at = [], {}, [], {}
    n_opt = n_fallback = n_skipped = 0
    for (kind, label, args), res, err, _ in sorted(results, key=lambda r: r[0][0] != "transfer"):  # chain order kept
        if err:
            n_skipped += 1
            rows.append({"kind": KINDS[kind][0], "where": label, "plan": err, "result": "skipped"})
            continue
        prob = ap.BUILDERS[kind](*args)
        k = res.found if res.found is not None else res.opt  # no valid sample: use the classical optimum
        for tot, state in ((qaoa, k), (best, res.opt)):
            plan = prob.decode(bits(state, prob.n))
            for key in keys:
                tot[key] += plan.get(key, 0)
        optimal = res.ratio == 1
        n_opt += optimal
        n_fallback += res.found is None
        p_opt = res.is_opt @ res.probs
        rows.append({"kind": KINDS[kind][0], "where": label, "qubits": prob.n, "plan": prob.describe(bits(k, prob.n)),
                     "result": "optimal" if optimal else "classical fallback" if res.found is None
                     else f"{1 / res.ratio - 1:.1%} above optimum",
                     "P(optimum)": p_opt, "vs random": p_opt / res.is_opt.mean()})
        if kind != "depots":
            at.setdefault(args[0], []).append(optimal)
        if kind == "transfer":
            hub, free = args[0], args[1]
            timed_at |= {r: hub for r in free}
            for r, minute in prob.decode(bits(k, prob.n))["arrive"].items():
                if r in free or timed_at.get(r) != hub:  # an earlier platform group's routes are already shown
                    clock.append({"hub": hub, "route": r, "minute": minute, "at": f":{int(minute):02d}",
                                  "how": "chosen at this hub" if r in free else "carried along the road",
                                  "from": "" if r in free else timed_at.get(r, "")})
    return {"qaoa": qaoa, "best": best, "rows": rows, "at": at, "clock": clock, "n_opt": n_opt,
            "n_fallback": n_fallback, "n_skipped": n_skipped, "solved": len(results) - n_skipped}


def state_kpis(report, secs):
    q, b = report["qaoa"], report["best"]
    c = st.columns(5)
    c[0].metric("Matched the optimum", f"{report['n_opt']} of {report['solved']}",
                f"{report['n_fallback']} fallbacks · {report['n_skipped']} skipped", delta_color="off",
                delta_arrow="off", border=True, help="Pieces where QAOA's plan equals the exhaustive-search optimum. "
                "Fallback: QAOA sampled no valid plan, so the classical answer was used.")
    for col, label, fmt in [
        (c[1], "Average transfer wait", lambda t: f"{t['pax_min'] / t['pax']:.1f} min"),
        (c[2], "Buses · empty km", lambda t: f"{t['buses']:.0f} · {t['empty_km']:,.0f}"),
        (c[3], "Crews · paid hours", lambda t: f"{t['crews']:.0f} · {t['paid_h']:,.1f}"),
        (c[4], "Depot cost per day", lambda t: inr(t["cost"])),
    ]:
        col.metric(label, fmt(q), f"optimum {fmt(b)}", delta_color="off", delta_arrow="off", border=True)
    st.caption(f"Planned in {secs:.0f} s. Each total adds up QAOA's plan for every piece; \"optimum\" is the same "
               "total with every piece solved by exhaustive search.")


def compare_days(base, now, scenario):
    """Statewide totals of QAOA's plan on a normal day against the same totals under `scenario`."""
    def totals(report):
        q = report["qaoa"]
        return {"Average transfer wait (min)": q["pax_min"] / q["pax"], "Passengers changing buses per hour": q["pax"],
                "Buses in service": q["buses"], "Empty kilometres": q["empty_km"], "Crew duties": q["crews"],
                "Paid crew hours": q["paid_h"], "Depot cost (₹ per day)": q["cost"]}
    a, b = totals(base), totals(now)

    def change(k):
        x = (b[k] - a[k]) / a[k] if a[k] else 0.0
        return f"{x:+.1%}" if abs(x) >= 5e-4 else "no change"

    digits = {"Average transfer wait (min)": 1}  # everything else reads best as whole numbers
    st.dataframe(pd.DataFrame([{"measure": k, "normal day": round(a[k], digits.get(k, 0)),
                                scenario: round(b[k], digits.get(k, 0)), "change": change(k)} for k in a]),
                 hide_index=True, column_config={"normal day": st.column_config.NumberColumn(format="localized"),
                                                 scenario: st.column_config.NumberColumn(format="localized")})


def plan_zip(results, report):
    """The plan as CSV files in a zip, ready for a depot's spreadsheets: hub timetables, bus duties,
    crew rosters, depot allocation, and one row per piece."""
    tables = {"hub_timetables": [{"hub": r["hub"], "route": r["route"], "arrives": r["at"],
                                  "departs": f":{(int(r['minute']) + ap.DWELL_MIN) % 60:02d}", "time set": r["how"],
                                  "carried from": r["from"]} for r in report["clock"]],
              "bus_duties": [], "crew_rosters": [], "depot_allocation": []}
    for (kind, label, args), res, err, _ in results:
        if err or kind == "transfer":
            continue
        prob = ap.BUILDERS[kind](*args)
        plan = prob.decode(bits(res.found if res.found is not None else res.opt, prob.n))
        trips = prob.data.get("trips")

        def trip(i):
            return {"trip": trips[i].name, "from": trips[i].frm, "to": trips[i].to, "departs": ap.clock(trips[i].start),
                    "arrives": ap.clock(trips[i].end)}

        if kind == "buses":
            tables["bus_duties"] += [{"depot": label, "bus": n, **trip(i)}
                                     for n, chain in enumerate(plan["blocks"], 1) for i in chain]
        elif kind == "crews":
            tables["crew_rosters"] += [{"depot": label, "crew": n, "paid hours": round(h, 2), **trip(i)}
                                       for n, (duty, h) in enumerate(zip(plan["duties"], plan["hours"]), 1) for i in duty]
        else:
            tables["depot_allocation"] += [{"region": label, "town": t, "served by": d, "open depots": ", ".join(plan["open"])}
                                           for t, d in plan["serve"].items()]
    tables["pieces"] = report["rows"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, rows in tables.items():  # utf-8-sig so Excel opens the ₹ and → characters correctly
            z.writestr(f"{name}.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"))
    return buf.getvalue()


def pieces_table(rows):
    left, right = st.columns([3, 2], vertical_alignment="center")
    show = left.segmented_control("Show", ["All", *[k[0] for k in KINDS.values()]], default="All",
                                  key="piece-kind", label_visibility="collapsed")
    missed = right.toggle("Only pieces where QAOA missed the optimum", key="piece-missed")
    df = pd.DataFrame(rows)
    if show and show != "All":
        df = df[df.kind == show]
    if missed:
        df = df[df.result != "optimal"]
    if df.empty:
        empty_state("Nothing to show", "QAOA matched the optimum on every piece in this view.")
        return
    st.dataframe(df, hide_index=True, height=min(36 * len(df) + 40, 520), column_config={
        "kind": "piece", "qubits": st.column_config.NumberColumn(format="%d", width="small"),
        "plan": st.column_config.TextColumn(width="large"), "result": st.column_config.TextColumn(width="medium"),
        "P(optimum)": st.column_config.NumberColumn(format="percent", help="Chance per shot of sampling the optimum"),
        "vs random": st.column_config.NumberColumn(format="%.0f×", help="P(optimum) over a random guess's chance")})
