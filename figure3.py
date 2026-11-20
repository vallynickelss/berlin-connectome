"""
Figure 3: closing stations.

(a) Network efficiency left as stations close, at random versus the most
    central ones first (highest betweenness).
(b) The map with the worst single loss (Lichtenberg) closed: which stations
    are cut off, and which need a longer detour.

Put this file in the same folder as figure1.py and berlin_brain_network.py,
then run:  python figure3.py
It uses exactly the same network as Figure 1 and prints the numbers from your
text (89%, 35%, 11.7%, 8.7%, 7.6%) so you can check them.
"""

import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from figure1 import (ANALYSIS_SCRIPT, GTFS_DIR, NODE, EDGE, FONT,
                     load_from_analysis, build_berlin_network)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
OUT_FILE = "figure3_closing_stations"
MAX_CLOSED = 20        # x axis of panel (a)
RANDOM_RUNS = 200      # random closures are repeated this often and averaged
SEED = 1               # same random draws every time you run it

RANDOM_COL = "#7A7974"     # random closures: grey
TARGET_COL = "#4A9FE8"     # targeted closures: the blue of B in Figure 2
CUT_COL = "#D9483B"        # cut off from the rest of the network
DETOUR_COL = "#F2A65A"     # still reachable, but with a longer trip


def name(G, n):
    return str(G.nodes[n].get("name", n))


def efficiency_after(G, closed):
    H = G.copy()
    H.remove_nodes_from(closed)
    return nx.global_efficiency(H)


# ----------------------------------------------------------------------------
# (a) Random versus targeted closures
# ----------------------------------------------------------------------------
def closure_curves(G):
    base = nx.global_efficiency(G)
    steps = range(MAX_CLOSED + 1)

    # Targeted, as in your analysis: after every closure, find the station that is
    # now the most central (highest betweenness) and close that one next
    H = G.copy()
    targeted = [100.0]
    for _ in range(MAX_CLOSED):
        bt = nx.betweenness_centrality(H)
        H.remove_node(max(bt, key=bt.get))
        targeted.append(100 * nx.global_efficiency(H) / base)

    rng = random.Random(SEED)
    nodes = list(G)
    runs = []
    for _ in range(RANDOM_RUNS):
        order = rng.sample(nodes, MAX_CLOSED)
        runs.append([100 * efficiency_after(G, order[:k]) / base for k in steps])
    runs = np.array(runs)
    return np.array(steps), np.array(targeted), runs


def draw_curves(ax, steps, targeted, runs):
    mean = runs.mean(axis=0)
    lo, hi = np.percentile(runs, [5, 95], axis=0)
    ax.fill_between(steps, lo, hi, color=RANDOM_COL, alpha=0.15, linewidth=0)
    ax.plot(steps, mean, color=RANDOM_COL, lw=2.2)
    ax.plot(steps, targeted, color=TARGET_COL, lw=2.2)

    # Label the lines directly instead of a legend
    ax.text(steps[-1] + 0.4, mean[-1], "random\nclosures", color=RANDOM_COL,
            va="center", fontsize=10, fontweight="bold")
    ax.text(steps[-1] + 0.4, targeted[-1], "most central\nstations first", color=TARGET_COL,
            va="center", fontsize=10, fontweight="bold")

    # Mark the ten closures from the text
    ax.axvline(10, color=EDGE, lw=0.8, ls=":")
    for y, col in [(mean[10], RANDOM_COL), (targeted[10], TARGET_COL)]:
        ax.plot(10, y, "o", color=col, ms=6, zorder=3)
        ax.annotate(f"{y:.0f}%", (10, y), xytext=(8, 8), textcoords="offset points",
                    ha="left", color=col, fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Stations closed")
    ax.set_ylabel("Efficiency left (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------
# (b) Closing the single worst station
# ----------------------------------------------------------------------------
def single_losses(G):
    base = nx.global_efficiency(G)
    loss = {n: 100 * (1 - efficiency_after(G, [n]) / base) for n in G}
    return sorted(loss.items(), key=lambda x: -x[1])


def affected_stations(G, station):
    """Stations cut off from the main network, and stations whose average trip gets longer."""
    H = G.copy()
    H.remove_node(station)
    main = max(nx.connected_components(H), key=len)
    cut = set(H) - main

    before = dict(nx.all_pairs_shortest_path_length(G))
    after = dict(nx.all_pairs_shortest_path_length(H))
    detour = set()
    for n in main:
        others = [m for m in main if m != n]
        b = np.mean([before[n][m] for m in others])
        a = np.mean([after[n][m] for m in others])
        if a > b * 1.10:                 # trips at least 10% longer on average
            detour.add(n)
    return cut, detour


def draw_closure_map(ax, G, station, cut, detour):
    lon = nx.get_node_attributes(G, "lon")
    lat = nx.get_node_attributes(G, "lat")
    segments = [[(lon[u], lat[u]), (lon[v], lat[v])] for u, v in G.edges if station not in (u, v)]
    ax.add_collection(LineCollection(segments, colors=EDGE, linewidths=0.8, zorder=1))

    def dots(nodes, colour, size):
        nodes = list(nodes)
        if nodes:
            ax.scatter([lon[n] for n in nodes], [lat[n] for n in nodes],
                       s=size, c=colour, linewidths=0, zorder=2)

    dots(set(G) - cut - detour - {station}, NODE, 3)
    dots(detour, DETOUR_COL, 9)
    dots(cut, CUT_COL, 9)
    ax.scatter(lon[station], lat[station], s=90, marker="X", c="black", zorder=3)
    label = name(G, station).replace("S+U ", "").replace(" Bhf", "").replace(" (Berlin)", "")
    ax.annotate(label, (lon[station], lat[station]), xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    legend = [Line2D([], [], marker="X", ls="", color="black", label="closed", ms=8)]
    if cut:
        legend.append(Line2D([], [], marker="o", ls="", color=CUT_COL, label="cut off", ms=6))
    if detour:
        legend.append(Line2D([], [], marker="o", ls="", color=DETOUR_COL, label="longer trips", ms=6))
    ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=9)

    ax.set_aspect(1 / np.cos(np.radians(52.52)))
    ax.autoscale_view()
    ax.axis("off")


def make_figure(G):
    plt.rcParams["font.family"] = FONT
    print("Calculating closure curves (this takes a minute) ...")
    steps, targeted, runs = closure_curves(G)
    print(f"  10 random closures:  {runs.mean(axis=0)[10]:.0f}% left (your text: 89%)")
    print(f"  10 central closures: {targeted[10]:.0f}% left (your text: 35%)")

    print("Closing each station on its own ...")
    losses = single_losses(G)
    for n, l in losses[:3]:
        print(f"  {name(G, n)}: {l:.1f}% lost")
    print("  (your text: Lichtenberg 11.7%, Südkreuz 8.7%, Ostkreuz 7.6%)")
    worst = losses[0][0]
    cut, detour = affected_stations(G, worst)
    print(f"  Closing {name(G, worst)}: {len(cut)} stations cut off, {len(detour)} with longer trips")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.8),
                                     gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.35})
    draw_curves(ax_a, steps, targeted, runs)
    draw_closure_map(ax_b, G, worst, cut, detour)
    for ax, label in [(ax_a, "a"), (ax_b, "b")]:
        fig.text(ax.get_position().x0 - 0.04, 0.95, label, fontsize=14, fontweight="bold", color=NODE)

    fig.savefig(f"{OUT_FILE}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT_FILE}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_FILE}.png and {OUT_FILE}.pdf")


if __name__ == "__main__":
    if ANALYSIS_SCRIPT.exists():
        G = load_from_analysis(ANALYSIS_SCRIPT, GTFS_DIR)
    else:
        G = build_berlin_network(GTFS_DIR)
    print(f"Berlin: {G.number_of_nodes()} stations, {G.number_of_edges()} connections")
    make_figure(G)
