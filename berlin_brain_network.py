"""
Would Berlin be a good brain?
Network analysis of the Berlin U Bahn and S Bahn, using the same graph
measures that neuroscientists use on structural brain networks (connectomes).

WHAT YOU NEED
  Python 3 with: pandas, networkx, numpy, matplotlib
    pip install pandas networkx numpy matplotlib
  The VBB GTFS timetable, unzipped into one folder:
    https://unternehmen.vbb.de/digitale-services/datensaetze/  (GTFS Daten, Download)
  Optional: the VBB line colour file (Linienfarben) for nicer map colours.

HOW TO RUN
  python berlin_brain_network.py --gtfs path/to/gtfs_folder --colours path/to/linienfarben.csv

WHAT IT DOES, STEP BY STEP
  1. Keeps only U Bahn (BVG) and S Bahn (S Bahn Berlin GmbH) trips.
  2. Builds a network: every station is a node, two stations are linked
     if a train runs directly from one to the other.
  3. Removes "skip" links that only exist because of construction diversions.
  4. Finds the hubs (degree and betweenness centrality).
  5. Tests for a rich club against randomised networks.
  6. Tests whether Berlin is a small world (clustering and path length).
  7. Runs lesion experiments: close stations and measure what is lost.
  8. Draws the figure.

The randomisation steps take a few minutes. Results print to the screen and
the figure is saved as berlin_brain.png in the folder you run it from.
"""

import argparse
import math
import random
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# STEP 1: LOAD THE TIMETABLE AND KEEP ONLY U BAHN AND S BAHN

def load_rail_trips(gtfs):
    """Return one row per train stop for all U and S Bahn trips."""
    routes = pd.read_csv(f"{gtfs}/routes.txt")
    # In GTFS, route_type 400 is metro/U Bahn and 109 is suburban rail/S Bahn.
    # agency 796 is the BVG, agency 1 is S Bahn Berlin GmbH. We filter on both,
    # because a few S lines elsewhere in the VBB region belong to other companies.
    is_ubahn = (routes.route_type == 400) & (routes.agency_id == 796)
    is_sbahn = (routes.route_type == 109) & (routes.agency_id == 1)
    rail = routes[is_ubahn | is_sbahn][["route_id", "route_short_name"]]

    trips = pd.read_csv(f"{gtfs}/trips.txt", usecols=["route_id", "trip_id"])
    trips = trips[trips.route_id.isin(rail.route_id)]
    wanted = set(trips.trip_id)

    # stop_times.txt is huge (about 300 MB), so we read it in chunks and keep
    # only the rows that belong to U or S Bahn trips.
    parts = []
    for chunk in pd.read_csv(f"{gtfs}/stop_times.txt",
                             usecols=["trip_id", "stop_id", "stop_sequence"],
                             chunksize=1_000_000):
        parts.append(chunk[chunk.trip_id.isin(wanted)])
    stop_times = pd.concat(parts)
    stop_times = stop_times.merge(trips, on="trip_id").merge(rail, on="route_id")
    return stop_times


# STEP 2: BUILD THE NETWORK

def build_network(gtfs, stop_times):
    stops = pd.read_csv(f"{gtfs}/stops.txt", dtype=str)
    parent = dict(zip(stops.stop_id, stops.parent_station))
    name = dict(zip(stops.stop_id, stops.stop_name))
    lat = dict(zip(stops.stop_id, stops.stop_lat.astype(float)))
    lon = dict(zip(stops.stop_id, stops.stop_lon.astype(float)))

    def station_of(stop_id):
        # Each platform has its own stop_id. We merge platforms into their
        # station ("parent_station"), otherwise Alexanderplatz would count as
        # several separate nodes.
        p = parent.get(stop_id)
        if isinstance(p, str) and p:
            return p
        m = re.match(r"(de:\d+:\d+)", stop_id)
        return m.group(1) if m else stop_id

    st = stop_times.copy()
    st["station"] = st.stop_id.map(station_of)
    st = st.sort_values(["trip_id", "stop_sequence"])

    G = nx.Graph()
    # Walk along every trip and link each station to the next one.
    for (trip, line), g in st.groupby(["trip_id", "route_short_name"]):
        seq = list(g.station)
        for a, b in zip(seq, seq[1:]):
            if a == b:
                continue
            if not G.has_edge(a, b):
                G.add_edge(a, b, lines=set(), trips=0)
            G[a][b]["lines"].add(line)
            G[a][b]["trips"] += 1

    for n in G:
        G.nodes[n]["name"] = name.get(n, n)
        G.nodes[n]["lat"] = lat.get(n)
        G.nodes[n]["lon"] = lon.get(n)

    # STEP 3: REMOVE CONSTRUCTION SHORTCUTS
    # When a station is closed for works, trains skip it, and the timetable
    # then contains a direct link that jumps over that station. That shows up
    # as a triangle served by the same line. In each such triangle we delete
    # the longest side, which is the jump.
    def km(u, v):
        return math.hypot((lat[u] - lat[v]) * 111, (lon[u] - lon[v]) * 68)

    removed = True
    while removed:
        removed = False
        for a, b in list(G.edges()):
            for x in set(G[a]) & set(G[b]):
                if G[a][b]["lines"] & G[a][x]["lines"] & G[x][b]["lines"]:
                    u, v = max([(a, b), (a, x), (x, b)], key=lambda e: km(*e))
                    print("  removed construction shortcut:",
                          name.get(u, u), "to", name.get(v, v))
                    G.remove_edge(u, v)
                    removed = True
                    break
            if removed:
                break

    # Two interchanges are listed as separate stations in the data although
    # they are one transfer point. We merge them by hand.
    def find(text):
        return [n for n in G if text in G.nodes[n]["name"]]

    for pair in (find("Yorckstr"), [find("S Spandau Bhf")[0], find("Rathaus Spandau")[0]]):
        if len(pair) == 2:
            G = nx.contracted_nodes(G, pair[0], pair[1], self_loops=False)
    for n in G:
        G.nodes[n].pop("contraction", None)
    return G


def short(G, n):
    return G.nodes[n]["name"].replace(" (Berlin)", "").replace(" Bhf", "")


# STEP 4: HUBS

def hubs(G):
    deg = dict(G.degree())
    # Betweenness: the share of all shortest routes in the network that pass
    # through a station. High betweenness means a bottleneck.
    bt = nx.betweenness_centrality(G)
    print("\nNumber of neighbours per station:", sorted(Counter(deg.values()).items()))
    print("\nMost neighbours (degree):")
    for n, d in sorted(deg.items(), key=lambda x: -x[1])[:10]:
        print(f"  {d}  {short(G, n)}")
    print("\nMost shortest routes passing through (betweenness):")
    for n, b in sorted(bt.items(), key=lambda x: -x[1])[:10]:
        print(f"  {b:.3f}  {short(G, n)}")
    return deg, bt


def randomised(G, seed):
    """A random network with exactly the same number of neighbours per
    station. This is the standard null model in connectomics."""
    H = G.copy()
    nx.double_edge_swap(H, nswap=10 * H.number_of_edges(), max_tries=10**6, seed=seed)
    return H


# STEP 5: RICH CLUB

def rich_club(G, n_random=300):
    """For each degree k: of all possible links among stations with more than
    k neighbours, what fraction exists? Compared with randomised networks.
    A ratio above 1 means the hubs are more interconnected than chance."""
    real = nx.rich_club_coefficient(G, normalized=False)
    rand = [nx.rich_club_coefficient(randomised(G, i), normalized=False)
            for i in range(n_random)]
    deg = dict(G.degree())
    print("\nRich club (k, stations with more than k neighbours, ratio real/random, p):")
    for k in sorted(real):
        r = np.array([x.get(k, np.nan) for x in rand])
        mean = np.nanmean(r)
        p = np.mean(r >= real[k])  # share of random networks at least as rich
        n_k = sum(1 for v in deg.values() if v > k)
        ratio = real[k] / mean if mean > 0 else float("nan")
        print(f"  k={k}  n={n_k:3d}  ratio={ratio:.2f}  p={p:.3f}")


# STEP 6: SMALL WORLD

def small_world(G, n_random=30):
    """Brains combine high clustering (neighbours of a node are linked to
    each other) with short paths. We compare Berlin with random networks."""
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    Cr, Lr = [], []
    for i in range(n_random):
        H = randomised(G, 1000 + i)
        Cr.append(nx.average_clustering(H))
        if nx.is_connected(H):
            Lr.append(nx.average_shortest_path_length(H))
    print(f"\nClustering: Berlin {C:.3f}, random {np.mean(Cr):.3f}")
    print(f"Average stops between two stations: Berlin {L:.1f}, random {np.mean(Lr):.1f}")


# STEP 7: LESIONS

def lesions(G, steps=30, n_random=50):
    """Global efficiency = average of 1/(number of stops) over all station
    pairs. It drops when stations close. Targeted attack: always close the
    station with the highest betweenness. Random: close stations at random."""
    E0 = nx.global_efficiency(G)

    single = {n: 1 - nx.global_efficiency(G.subgraph([m for m in G if m != n])) / E0
              for n in G}
    print("\nClosing one single station, efficiency lost:")
    for n, v in sorted(single.items(), key=lambda x: -x[1])[:10]:
        print(f"  {100 * v:4.1f}%  {short(G, n)}")

    def attack(pick):
        H = G.copy()
        curve = [1.0]
        for _ in range(steps):
            H.remove_node(pick(H))
            curve.append(nx.global_efficiency(H) / E0)
        return curve

    targeted = attack(lambda H: max(nx.betweenness_centrality(H).items(),
                                    key=lambda x: x[1])[0])
    rand = np.mean([attack(lambda H, r=random.Random(i): r.choice(list(H)))
                    for i in range(n_random)], axis=0)
    for s in (5, 10, 20):
        print(f"  after {s} closures: hubs first {100 * targeted[s]:.0f}% left, "
              f"random {100 * rand[s]:.0f}% left")
    return targeted, rand


# STEP 8: FIGURE

def figure(G, bt, targeted, rand, colour_file=None):
    colours = {}
    if colour_file:
        c = pd.read_csv(colour_file, sep=";", encoding="latin1")
        colours = dict(zip(c.Linie, c.background_Hex))
    deg = dict(G.degree())
    pos = {n: (G.nodes[n]["lon"] * 0.61, G.nodes[n]["lat"]) for n in G}  # 0.61 corrects for latitude

    fig = plt.figure(figsize=(13, 7.2), facecolor="white")
    ax = fig.add_axes([0.01, 0.13, 0.6, 0.79])
    ax.set_axis_off()
    for a, b in G.edges():
        lines = sorted(G[a][b].get("lines", [])) or ["?"]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=colours.get(lines[0], "#bbbbbb"), lw=1.6, alpha=0.55, zorder=1)
    rich = [n for n in G if deg[n] > 3]
    rest = [n for n in G if deg[n] <= 3]
    ax.scatter([pos[n][0] for n in rest], [pos[n][1] for n in rest],
               s=[6 + 900 * bt[n] for n in rest], c="#9aa0a6", edgecolors="white", lw=0.6, zorder=2)
    ax.scatter([pos[n][0] for n in rich], [pos[n][1] for n in rich],
               s=[14 + 900 * bt[n] for n in rich], c="#1f2937", edgecolors="white", lw=1.2, zorder=3)
    for n in sorted(G, key=lambda n: -bt[n])[:8]:
        ax.annotate(short(G, n).replace("S+U ", "").replace("S ", "", 1), pos[n],
                    xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    ax.set_xlim(13.17 * 0.61, 13.63 * 0.61)
    ax.set_ylim(52.405, 52.6)
    ax.set_title("Berlin U and S Bahn as a network", loc="left", fontsize=13, fontweight="bold")

    bx = fig.add_axes([0.67, 0.27, 0.30, 0.6])
    x = range(len(targeted))
    bx.plot(x, [100 * v for v in rand], color="#9aa0a6", lw=2, label="random stations")
    bx.plot(x, [100 * v for v in targeted], color="#b42318", lw=2, label="busiest hubs first")
    bx.set_xlabel("Stations closed")
    bx.set_ylabel("Network efficiency left (%)")
    bx.set_ylim(0, 105)
    bx.legend(frameon=False)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.set_title("Lesion test", loc="left", fontsize=13, fontweight="bold")
    fig.savefig("berlin_brain.png", dpi=200)
    print("\nFigure saved as berlin_brain.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtfs", required=True, help="folder with the unzipped VBB GTFS files")
    ap.add_argument("--colours", help="optional VBB Linienfarben csv")
    args = ap.parse_args()

    print("Loading timetable ...")
    st = load_rail_trips(args.gtfs)
    print("Building network ...")
    G = build_network(args.gtfs, st)
    print(f"\n{G.number_of_nodes()} stations, {G.number_of_edges()} connections")
    deg, bt = hubs(G)
    rich_club(G)
    small_world(G)
    targeted, rand = lesions(G)
    figure(G, bt, targeted, rand, args.colours)
