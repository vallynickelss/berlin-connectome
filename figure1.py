"""
Figure 1: Berlin's U and S Bahn network next to a human brain network.

Left:  stations at their real coordinates (VBB GTFS), linked when a train runs
       directly between them.
Right: a human structural connectome (Griffa et al., 2019, Lausanne atlas)
       on a glass brain, drawn in the same style.

Usage:
    pip install pandas networkx matplotlib nilearn netneurotools
    python figure1.py

Set GTFS_DIR below to the folder with your unzipped VBB GTFS files.
The script prints the number of stations and connections, so you can check
them against your analysis (309 and 361).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
GTFS_DIR = Path(".")               # folder with routes.txt, trips.txt, stops.txt, stop_times.txt
OUT_FILE = "figure1_berlin_vs_brain"  # saved as .png (for checking) and .pdf (for print)

# Your analysis script. If it sits next to this file, Figure 1 uses exactly the
# network from your analysis (309 stations, 361 connections).
ANALYSIS_SCRIPT = Path("berlin_brain_network.py")

# Used only if the analysis script is not found: the same filters as your analysis
U_AGENCY, S_AGENCY = "796", "1"     # BVG and S Bahn Berlin GmbH
SKIP_LINKS = [                      # the 6 construction shortcuts your analysis removed
    ("Schönholz", "Bornholmer Str."),
    ("Köpenick", "Friedrichshagen"),
    ("Rosa-Luxemburg-Platz", "Eberswalder Str."),
    ("Dahlem-Dorf", "Breitenbachplatz"),
    ("Spichernstr.", "Berliner Str."),
    ("Leopoldplatz", "Rehberge"),
]

BRAIN_DATASET = "human_struct_scale033"  # 83 regions; scale060 has more detail
BRAIN_EDGE_SHARE = 0.20                  # show the strongest 20% of brain connections

# Same colours as your other figures
NODE = "#3D3D3A"
EDGE = "#B4B2A9"
FONT = "DejaVu Sans"                     # swap for the font CNS uses


# ----------------------------------------------------------------------------
# 1. Build the Berlin network from GTFS
# ----------------------------------------------------------------------------
def clean_name(name):
    """'S+U Alexanderplatz Bhf (Berlin) [U2]' -> 'Alexanderplatz'"""
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"^(S\+U|S|U)\s+", "", name.strip())
    name = re.sub(r"\s+Bhf\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def load_from_analysis(script, gtfs_dir):
    """Build the network with the functions from your own analysis script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("analysis", script)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    st = analysis.load_rail_trips(str(gtfs_dir))
    G = analysis.build_network(str(gtfs_dir), st)
    add_coordinates(G, gtfs_dir)
    return G


def add_coordinates(G, gtfs_dir):
    """Make sure every station has 'lon' and 'lat', whatever the analysis called them."""
    if all("lon" in d and "lat" in d for _, d in G.nodes(data=True)):
        return
    stops = pd.read_csv(gtfs_dir / "stops.txt", dtype=str)
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)
    by_id = stops.set_index("stop_id")[["stop_lon", "stop_lat"]]
    by_name = stops.groupby("stop_name")[["stop_lon", "stop_lat"]].mean()
    missing = []
    for n, d in G.nodes(data=True):
        if "pos" in d:
            d["lon"], d["lat"] = d["pos"]
        elif "x" in d and "y" in d:
            d["lon"], d["lat"] = d["x"], d["y"]
        elif str(n) in by_id.index:
            d["lon"], d["lat"] = by_id.loc[str(n)]
        elif d.get("name") in by_name.index:
            d["lon"], d["lat"] = by_name.loc[d["name"]]
        elif n in by_name.index:
            d["lon"], d["lat"] = by_name.loc[n]
        else:
            missing.append(n)
    if missing:
        raise KeyError(f"No coordinates found for {len(missing)} stations, e.g. {missing[:3]}")


def build_berlin_network(gtfs_dir):
    routes = pd.read_csv(gtfs_dir / "routes.txt", dtype=str)
    agency = routes["agency_id"] if "agency_id" in routes else pd.Series("", index=routes.index)
    is_u = (routes["route_type"] == "400") & (agency == U_AGENCY)    # 400 = U Bahn, run by BVG
    is_s = (routes["route_type"] == "109") & (agency == S_AGENCY)    # 109 = S Bahn, run by S Bahn Berlin
    route_ids = set(routes.loc[is_u | is_s, "route_id"])

    trips = pd.read_csv(gtfs_dir / "trips.txt", usecols=["route_id", "trip_id"], dtype=str)
    trip_ids = set(trips.loc[trips["route_id"].isin(route_ids), "trip_id"])

    stops = pd.read_csv(gtfs_dir / "stops.txt", dtype=str)
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)
    stops["station"] = stops["stop_name"].map(clean_name)     # merges S and U platforms of one station
    station_of = dict(zip(stops["stop_id"], stops["station"]))

    # stop_times.txt is big, so read it in chunks and keep only S and U Bahn trips
    parts = []
    for chunk in pd.read_csv(gtfs_dir / "stop_times.txt",
                             usecols=["trip_id", "stop_id", "stop_sequence"],
                             dtype={"trip_id": str, "stop_id": str}, chunksize=2_000_000):
        parts.append(chunk[chunk["trip_id"].isin(trip_ids)])
    st = pd.concat(parts).sort_values(["trip_id", "stop_sequence"])
    st["station"] = st["stop_id"].map(station_of)

    # Consecutive stations within a trip are directly connected
    nxt = st.groupby("trip_id")["station"].shift(-1)
    pairs = pd.DataFrame({"a": st["station"], "b": nxt}).dropna()
    pairs = pairs[pairs["a"] != pairs["b"]]

    G = nx.Graph()
    G.add_edges_from(pairs.itertuples(index=False, name=None))
    G.remove_edges_from(SKIP_LINKS)

    # One coordinate per station: the mean of its platforms
    used = stops[stops["station"].isin(G.nodes)]
    coords = used.groupby("station")[["stop_lon", "stop_lat"]].mean()
    for s, (lon, lat) in coords.iterrows():
        G.nodes[s]["lon"], G.nodes[s]["lat"] = lon, lat
    return G


# ----------------------------------------------------------------------------
# 2. Load the brain network
# ----------------------------------------------------------------------------
def load_brain_network(dataset=BRAIN_DATASET):
    from netneurotools.datasets import fetch_famous_gmat
    data = fetch_famous_gmat(dataset)
    if "coords" not in data:
        raise KeyError(f"No coordinates in {dataset}; available keys: {list(data.keys())}")
    conn = np.asarray(data["conn"], dtype=float)
    np.fill_diagonal(conn, 0)
    return conn, np.asarray(data["coords"], dtype=float)


def keep_strongest(conn, share):
    """Binary matrix with only the strongest share of connections (all of them if conn is binary)."""
    w = conn[np.triu_indices_from(conn, 1)]
    w = w[w > 0]
    if np.unique(w).size > 1:
        cut = np.quantile(w, 1 - share)
        conn = np.where(conn >= cut, 1.0, 0.0)
    else:
        conn = (conn > 0).astype(float)
    return np.maximum(conn, conn.T)


# ----------------------------------------------------------------------------
# 3. Draw both side by side
# ----------------------------------------------------------------------------
def draw_berlin(ax, G):
    lon = nx.get_node_attributes(G, "lon")
    lat = nx.get_node_attributes(G, "lat")
    segments = [[(lon[u], lat[u]), (lon[v], lat[v])] for u, v in G.edges]
    ax.add_collection(LineCollection(segments, colors=EDGE, linewidths=0.9, zorder=1))
    ax.scatter(list(lon.values()), list(lat.values()), s=3, c=NODE, linewidths=0, zorder=2)
    ax.set_aspect(1 / np.cos(np.radians(52.52)))   # so Berlin isn't stretched
    ax.autoscale_view()
    ax.axis("off")


def draw_brain(ax, conn, coords):
    from nilearn import plotting
    one_colour = ListedColormap([EDGE, EDGE])
    plotting.plot_connectome(
        conn, coords, axes=ax, display_mode="z", annotate=False,
        node_color=NODE, node_size=12,
        edge_cmap=one_colour, edge_vmin=0, edge_vmax=1,
        edge_kwargs={"linewidth": 0.9},
        colorbar=False,
    )


def make_figure(G, conn, coords, out_file=OUT_FILE):
    plt.rcParams["font.family"] = FONT
    fig, (ax_city, ax_brain) = plt.subplots(1, 2, figsize=(10, 4.8),
                                            gridspec_kw={"width_ratios": [1.15, 1]})
    draw_berlin(ax_city, G)
    draw_brain(ax_brain, conn, coords)
    for ax, label in [(ax_city, "Berlin"), (ax_brain, "Brain")]:   # both labels on one line
        fig.text(ax.get_position().x0, 0.93, label, fontsize=13, fontweight="bold", color=NODE)
    fig.savefig(f"{out_file}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_file}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {out_file}.png and {out_file}.pdf")


if __name__ == "__main__":
    if ANALYSIS_SCRIPT.exists():
        print(f"Using the network from {ANALYSIS_SCRIPT.name}")
        G = load_from_analysis(ANALYSIS_SCRIPT, GTFS_DIR)
    else:
        print(f"{ANALYSIS_SCRIPT.name} not found here, rebuilding the network from GTFS")
        G = build_berlin_network(GTFS_DIR)
    print(f"Berlin: {G.number_of_nodes()} stations, {G.number_of_edges()} connections (your analysis: 309 and 361)")
    # Sanity check: the S Bahn reaches about 40 km from the centre, nothing should be further away
    far = [n for n, d in G.nodes(data=True)
           if np.hypot((d["lon"] - 13.411) * 68, (d["lat"] - 52.521) * 111) > 60]
    if far:
        print(f"Warning: {len(far)} stations lie more than 60 km from Alexanderplatz, e.g. {far[:5]}")
    conn, coords = load_brain_network()
    conn = keep_strongest(conn, BRAIN_EDGE_SHARE)
    print(f"Brain: {len(coords)} regions, {int(conn.sum() // 2)} connections shown")
    make_figure(G, conn, coords)
