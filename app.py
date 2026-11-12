"""
Homework 9 + 10 - Credit, banks and ATMs in Uzbekistan
Homework 9 (tabs 1-4): household bank loans and deposits per resident in the 14 regions
(Central Bank of Uzbekistan, 1 June 2026) - rate-based choropleth, projection, classification.
Homework 10 (tabs 5-7): bank branches and ATMs from OpenStreetMap as a point layer -
layered map, spatial join, attribute join + rate, DBSCAN clusters.

Run:  pip install -r requirements.txt
      streamlit run app_uz_banking_map.py
"""
import warnings

import geopandas as gpd
import mapclassify as mc
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

st.set_page_config(page_title="Credit, banks and ATMs in Uzbekistan", page_icon="🗺️", layout="wide")

# Equal-area Albers conic made for Uzbekistan.
# The country spans 37.2N-45.6N and 56E-73E -> standard parallels 38N and 44N, centre meridian 64E.
UZ_ALBERS = "+proj=aea +lat_0=41 +lon_0=64 +lat_1=38 +lat_2=44 +datum=WGS84 +units=m +no_defs"

SOURCE = ("Sources: Central Bank of Uzbekistan, 'Total loans and total deposits of banking system by regions', "
          "1 June 2026 · Statistics Agency (stat.uz), permanent population, 1 April 2026 · "
          "Boundaries: geoBoundaries ADM1 (simplified) · Cities: Natural Earth · "
          "Bank branches and ATMs: OpenStreetMap (Geofabrik extract, 10 Nov 2026)")
TYPE_COLORS = {"ATM": "#E67E22", "bank branch": "#8E1B10"}


# ------------------------------------------------------------------ 1. load + join + rates
@st.cache_data
def load_data():
    regions = gpd.read_file("data/uz_regions.geojson")   # 14 polygons, EPSG:4326
    stats = pd.read_csv("data/region_stats.csv")         # money: billion so'm, population: thousand
    cities = gpd.read_file("data/uz_cities.geojson")     # 29 points, EPSG:4326
    points = pd.read_csv("data/uz_banks_atms.csv")        # Homework 10: 2,894 bank branches and ATMs (OSM)
    points["name"] = points["name"].fillna("(no name)")

    # Attribute join on the ISO 3166-2 code, not on names ("Djizzakh" / "Jizzakh" / "Jizzax")
    gdf = regions.merge(stats, left_on="shapeISO", right_on="iso", how="inner")
    assert len(gdf) == 14, "some regions did not join"

    # Rates. billion so'm / thousand people = million so'm per person
    gdf["loan_pc"] = gdf["loans_households"] / gdf["pop_thousand"]
    gdf["dep_pc"] = gdf["deposits_households"] / gdf["pop_thousand"]
    gdf["ltd"] = gdf["loans_households"] / gdf["deposits_households"]

    # Homework 10 - SPATIAL JOIN: which region contains each bank point? (how="left" keeps the misses)
    pts = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
    pts = gpd.sjoin(pts, regions[["shapeISO", "geometry"]], predicate="within", how="left").drop(columns="index_right")
    return gdf.to_crs(UZ_ALBERS), cities, pts.to_crs(UZ_ALBERS)   # everything is drawn in the equal-area CRS


gdf, cities, all_points = load_data()
TK = (gdf["region"] == "Tashkent city").values    # 337 km2: too small to see, drawn as a dot
# small label shifts (metres) so the crowded east does not overlap
NUDGE = {"Tashkent region": (60_000, 75_000), "Namangan": (0, 35_000), "Andijan": (35_000, -10_000),
         "Fergana": (-5_000, -35_000)}

# ------------------------------------------------------------------ Homework 10: shared point filter
st.sidebar.header("Bank points (Homework 10)")
point_choice = st.sidebar.radio("Which points?", ["Bank branches + ATMs", "ATMs only", "Bank branches only"],
                                help="Moves tabs 5, 6 and 7, and the 'bank points' measure in tabs 1-3.")
keep = {"Bank branches + ATMs": ["ATM", "bank branch"], "ATMs only": ["ATM"],
        "Bank branches only": ["bank branch"]}[point_choice]
points = all_points[all_points["type"].isin(keep)]
matched = points[points["shapeISO"].notna()]                 # inside one of the 14 regions
unmatched = points[points["shapeISO"].isna()]                # matched no region

# ATTRIBUTE JOIN (population was joined on the ISO code above) -> rate per 100,000 residents
gdf["points"] = gdf["shapeISO"].map(matched.groupby("shapeISO").size()).fillna(0)
gdf["pts_100k"] = gdf["points"] / gdf["pop_thousand"] * 100     # points / (thousand people) * 100

MEASURES = {   # label: (column, unit, kind)
    "Household loans per resident": ("loan_pc", "million so'm per person", "rate"),
    "Household deposits per resident": ("dep_pc", "million so'm per person", "rate"),
    "Household loans ÷ household deposits": ("ltd", "so'm borrowed per 1 so'm saved", "ratio"),
    "Bank branches + ATMs per 100,000 residents (Homework 10)": ("pts_100k", "points per 100,000 residents", "rate"),
    "Household loans, total (count, for comparison only)": ("loans_households", "billion so'm", "count"),
}
METHODS = ["Natural breaks (Fisher-Jenks)", "Quantiles", "Equal interval"]
FIXED = "Fixed breaks at 1, 2, 3 (diverging)"


# ------------------------------------------------------------------ 2. small helpers
def classify(values, method, k):
    if method == FIXED:
        return mc.UserDefined(values, bins=[1, 2, 3, values.max()])
    if method == "Quantiles":
        return mc.Quantiles(values, k=k)
    if method == "Equal interval":
        return mc.EqualInterval(values, k=k)
    return mc.FisherJenks(values, k=k)


def palette(n, method):
    if method == FIXED:                        # blue = saves more than borrows, reds = borrows more
        return ["#4393c3", "#fddbc7", "#ef8a62", "#b2182b"][:n]
    cmap = plt.get_cmap("YlGnBu")              # sequential: light = low, dark = high
    return [cmap(0.15 + 0.85 * i / max(n - 1, 1)) for i in range(n)]


def gvf(values, c):
    """Goodness of variance fit = 1 - SDCM / SDAM (1 = classes are perfectly uniform)."""
    sdam = ((values - values.mean()) ** 2).sum()
    sdcm = sum(((values[c.yb == i] - values[c.yb == i].mean()) ** 2).sum() for i in range(len(c.bins)))
    return 1 - sdcm / sdam


def fmt(x):
    return f"{x:,.0f}" if x >= 100 else f"{x:.2f}"


def draw_map(ax, col, c, colors, title, unit, labels=True):
    fill = [colors[i] for i in c.yb]
    gdf.plot(ax=ax, color=fill, edgecolor="white", linewidth=0.7)
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    # Tashkent city: big outlined dot so the most important region does not disappear
    p = gdf[TK].geometry.representative_point().iloc[0]
    ax.scatter(p.x, p.y, s=170, color=fill[TK.argmax()], edgecolor="black", linewidth=1.2, zorder=5)
    ax.annotate(f"Tashkent city\n{fmt(gdf.loc[TK, col].iloc[0])}", (p.x, p.y),
                xytext=(p.x - 230_000, p.y + 130_000), fontsize=8, fontweight="bold", path_effects=halo,
                arrowprops=dict(arrowstyle="-", lw=0.8))
    if labels:
        for _, r in gdf[~TK].iterrows():
            q = r.geometry.representative_point()
            dx, dy = NUDGE.get(r["region"], (0, 0))
            ax.text(q.x + dx, q.y + dy, f"{r['region']}\n{fmt(r[col])}", fontsize=6.5, ha="center", va="center",
                    path_effects=halo)
    lows = [c.y.min()] + list(c.bins[:-1])
    handles = [Patch(facecolor=colors[i], edgecolor="grey", label=f"{fmt(lows[i])} – {fmt(c.bins[i])}  ({c.counts[i]})")
               for i in range(len(c.bins))]
    ax.legend(handles=handles, title=unit, loc="lower left", fontsize=7.5, title_fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_axis_off()


# ------------------------------------------------------------------ 3. sidebar
st.sidebar.header("Choropleth (Homework 9)")
measure = st.sidebar.selectbox("Measure", list(MEASURES))
col, unit, kind = MEASURES[measure]
method = st.sidebar.selectbox("Classification", [FIXED] + METHODS if kind == "ratio" else METHODS)
k = 4 if method == FIXED else st.sidebar.slider("Number of classes", 3, 6, 5)
if kind == "count":
    st.sidebar.warning("This is a raw total. It mostly shows where people and big banks are. "
                       "Use it only to compare with the rate maps.")
st.sidebar.caption("Projection is fixed: Albers equal-area conic for Uzbekistan (see tab 4).")

values = gdf[col]
c = classify(values, method, k)
colors = palette(len(c.bins), method)

# ------------------------------------------------------------------ 4. header + key numbers
st.title("🗺️ Credit reached the regions, banks and savings did not")
st.markdown("**Homework 9 (tabs 1–4):** household **bank loans and deposits per resident** in Uzbekistan's 14 regions "
            "(1 June 2026). Today about **80% of new retail loans are approved by automated systems** (Central Bank, "
            "2026), so credit can reach any phone in any region. Deposits tell a different story.  \n"
            "**Homework 10 (tabs 5–7):** I add a point dataset — every **bank branch and ATM in OpenStreetMap** "
            "(2,894 points) — to see where people can physically meet a bank.")

nat_loan = gdf["loans_households"].sum() / gdf["pop_thousand"].sum()
nat_dep = gdf["deposits_households"].sum() / gdf["pop_thousand"].sum()
tk_dep_share = gdf.loc[TK, "deposits_households"].iloc[0] / gdf["deposits_households"].sum()
tk_pop_share = gdf.loc[TK, "pop_thousand"].iloc[0] / gdf["pop_thousand"].sum()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Loans per resident", f"{nat_loan:.2f} mln so'm")
m2.metric("Deposits per resident", f"{nat_dep:.2f} mln so'm")
m3.metric("Tashkent city deposit share", f"{tk_dep_share:.0%}",
          f"only {tk_pop_share:.0%} of the population", delta_color="off")
m4.metric("Regions: loans > deposits", f"{(gdf['ltd'] > 1).sum()} of 14")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["🗺️ Map", "🔢 Count vs rate", "📊 Classification", "🌐 Projection & spatial join",
     "🏦 Banks & ATMs (layered map)", "🔗 Points → regions (joins)", "🔵 Clusters (bonus)", "📝 Justification"])

# ------------------------------------------------------------------ tab 1: main map + sorted bars
with tab1:
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(9, 6.2))
        draw_map(ax, col, c, colors, measure, unit)
        st.pyplot(fig)
    with right:   # bar chart: small regions get the same space as big ones
        order = gdf.assign(cls=c.yb).sort_values(col)
        fig2, ax2 = plt.subplots(figsize=(5, 6.2))
        ax2.barh(order["region"], order[col], color=[colors[i] for i in order["cls"]], edgecolor="grey", lw=0.4)
        ref = {"loan_pc": nat_loan, "dep_pc": nat_dep, "ltd": 1}.get(col)
        if ref is not None:
            ax2.axvline(ref, color="#C0392B", ls="--", lw=1)
            ax2.text(ref, -0.9, f" {'national ' + fmt(ref) if col != 'ltd' else '1 = borrow as much as save'}",
                     color="#C0392B", fontsize=8)
        ax2.set_xlabel(unit)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_title("Same values, sorted", fontsize=10, loc="left")
        st.pyplot(fig2)

    ramp = "diverging ramp centred on 1.0" if method == FIXED else "sequential ramp YlGnBu"
    st.caption(f"**{measure}** ({unit}) · {method}, {len(c.bins)} classes (number of regions in brackets) · "
               f"{ramp} · Albers equal-area conic for Uzbekistan · Tashkent city (337 km²) is drawn as a dot · "
               f"'per resident' = all residents incl. children. {SOURCE}")

    top, low = gdf.loc[values.idxmax()], gdf.loc[values.idxmin()]
    st.info(f"**Reading:** highest is **{top['region']} ({fmt(top[col])})**, lowest is "
            f"**{low['region']} ({fmt(low[col])})**. "
            + {"loan_pc": "Navoi (home of large mining and metal companies) is second only to Tashkent city. The big Fergana "
                          "Valley regions and Samarkand are at the bottom per person, although they are near the "
                          "top in total loans (see tab 2).",
               "dep_pc": "The gap is huge: Tashkent city keeps about 34× more deposits per person than Namangan. "
                         "The three Fergana Valley regions are the lowest; the west (Navoi, Khorezm, Bukhara) "
                         "saves 3–4× more per person than the valley.",
               "ltd": "Only Tashkent city is below 1. Outside the capital households borrow about 2 so'm for every "
                      "1 so'm they keep in banks; in Namangan 3.6 so'm. Credit spread faster than savings — a "
                      "warning sign for automated (AI) credit scoring in these regions.",
               "pts_100k": "Mapped bank points per person follow credit: Tashkent city, Navoi and the west are high, "
                           "Namangan, Kashkadarya and Surkhandarya are low. Karakalpakstan is high mainly because "
                           "volunteers mapped many ATMs there (see tab 5).",
               "loans_households": "This map repeats the population map. Compare it with the rate map in tab 2."}[col])

# ------------------------------------------------------------------ tab 2: count vs rate
with tab2:
    st.subheader("Same table, two questions: “how much?” vs “how much per person?”")
    cnt, rate = gdf["loans_households"], gdf["loan_pc"]
    m = METHODS[0] if method == FIXED else method
    kk = 5 if method == FIXED else k
    c_cnt, c_rate = classify(cnt, m, kk), classify(rate, m, kk)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    draw_map(axes[0], "loans_households", c_cnt, palette(len(c_cnt.bins), m),
             "COUNT: household loans, billion so'm", "billion so'm")
    draw_map(axes[1], "loan_pc", c_rate, palette(len(c_rate.bins), m),
             "RATE: household loans per resident", "million so'm per person")
    st.pyplot(fig)

    ranks = pd.DataFrame({
        "region": gdf["region"],
        "population (thousand)": gdf["pop_thousand"],
        "household loans (bn so'm)": cnt,
        "rank by total": cnt.rank(ascending=False).astype(int),
        "per resident (mln so'm)": rate.round(2),
        "rank per resident": rate.rank(ascending=False).astype(int),
    }).sort_values("rank by total")
    st.dataframe(ranks, hide_index=True, width="stretch")
    r_cnt = cnt.corr(gdf["pop_thousand"], method="spearman")
    r_rate = rate.corr(gdf["pop_thousand"], method="spearman")
    st.markdown(f"Rank correlation with population: **total {r_cnt:+.2f}**, **per resident {r_rate:+.2f}**. "
                "Big regions get big totals. **Samarkand is 2nd by total but 13th per resident; Navoi is 9th by "
                "total but 2nd per resident.** That is why the main map uses a rate.")

# ------------------------------------------------------------------ tab 3: classification
with tab3:
    kk = 5 if method == FIXED else k
    st.subheader(f"{measure}: three ways to cut the same 14 numbers ({kk} classes)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    rows = []
    for ax, mth in zip(axes, METHODS):
        cc = classify(values, mth, kk)
        draw_map(ax, col, cc, palette(len(cc.bins), mth), mth, unit, labels=False)
        rows.append({"method": mth, "upper breaks": ", ".join(fmt(b) for b in cc.bins),
                     "regions per class": " / ".join(str(n) for n in cc.counts),
                     "GVF (1 = perfect)": round(gvf(values.values, cc), 3)})
    st.pyplot(fig)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # every region as a dot on the value axis (sorted) + the breaks of the selected method
    order = gdf.assign(cls=c.yb).sort_values(col)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(order[col], order["region"], s=60, color=[colors[i] for i in order["cls"]], edgecolor="black", zorder=3)
    for b in c.bins[:-1]:
        ax.axvline(b, color="#C0392B", ls="--", lw=1)
    ax.grid(axis="y", color="#eeeeee")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(f"{unit}   (red dashed lines = breaks of '{method}')")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.markdown("**What to notice:** Tashkent city is far above every other region. **Equal interval** then puts "
                "almost all regions in the first class (12 / 0 / 1 / 0 / 1 for loans per resident) and the map "
                "says nothing. **Quantiles** forces ~3 regions per class, so very different values share a colour "
                "(lowest GVF). **Natural breaks** keeps Tashkent city alone and still separates the other 13 "
                "regions, with the highest GVF — so it is the default for the per-resident maps. (Equal interval also "
                "has a high GVF, but only because Tashkent city alone carries most of the variance — the map is still almost one colour.) For the "
                "loans ÷ deposits ratio there is a meaningful value, **1.0**, so fixed breaks 1, 2, 3 with a "
                "diverging ramp are easier to read than computed breaks.")

# ------------------------------------------------------------------ tab 4: projection + spatial join
with tab4:
    st.subheader("1) Which CRS? Degrees are not area")
    geo = gdf.to_crs(4326)
    with warnings.catch_warnings():            # geopandas warns: area in a geographic CRS is meaningless
        warnings.simplefilter("ignore")
        sq_deg = geo.area
    area = pd.DataFrame({
        "region": gdf["region"],
        "EPSG:4326 'area' (square degrees ✘)": sq_deg.round(2),
        "Albers equal-area (km² ✔)": (gdf.area / 1e6).round(0),
        "Web Mercator EPSG:3857 (km²)": (geo.to_crs(3857).area / 1e6).round(0),
    })
    area["Mercator ÷ Albers"] = (area.iloc[:, 3] / area.iloc[:, 2]).round(2)
    st.dataframe(area.sort_values("Mercator ÷ Albers"), hide_index=True, width="stretch")
    lo, hi = area.loc[area["Mercator ÷ Albers"].idxmin()], area.loc[area["Mercator ÷ Albers"].idxmax()]
    st.markdown(f"Total in Albers: **{area.iloc[:, 2].sum():,.0f} km²** vs the official ~449,000 km² (about 1% "
                "less, because the boundary file is simplified). Web Mercator inflates areas from "
                f"**{lo['Mercator ÷ Albers']}× ({lo['region']}, south) to {hi['Mercator ÷ Albers']}× "
                f"({hi['region']}, north)**. A choropleth makes the eye compare coloured areas, so the map uses "
                "an equal-area projection.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (crs, name) in zip(axes, [(4326, "EPSG:4326 (lon/lat drawn flat)"),
                                      (3857, "EPSG:3857 Web Mercator"),
                                      (UZ_ALBERS, "Albers equal-area for Uzbekistan ✔")]):
        gdf.to_crs(crs).plot(ax=ax, color="#EAF2F6", edgecolor="#2E6E8E", lw=0.7)
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    st.pyplot(fig)

    st.subheader("2) Spatial join: which region is each city in?")
    print_crs = f"cities: `{cities.crs.to_string()}` → regions: Albers (metres)"
    cities_eq = cities.to_crs(gdf.crs)                          # ✔ same CRS first
    joined = gpd.sjoin(cities_eq, gdf[["region", "geometry"]], predicate="within", how="left")
    agree = (joined["region"].str[:4] == joined["adm1name"].replace(
        {"Bukhoro": "Bukhara", "Andijon": "Andijan", "Ferghana": "Fergana", "Sirdaryo": "Sirdarya"}).str[:4]).sum()
    with warnings.catch_warnings():                             # ✘ the wrong way: degrees vs metres
        warnings.simplefilter("ignore")
        wrong = gpd.sjoin(cities, gdf[["region", "geometry"]], predicate="within", how="left")
    a, b = st.columns([3, 2])
    with a:
        fig, ax = plt.subplots(figsize=(9, 6))
        gdf.plot(ax=ax, column="region", cmap="tab20", alpha=0.35, edgecolor="white")
        cities_eq.plot(ax=ax, color="black", markersize=cities_eq["pop_max"] / 6000)
        for _, r in joined[joined["pop_max"] > 200_000].iterrows():
            ax.annotate(r["name"], (r.geometry.x, r.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title("29 Natural Earth cities (dot size ~ population) over the 14 regions", fontsize=10)
        ax.set_axis_off()
        st.pyplot(fig)
    with b:
        st.markdown(f"CRS check before the join — {print_crs}")
        per_region = (joined.sort_values("pop_max", ascending=False).groupby("region")
                      .agg(cities=("name", "size"), largest_city=("name", "first")).reset_index())
        st.dataframe(per_region, hide_index=True, width="stretch")
        st.success(f"**Same CRS:** {joined['region'].notna().sum()} of {len(cities)} cities matched a region, "
                   f"and {agree} of {len(cities)} agree with Natural Earth's own province label. "
                   "Every region has at least one city.")
        st.error(f"**Different CRS (no `to_crs`):** {wrong['region'].notna().sum()} of {len(cities)} cities "
                 f"'matched' — all to **{wrong['region'].mode()[0]}**. The city coordinates in degrees "
                 "(≈ 60–71, 37–45) are read as a few metres from the Albers origin (64°E, 41°N), which lies "
                 "inside that region. No error is raised. This is the silent bug from the lecture.")

# ------------------------------------------------------------------ tab 5: Homework 10 - layered map
with tab5:
    st.subheader(f"Two layers: {measure} + {point_choice}")
    a, b, c3, d = st.columns(4)
    a.metric("Points (filter)", f"{len(points):,}")
    b.metric("Inside the 14 regions", f"{len(matched):,}", f"{len(unmatched)} outside", delta_color="off")
    b_tk = gdf.loc[TK, "points"].iloc[0] / len(matched)
    c3.metric("Tashkent city share of the points", f"{b_tk:.0%}",
              f"with {tk_pop_share:.0%} of the people", delta_color="off")
    d.metric("Official ATMs + kiosks (Central Bank)", "40,090", "OSM has 1,315 ATMs", delta_color="off")

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_map(ax, col, c, colors, f"Layer 1: {measure}  ·  Layer 2: {point_choice}", unit, labels=False)
    first_legend = ax.get_legend()
    for t in reversed(keep):                     # layer 2: the points (ATMs drawn last, on top)
        sel = matched[matched["type"] == t]
        ax.scatter(sel.geometry.x, sel.geometry.y, s=5, color=TYPE_COLORS[t], alpha=0.6, label=f"{t} ({len(sel):,})",
                   zorder=6)                 # on top of the Tashkent city dot
    ax.legend(loc="upper right", fontsize=8, markerscale=3, frameon=False)
    ax.add_artist(first_legend)                  # keep both legends
    st.pyplot(fig)
    st.caption(f"**Layer 1 (regions):** {measure} ({unit}), {method}, {len(c.bins)} classes, {ramp}. "
               f"**Layer 2 (points):** {point_choice} from OpenStreetMap, one dot per object "
               f"(orange = ATM, dark red = bank branch). Projection: Albers equal-area conic for Uzbekistan. "
               f"{len(unmatched)} points outside the 14 regions are not drawn (tab 6). Change layer 1 in the "
               f"sidebar (e.g. household loans per resident). {SOURCE}")

    r = gdf["pts_100k"].corr(gdf["loan_pc"], method="spearman")
    st.info(f"**Reading:** the bank points sit where the credit is. Mapped bank points per 100,000 residents and "
            f"household loans per resident have a rank correlation of **{r:+.2f}**. Tashkent city has "
            f"**{b_tk:.0%} of all mapped points with {tk_pop_share:.0%} of the people**. Namangan and Kashkadarya "
            "are low on both maps — little credit per person and few bank points. "
            "**Important:** OpenStreetMap is made by volunteers. It has 1,315 ATMs, while the Central Bank counts "
            "40,090 ATMs and self-service kiosks (1 Jan 2026). So the dots show *mapped* bank places — good for "
            "comparing regions, not for real totals. Karakalpakstan is high mostly because people there mapped "
            "many ATMs.")

# ------------------------------------------------------------------ tab 6: Homework 10 - the two joins
with tab6:
    st.subheader("1) Spatial join: bank points → the region that contains them")
    st.code('gpd.sjoin(points, regions, predicate="within", how="left")   # "left" keeps the points that match nothing',
            language="python")
    border = gdf.union_all().boundary                          # Uzbekistan's border (metres, equal-area CRS)
    miss = unmatched.assign(km_to_border=unmatched.distance(border) / 1000).sort_values("km_to_border")
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"**{len(points):,} points in → {len(matched):,} matched a region → "
                    f"{len(unmatched)} matched nothing.**")
        st.dataframe(miss[["type", "name", "lat", "lon", "km_to_border"]].round(2), hide_index=True, width="stretch")
        st.markdown("**Why they did not match:** the OpenStreetMap file for Uzbekistan is cut with a slightly wider "
                    "outline than my boundary file, so it brings a few points from across the border. Some names "
                    "show it: *Банк Компаньон* and *МолБулак* are Kyrgyz, *Daýhanbank* is Turkmen — really outside "
                    "Uzbekistan. The others are within a few km of the border, where my simplified boundary is not "
                    "exact. They are less than 0.3% of the points, so I leave them out of the rate and list them here.")
    with b:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        gdf.plot(ax=ax, color="#EAF2F6", edgecolor="#2E6E8E", linewidth=0.6)
        ax.scatter(miss.geometry.x, miss.geometry.y, color="#C0392B", marker="x", s=60)
        ax.set_title("Points that matched no region", fontsize=10)
        ax.set_axis_off()
        st.pyplot(fig)

    st.subheader("2) Attribute join + rate: population → regions → points per 100,000 residents")
    st.code('regions.merge(stats, left_on="shapeISO", right_on="iso")   # external table, key = ISO 3166-2 code\n'
            'rate = points / population * 100,000', language="python")
    st.markdown("The population table (stat.uz, 1 April 2026) is joined to the regions on the **ISO code** "
                "(UZ-TK, UZ-SA, …), not on names, because names are spelled in many ways (*Jizzakh / Djizzakh / "
                f"Jizzax*). **{gdf['pop_thousand'].notna().sum()} of 14** regions matched. I divide by population, "
                "not by area: the question is how many bank places people have, not how many per km².")
    per_region = (matched.groupby(["shapeISO", "type"]).size().unstack(fill_value=0).reindex(columns=keep, fill_value=0))
    table = gdf[["shapeISO", "region", "pop_thousand"]].merge(per_region, left_on="shapeISO", right_index=True, how="left")
    table["points per 100,000"] = gdf["pts_100k"].round(1).values
    st.dataframe(table.sort_values("points per 100,000", ascending=False), hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab 7: Homework 10 bonus - DBSCAN
with tab7:
    st.subheader("Which bank points form a group? DBSCAN with haversine distance")
    st.markdown("DBSCAN groups points that are close together; points without a group are **noise (−1)**. "
                "Latitude and longitude are degrees, not km, so I use the **haversine** distance (on the sphere): "
                "eps in km is divided by the Earth radius (6,371 km).")
    min_s = st.slider("min_samples (points needed to make a group)", 3, 10, 5)
    X = np.radians(matched[["lat", "lon"]].values)
    results = {}
    cols = st.columns(2)
    for eps_km, place in zip([5, 20], cols):
        labels = DBSCAN(eps=eps_km / 6371.0088, min_samples=min_s, metric="haversine").fit_predict(X)
        results[eps_km] = labels
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        noise = labels == -1
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        gdf.plot(ax=ax, color="#EEEEEE", edgecolor="white")
        ax.scatter(matched.geometry.x[noise], matched.geometry.y[noise], s=3, color="black")
        ax.scatter(matched.geometry.x[~noise], matched.geometry.y[~noise], s=6, c=labels[~noise] % 20, cmap="tab20")
        ax.set_title(f"eps = {eps_km} km: {n_cl} clusters, {noise.sum()} noise points", fontsize=10)
        ax.set_axis_off()
        place.pyplot(fig)
    ari = adjusted_rand_score(results[5], results[20])
    changed = ((results[5] == -1) != (results[20] == -1)).sum()
    st.caption(f"Layer 1: the 14 regions (grey). Layer 2: DBSCAN clusters of {point_choice}, haversine "
               f"distance, min_samples = {min_s}, eps = 5 km (left) and 20 km (right); black = noise. "
               "Albers equal-area projection.")
    st.markdown(f"**Comparing the two eps:** adjusted Rand index = **{ari:.2f}** (1 = the same grouping), and "
                f"**{changed}** points change between cluster and noise. Cluster numbers are random labels, so I "
                "compare the groupings, not the numbers.")
    st.markdown("**What I see:** with **eps = 5 km** the clusters are towns and cities — Tashkent is by far the "
                "biggest, then Nukus, Samarkand, Bukhara, Fergana, Andijan, Navoi, Urgench, Karshi and others. "
                "This matches places I know, so the parameters make sense. With **eps = 20 km** Tashkent joins the "
                "towns around it (Chirchiq, Yangiyul) and there are fewer clusters and less noise. I prefer 5 km "
                "because I want to see each town. The noise points are single branches or ATMs in villages and on "
                "roads — exactly where access to banks is thin — so I keep them on the map in black.")

# ------------------------------------------------------------------ tab 8: written justification
with tab8:
    st.markdown(open("NOTES.md", encoding="utf-8").read())
