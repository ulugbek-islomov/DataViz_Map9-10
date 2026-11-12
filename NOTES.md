## Justification of cartographic choices

# Part 1 — Homework 9 (tabs 1–4)

**Question.** Where in Uzbekistan do households use banks — for borrowing and for saving?
This matters for fintech: around **80% of new retail loans are now approved by automated systems**
(Central Bank of Uzbekistan, financial inclusion index, Aug 2026), and lending to households grew **21.2%** in a year
versus **12%** for businesses. Automated credit can reach every region. The map checks whether savings followed.

**Data.** Central Bank of Uzbekistan, *Total loans and total deposits of banking system by regions*, as of 1 June 2026
(billion so'm). Population: Statistics Agency (stat.uz), permanent population by region, 1 April 2026 (the 14 values
add up exactly to the national total, 38,382.7 thousand). Boundaries: geoBoundaries ADM1, 14 regions. Geographic key:
the **ISO 3166-2 code** (UZ-SA, UZ-TK, …).

### 1. Projection (CRS)
- The boundary file comes in **EPSG:4326** (longitude/latitude in degrees). Degrees are not a length, so areas
  computed there are "square degrees" — meaningless (tab 4).
- A choropleth asks the reader to compare **coloured areas**, so the map must be **equal-area**.
  I use an **Albers equal-area conic centred on Uzbekistan** (standard parallels 38°N and 44°N, central meridian 64°E).
  The two parallels sit inside the country's 37°–46°N band, so distortion of shape is also small.
- Why not the others: **Web Mercator (EPSG:3857)** inflates the area of every region by 1.61× (Surkhandarya, south) to
  1.90× (Karakalpakstan, north), so the north looks ~18% bigger than it should next to the south.
  **UTM zone 41N (EPSG:32641)** is conformal, not equal-area, and Uzbekistan (56°–73°E) spans four UTM zones (40–43).
- Check: the Albers total is ≈ 443,700 km² vs the official ≈ 449,000 km² (−1%, because the boundary file is simplified).

### 2. Joins
- **Attribute join** bank table → regions on the ISO code, not on region names (spellings differ:
  "Djizzakh / Jizzakh / Jizzax", "Ferghana / Fergana"). An `assert` checks that all 14 regions joined.
- **Spatial join** (`gpd.sjoin`, `predicate="within"`) puts 29 Natural Earth cities into regions **after** converting
  them to the same CRS: 29 / 29 match and all agree with Natural Earth's own province labels — a check that the
  boundaries and the CRS are right. Without `to_crs` the join gives no error, but all 29 cities "land" in Navoi region
  (the Albers origin) — the silent bug shown in tab 4.

### 3. Normalization (why a rate)
- A **total** map shows where people and banks are: total household loans have a rank correlation of **+0.78** with
  population. Samarkand is **2nd by total but 13th per resident**; Navoi is **9th by total but 2nd per resident**.
- So the main map divides by population: **household loans per resident** and **household deposits per resident**
  (million so'm per person). The third measure, **loans ÷ deposits**, compares the two sides of the same households
  and needs no population at all.
- I use **household (individual)** loans and deposits, not totals: company money is booked where the head office is,
  so Tashkent city holds 76% of all deposits but only 8% of the people.
- Limitations: population includes children (no adult population by region in the same release), and the bank data
  (June) and population (April) are two months apart — small next to the 6× differences on the map.

### 4. Classification and colour
- **5 classes, natural breaks (Fisher-Jenks)** for the per-resident maps. Tashkent city is an extreme outlier
  (21.7 mln so'm of loans per person vs 3.4–11.1 elsewhere):
  - **Equal interval** puts 12 of 14 regions in one class → a one-colour map.
  - **Quantiles** forces ~3 regions per class and mixes very different values (GVF 0.58).
  - **Natural breaks** gives Tashkent city its own class and still separates the other 13 regions (GVF 0.998).
  With only 14 regions, more than 5 classes would leave 1–2 regions per class.
- **Sequential ramp (YlGnBu, light → dark)** for per-resident values: they only go from low to high.
- For **loans ÷ deposits** there is a meaningful midpoint, **1.0** (borrow as much as you save), so I use
  **fixed breaks 1, 2, 3** with a **diverging ramp**: blue = saves more than borrows, reds = borrows more.
  Fixed breaks are also easy to say in words and stay the same if the data is updated.
- **Tashkent city** is only 337 km², invisible at country scale, so it is drawn as an outlined dot with its value.
  A sorted bar chart next to the map shows exact values, so small regions are not hidden by big ones.
  No region has missing data.

### 5. What the map shows
- Credit has spread across the country much more than savings: outside Tashkent city households hold
  **4.65 mln so'm of loans but only 2.23 mln so'm of deposits per person**.
- In **13 of 14 regions households borrow more than they save**; the ratio is highest in the Fergana Valley
  (Namangan 3.6, Andijan 3.2, Fergana 3.0). Only Tashkent city is below 1 (0.69).
- The west (Navoi, Khorezm, Bukhara) has the highest savings per person outside the capital and ratios close to 1.
- For banks using automated / AI credit scoring this is the risk map: the regions with the most borrowing relative
  to savings are also the most populous, and the Central Bank already calls household debt growth "worrying".

# Part 2 — Homework 10 (tabs 5–7)

**Point dataset.** Every bank branch and ATM in **OpenStreetMap** (Geofabrik extract of Uzbekistan, 10 Nov 2026):
**2,894 points** — 1,579 bank branches and 1,315 ATMs. Banks drawn as buildings were turned into one point inside
the building. Same 14 regions and population table as Part 1.

**Data limitation.** OpenStreetMap is made by volunteers. It has 1,315 ATMs, but the Central Bank counts
**40,090 ATMs and self-service kiosks** (1 Jan 2026). So the points show *mapped* bank places: good for comparing
regions, not for real totals. Mapping effort also matters (Karakalpakstan has many mapped ATMs).

### 6. Layered map (tab 5)
- **Layer 1 (regions):** any measure from the sidebar — household loans/deposits per resident (Part 1) or
  bank points per 100,000 residents — with the same classification and colours as Part 1.
- **Layer 2 (points):** bank branches (dark red) and ATMs (orange) on top. Warm dots on cool polygons, so both layers
  are readable. Same Albers equal-area projection; no basemap tiles.

### 7. Spatial join (tab 6)
`gpd.sjoin(points, regions, predicate="within", how="left")`: **2,894 points → 2,886 matched, 8 matched nothing.**
The OpenStreetMap file is cut with a slightly wider outline of Uzbekistan than my boundary file. Some of the 8 have
names of Kyrgyz or Turkmen banks (Kompanion, MolBulak, Daýhanbank) — really outside the country; the others are
within a few km of the border, where the simplified boundary is not exact. They are 0.3% of the points, so I leave them
out of the rate and list them in the app.

### 8. Attribute join and rate (tab 6)
The population table (stat.uz) is joined on the **ISO 3166-2 code** (14 of 14 matched).
**Rate = points / population × 100,000.** I divide by people, not by area, because the question is how many bank
places people have. Tashkent city has **33% of all mapped points with 8% of the people**; Namangan has the fewest
(2.8 per 100,000). Points per person and household loans per person have a rank correlation of **+0.66**: the bank
places are where the credit is.

### 9. Shared filter
The sidebar **"Which points?"** (both / ATMs only / bank branches only) moves tabs 5, 6 and 7 and the
"bank points per 100,000" measure in tabs 1–3.

### 10. Clusters — bonus (tab 7)
- **DBSCAN** with **haversine** distance (latitude/longitude are degrees, not km), **min_samples = 5**,
  **eps = 5 km vs 20 km**. With both types: 5 km gives **84 clusters and 292 noise points**; 20 km gives
  **30 clusters and 58 noise points**. Adjusted Rand index between them **0.68**.
- 5 km finds towns and cities (Tashkent, Nukus, Samarkand, Bukhara, Fergana…), which matches places I know.
  20 km merges Tashkent with its satellite towns. I prefer 5 km. Noise = single branches/ATMs in villages and on
  roads, where bank access is thin, so they stay on the map.

### 11. Conclusion
Credit reached every region, but **savings and bank places did not**: outside Tashkent city households borrow about
2 so'm for every 1 so'm saved, and most mapped bank branches and ATMs are in Tashkent city. Regions like Namangan and
Kashkadarya are low on both — there, digital banking matters most.
