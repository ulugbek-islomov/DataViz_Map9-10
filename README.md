# Homework 9 + 10 — Credit, banks and ATMs in Uzbekistan

Streamlit app. Homework 9 (tabs 1–4): rate-based choropleth of household loans and deposits per resident.
Homework 10 (tabs 5–7): bank branches and ATMs from OpenStreetMap — layered map, spatial join, attribute join + rate, DBSCAN.

Live app: (paste link here)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabs
| Tab | Homework | What |
|---|---|---|
| 🗺️ Map | 9 | rate choropleth + sorted bar chart |
| 🔢 Count vs rate | 9 | why a rate |
| 📊 Classification | 9 | natural breaks vs quantiles vs equal interval |
| 🌐 Projection & spatial join | 9 | Albers equal-area, city spatial join, wrong-CRS bug |
| 🏦 Banks & ATMs (layered map) | 10 | layer 1 = rate choropleth, layer 2 = bank/ATM points |
| 🔗 Points → regions (joins) | 10 | spatial join (8 unmatched explained) + attribute join + rate |
| 🔵 Clusters (bonus) | 10 | DBSCAN, haversine, eps 5 km vs 20 km |
| 📝 Justification | 9 + 10 | NOTES.md |

Shared sidebar filter "Which points?" moves tabs 5–7 and the bank-points measure.

## Data
- data/region_stats.csv — population (stat.uz, 1 Apr 2026); household loans and deposits (Central Bank of Uzbekistan, 1 Jun 2026)
- data/uz_banks_atms.csv — 2,894 bank branches and ATMs, OpenStreetMap via Geofabrik (10 Nov 2026)
- data/uz_regions.geojson — 14 regions (geoBoundaries ADM1)
- data/uz_cities.geojson — 29 cities (Natural Earth), for the Homework 9 spatial join
