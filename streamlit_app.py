import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from scipy.spatial import cKDTree
from shapely.geometry import Point, MultiPolygon, shape, Polygon
import json
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import random
import time
import threading

st.markdown("""
    <style>
    .sims-loader {
        background-color: #DDEEFF;
        padding: 2em;
        border-radius: 1em;
        text-align: center;
        font-family: 'Comic Sans MS', cursive;
        color: #003366;
        margin-bottom: 2em;
    }
    .sims-title {
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 1em;
    }
    .sims-quote {
        font-size: 1.2em;
        margin: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)

# Loader quotes
quotes = [
    "Grid aan het genereren over Nederland…",
    "Reticulating splines…",
    "Top 20 wordt geselecteerd op basis van maximale isolement…",
    "Verstrooide pixels worden verzameld…",
    "De platte aarde wordt kort gereconstrueerd voor afstandsberekeningen…",
    "Nederland wordt opgemeten in 90.000 stukjes…",
    "Onzichtbare grenzen van Drenthe worden gerespecteerd…",
    "We meten hoever je ooit van je camera bent weggelopen…",
    "Het raster maakt zich klaar voor een diepe analyse…",
    "De konijnenhol-fotodatabase wordt gesynchroniseerd…",
    "Bezig met het opsporen van je fotografische blinde vlekken…"
]

# Titel en instructie
st.title("📷 Geo-Exif Analyse - De 20 lokaties in Nederland die het verst van 1 van je bestaande foto's zijn.")
instructions_box = st.empty()

instructions_box.write("""
Upload een CSV met geotags (GPSLatitude, GPSLongitude in DMS-formaat), en ontdek de 20 verste plekken in Nederland waar je nog geen foto hebt gemaakt.
## 🗺️ Stappenplan (voor Mac)

1. Open je terminal 
2. Installeer, indien nodig, homebrew en exiftool:
  - homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
  - exiftool: `brew install exiftool`
3. Ga in de terminal naar de map waar je foto's staan. Voor de Apple Photo's app is dat bijvoorbeeld zoiets als `~/Pictures/Photos Library.photoslibrary/originals`
4. Voer dit commando uit: `exiftool -csv -gpslatitude -gpslongitude -datetimeoriginal -c "%d %d %.8f" -fast2 -r . > ~/Desktop/gps-data.csv`
5. Upload dit bestand hier.
6. Voila, je ziet een tabel, een kaartje en een csv-download. Op https://www.google.com/mymaps kan je daarmee een mooi eigen kaartje maken.
""")

# Helper: DMS naar decimalen
import re
def dms_to_decimal(dms_str):
    match = re.match(r"(\d+)\s+(\d+)\s+([\d.]+)\s+([NSEW])", str(dms_str).strip())
    if not match:
        return None
    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# Laad Nederland-grens uit GeoJSON (van GADM bijvoorbeeld)
def load_nl_shape():
    with open("nederland_vasteland.geojson", "r") as f:
        data = json.load(f)
    geom = shape(data['features'][0]['geometry'])
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)  # grootste polygon = vasteland
    return geom

# Uploadbestand
uploaded_file = st.file_uploader("Upload je CSV-bestand met geotags", type=["csv"])

# Laad shape van Nederland
nl_shape = load_nl_shape()

# Verwerk bestand als het is geüpload
if uploaded_file:
    instructions_box.empty()
    loader_box = st.empty()
    progress_bar = st.progress(0)

    for i in range(10):
        quote = random.choice(quotes)
        loader_box.markdown(f"""
            <div class="sims-loader">
                <div class="sims-title">Now loading…</div>
                <div class="sims-quote">{quote}</div>
            </div>
        """, unsafe_allow_html=True)
        progress_bar.progress((i + 1) * 10)
        time.sleep(0.6)

    df = pd.read_csv(uploaded_file)

    # Omzetten naar decimale coördinaten
    df['Latitude'] = df['GPSLatitude'].dropna().apply(dms_to_decimal)
    df['Longitude'] = df['GPSLongitude'].dropna().apply(dms_to_decimal)
    valid_coords = df.dropna(subset=['Latitude', 'Longitude'])[['Latitude', 'Longitude']].reset_index(drop=True)

    if valid_coords.empty:
        st.error("Geen geldige GPS-gegevens gevonden in het bestand.")
    else:
        # KD-tree voor snelste afstandsberekening
        photo_coords = valid_coords[['Longitude', 'Latitude']].to_numpy()
        photo_tree = cKDTree(photo_coords)

        # Grid genereren over Nederland
        # Stap 1: Grof raster
        lon_vals_coarse = np.linspace(3.3, 7.2, 100)
        lat_vals_coarse = np.linspace(50.75, 53.5, 100)
        coarse_grid = [Point(lon, lat) for lat in lat_vals_coarse for lon in lon_vals_coarse if nl_shape.contains(Point(lon, lat))]
        coarse_coords = np.array([(p.x, p.y) for p in coarse_grid])
        coarse_dists, _ = photo_tree.query(coarse_coords)

        # Selecteer top 100 grofste punten
        top_indices = np.argsort(coarse_dists)[-100:]
        focus_points = [coarse_grid[i] for i in top_indices]

        # Stap 2: Fijn raster rond elk top-punt
        fine_grid = []
        for p in focus_points:
            lon_vals_fine = np.linspace(p.x - 0.05, p.x + 0.05, 15)
            lat_vals_fine = np.linspace(p.y - 0.05, p.y + 0.05, 15)
            fine_grid += [Point(lon, lat) for lat in lat_vals_fine for lon in lon_vals_fine if nl_shape.contains(Point(lon, lat))]

        # Gebruik fine_grid als nieuw raster voor analyse
        grid_points = fine_grid
        grid_coords = np.array([(p.x, p.y) for p in grid_points])

        dists, indices = photo_tree.query(grid_coords)
        
        # Bouw dataframe met afstand + dichtstbijzijnde index
        df_all = pd.DataFrame({
            'geometry': grid_points,
            'distance_km': dists * 111,
            'nearest_photo_index': indices
        }).sort_values(by='distance_km', ascending=False).reset_index(drop=True)

        # Selecteer top 20 met unieke dichtstbijzijnde foto-indexen
        used_photo_indices = set()
        selected = []

        for _, row in df_all.iterrows():
            if row['nearest_photo_index'] not in used_photo_indices:
                selected.append(row)
                used_photo_indices.add(row['nearest_photo_index'])
            if len(selected) == 20:
                break

        # Resultaat tonen
        result_df = pd.DataFrame({
            'Name': [
                f"[Locatie {i+1}](https://www.google.com/maps/search/?api=1&query={r.geometry.y},{r.geometry.x})"
                for i, r in enumerate(selected)
            ],
            'Afstand': [round(r.distance_km, 1) for r in selected],
            'Latitude': [r.geometry.y for r in selected],
            'Longitude': [r.geometry.x for r in selected]
        })
        line_data = pd.DataFrame([
            {
            "from_lon": row.geometry.x,
            "from_lat": row.geometry.y,
            "to_lon": photo_coords[row.nearest_photo_index][0],
            "to_lat": photo_coords[row.nearest_photo_index][1]
            }
            for row in selected
        ])

        # Opruimen
        loader_box.empty()
        progress_bar.empty()

        st.success("Analyse voltooid! Hieronder zie je de top 20 in een tabel, een kaartje en als download.")

        st.markdown(result_df.to_markdown(index=False), unsafe_allow_html=True)

        # Streamlit map visualisatie
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            tooltip={"text": "Afstand: {Afstand} km"},
            initial_view_state=pdk.ViewState(
                latitude=52.1,
                longitude=5.4,
                zoom=6.2,
                pitch=0
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=result_df,
                    get_position='[Longitude, Latitude]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=1500,
                    pickable=True,
                    tooltip=True,
                ),
                # LineLayer toevoegen
                pdk.Layer(
                    "LineLayer",
                    data=line_data,
                    get_source_position='[from_lon, from_lat]',
                    get_target_position='[to_lon, to_lat]',
                    get_width=2,
                    get_color='[100, 100, 200]',
                    pickable=False
                )
            ]
        ))

        # Downloadlink
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV voor Google My Maps",
            data=csv,
            file_name='top20_verste_punten.csv',
            mime='text/csv'
        )

