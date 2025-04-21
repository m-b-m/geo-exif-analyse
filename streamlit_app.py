import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, MultiPolygon, shape
import json
import io

# Titel en instructie
st.title("📷 Geo-Exif Analyse - De 20 lokaties in Nederland die het verst van 1 van je bestaande foto's zijn.")
st.write("Upload een CSV met geotags (GPSLatitude, GPSLongitude in DMS-formaat), en ontdek de 20 verste plekken in Nederland waar je nog geen foto hebt gemaakt.")
"""
## 🗺️ Stappenplan (voor Mac)

1. Open je terminal 
2. Installeer, indien nodig, homebrew en exiftool:
  - homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
  - exiftool: `brew install exiftool`
3. Ga in de terminal naar de map waar je foto's staan. Voor de Apple Photo's app is dat bijvoorbeeld zoiets als `~/Pictures/Photos Library.photoslibrary/originals`
4. Voer dit commando uit: `exiftool -csv -gpslatitude -gpslongitude -datetimeoriginal -c "%d %d %.8f" -fast2 -r . > ~/Desktop/gps-data.csv`
5. Upload dit bestand hier.
6. Op https://www.google.com/mymaps kan je een mooi kaartje maken met de .csv
"""

# Uploadbestand
uploaded_file = st.file_uploader("Upload je CSV-bestand met geotags", type=["csv"])

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

# Verwerk bestand als het is geüpload
if uploaded_file:
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
        st.write("Bezig met locatie-analyse...")
        nl_shape = load_nl_shape()
        lon_vals = np.linspace(3.3, 7.2, 300)
        lat_vals = np.linspace(50.75, 53.5, 300)
        grid_points = [Point(lon, lat) for lat in lat_vals for lon in lon_vals if nl_shape.contains(Point(lon, lat))]

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
            'Name': [f'Locatie {i+1}' for i in range(len(selected))],
            'Afstand': [round(r.distance_km) for r in selected],
            'Latitude': [r.geometry.y for r in selected],
            'Longitude': [r.geometry.x for r in selected]
        })

        st.success("Analyse voltooid! Hieronder zie je de top 20.")
        st.dataframe(result_df)

        # Downloadlink
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV voor Google My Maps",
            data=csv,
            file_name='top20_verste_punten.csv',
            mime='text/csv'
        )
