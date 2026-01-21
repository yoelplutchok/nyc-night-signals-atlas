#!/usr/bin/env python3
"""
08c_build_interactive_maps.py

Generate interactive Folium maps for the NYC Night Signals Atlas.
- Night Signatures (CD and NTA)
- Hotspot Concentration
- Heat Sensitivity

Outputs:
- reports/cd_typology.html
- reports/nta_typology.html
- reports/hotspots.html
- reports/heat_sensitivity.html
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import numpy as np
from datetime import datetime

from sleep_esi.paths import PROCESSED_DIR, GEO_DIR
from sleep_esi.logging_utils import get_logger

# Constants
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
NYC_CENTER = [40.7128, -74.0060]

def create_base_map():
    return folium.Map(location=NYC_CENTER, zoom_start=11, tiles="cartodbpositron")

def add_typology_layer(m, gdf, id_col, label_col, title):
    # Distinct colors for clusters
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000'
    ]
    
    # Map cluster_id_pooled to color
    cluster_ids = sorted(gdf['cluster_id_pooled'].unique())
    id_to_color = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}
    if -1 in id_to_color: id_to_color[-1] = '#808080' # Gray for Low Volume
    
    def style_fn(feature):
        cid = feature['properties'].get('cluster_id_pooled')
        return {
            'fillColor': id_to_color.get(cid, '#ffffff'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    folium.GeoJson(
        gdf,
        name=title,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[id_col, label_col, 'cluster_label_long', 'stability_class2'],
            aliases=['ID', 'Name', 'Type', 'Stability'],
            localize=True
        )
    ).add_to(m)

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger("08c_build_interactive_maps")
    logger.info("Generating interactive maps...")

    # 1. CD Typology
    logger.info("Mapping CD Typology...")
    cd_types = gpd.read_file(PROCESSED_DIR / "reports" / "cd_types_with_stability.geojson")
    m_cd = create_base_map()
    add_typology_layer(m_cd, cd_types, 'boro_cd', 'cd_label', 'CD Night Signatures')
    m_cd.save(REPORTS_DIR / "cd_typology.html")

    # 2. NTA Typology
    logger.info("Mapping NTA Typology...")
    nta_types = gpd.read_file(PROCESSED_DIR / "reports" / "nta_types_with_stability_residential.geojson")
    m_nta = create_base_map()
    add_typology_layer(m_nta, nta_types, 'ntacode', 'nta_name', 'NTA Night Signatures')
    m_nta.save(REPORTS_DIR / "nta_typology.html")

    # 3. Hotspots
    logger.info("Mapping Hotspots...")
    hotspots = gpd.read_file(PROCESSED_DIR / "hotspots" / "hotspot_cells_ge50.geojson")
    m_hot = create_base_map()
    
    # Heatmap of hotspots
    hotspots_proj = hotspots.to_crs(2263)
    hotspots_4326 = hotspots.to_crs(4326)
    centroids = hotspots_proj.geometry.centroid.to_crs(4326)
    
    heat_data = [[point.y, point.x, c] for point, c in zip(centroids, hotspots_4326['count'])]
    plugins.HeatMap(heat_data, name="Hotspot Intensity").add_to(m_hot)
    
    # Points for top hotspots
    for i, row in hotspots_4326.nlargest(100, 'count').iterrows():
        point = centroids.loc[i]
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=5,
            popup=f"Complaints: {row['count']}",
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m_hot)
    
    m_hot.save(REPORTS_DIR / "hotspots.html")

    # 4. Heat Sensitivity
    logger.info("Mapping Heat Sensitivity...")
    sens = pd.read_parquet(PROCESSED_DIR / "weather" / "cd_heat_sensitivity.parquet")
    cd59 = gpd.read_parquet(GEO_DIR / "cd59.parquet")
    sens_gdf = cd59.merge(sens, on='boro_cd')
    
    m_sens = create_base_map()
    folium.Choropleth(
        geo_data=sens_gdf.__geo_interface__,
        name='Heat Sensitivity',
        data=sens,
        columns=['boro_cd', 'pct_increase_per_c'],
        key_on='feature.properties.boro_cd',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='% Increase in Complaints per 1°C Tmin'
    ).add_to(m_sens)
    
    # Add tooltips manually
    folium.GeoJson(
        sens_gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
        tooltip=folium.GeoJsonTooltip(
            fields=['cd_label', 'pct_increase_per_c', 'pvalue'],
            aliases=['CD', '% Increase/°C', 'p-value'],
            localize=True
        )
    ).add_to(m_sens)
    
    m_sens.save(REPORTS_DIR / "heat_sensitivity.html")

    logger.info("SUCCESS: All interactive maps generated.")
    print(f"\n✓ Interactive maps saved to {REPORTS_DIR}")

if __name__ == "__main__":
    main()
