import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib_venn import venn2

# user written modules
import dataproject
import folium
import pycountry
import json

# We load in our datasets 
RDependency = pd.read_csv('RDependency.csv', sep=';')
RGDPChange = pd.read_csv('RGDPChange.csv', sep=';')
RINFChange = pd.read_csv('RINFChange.csv', sep=';')

# Replaced , with . for python use 
RDependency['RussianDependency'] = RDependency['RussianDependency'].str.replace(',', '.')
RGDPChange['GDPChange'] = RGDPChange['GDPChange'].str.replace(',', '.')
RINFChange['INFChange'] = RINFChange['INFChange'].str.replace(',', '.')

# Change data type to numeric
RDependency['RussianDependency'] = pd.to_numeric(RDependency['RussianDependency'], errors='coerce')
RGDPChange['GDPChange'] = pd.to_numeric(RGDPChange['GDPChange'], errors='coerce')
RINFChange['INFChange'] = pd.to_numeric(RINFChange['INFChange'], errors='coerce')

# Merge our data sets into a single DataFrame
merged_df = pd.merge(RDependency, RGDPChange, on='Country', how='inner')
final_merged_df = pd.merge(merged_df, RINFChange, on='Country', how='inner')

# Round decimals to 2 
FinalDF = final_merged_df.round({'RussianDependency': 2, 'GDPChange': 2, 'INFChange': 2})

# Adding country codes to our dataframe
def get_country_code(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return None 

FinalDF['Country_Code'] = FinalDF['Country'].apply(get_country_code)



# Print to check DataFrame
print(FinalDF)

# Data plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot each variable, sorted by the values of that variable
variables = ['RussianDependency', 'GDPChange', 'INFChange']
for i, variable in enumerate(variables):
    # Sort the DataFrame by the current variable
    sorted_df = FinalDF.sort_values(by=variable)
    ax = axes[i]
    ax.bar(sorted_df['Country'], sorted_df[variable], color='skyblue')
    ax.set_xlabel('Country')
    ax.set_ylabel(variable)
    ax.set_title(f'{variable} by Country')
    ax.tick_params(axis='x', rotation=90)

plt.tight_layout()


plt.show()

# Define the lookup table for country coordinates
country_coordinates = {
    "Austria": (47.5162, 14.5501),
    "Belgium": (50.8503, 4.3517),
    "Bulgaria": (42.7339, 25.4858),
    "Croatia": (45.1, 15.2),
    "Cyprus": (35.1264, 33.4299),
    "Czech Republic": (49.8175, 15.4729),
    "Denmark": (56.2639, 9.5018),
    "Estonia": (58.5953, 25.0136),
    "Finland": (61.9241, 25.7482),
    "France": (46.6034, 1.8883),
    "Germany": (51.1657, 10.4515),
    "Greece": (39.0742, 21.8243),
    "Hungary": (47.1625, 19.5033),
    "Ireland": (53.1424, -7.6921),
    "Italy": (41.8719, 12.5674),
    "Latvia": (56.8796, 24.6032),
    "Lithuania": (55.1694, 23.8813),
    "Luxembourg": (49.8153, 6.1296),
    "Malta": (35.9375, 14.3754),
    "Netherlands": (52.1326, 5.2913),
    "Poland": (51.9194, 19.1451),
    "Portugal": (39.3999, -8.2245),
    "Romania": (45.9432, 24.9668),
    "Slovakia": (48.669, 19.699),
    "Slovenia": (46.1512, 14.9955),
    "Spain": (40.4637, -3.7492),
    "Sweden": (60.1282, 18.6435)
   
}

# Initialize a map
m = folium.Map(location=[54, 15], tiles="OpenStreetMap", zoom_start=4)

# Function to create a popup for each country
def make_popup(country_row):
    return folium.Popup(f"Country: {country_row['Country']}<br>"
                        f"RussianDependency: {country_row['RussianDependency']}<br>"
                        f"GDPChange: {country_row['GDPChange']}<br>"
                        f"INFChange: {country_row['INFChange']}", max_width=300)

# Add markers to the map
for idx, row in FinalDF.iterrows():
    country = row['Country']
    coordinates = country_coordinates.get(country)
    if coordinates:
        folium.Marker(
            location=coordinates,
            popup=make_popup(row),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

# Display the map
m

# Define country codes
eu_country_codes = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'GR', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

# Load GeoJSON file to define country boundaries
with open('eu_countries.geojson', 'r') as f:
    country_geo = json.load(f)

# Change country code for Greece from EL to GR in the GeoJSON file
for feature in country_geo['features']:
    if feature['properties']['NUTS_ID'] == 'EL':
        feature['properties']['NUTS_ID'] = 'GR'

# Filter out non-EU countries from GeoJSON data
eu_country_geo = {
    'type': 'FeatureCollection',
    'features': [feature for feature in country_geo['features'] if feature['properties']['NUTS_ID'] in eu_country_codes]
}

# Initialize the map centered around a point
m = folium.Map(location=[54, 15], zoom_start=4)

# Create a Choropleth map where the color intensity is based on the inflation change (INFChange)
folium.Choropleth(
    geo_data=eu_country_geo,  
    name='choropleth',
    data=FinalDF,  
    columns=['Country_Code', 'INFChange'],  
    key_on='feature.properties.NUTS_ID',  
    fill_color='YlOrRd',  
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Inflation Change Rate (%)'
).add_to(m)

folium.LayerControl().add_to(m)

# Display the map
m

# Define country codes
eu_country_codes = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'GR', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

# Load GeoJSON file to define country boundaries
with open('eu_countries.geojson', 'r') as f:
    country_geo = json.load(f)

# Change country code for Greece from EL to GR in the GeoJSON file
for feature in country_geo['features']:
    if feature['properties']['NUTS_ID'] == 'EL':
        feature['properties']['NUTS_ID'] = 'GR'

# Filter out non-EU countries from GeoJSON data
eu_country_geo = {
    'type': 'FeatureCollection',
    'features': [feature for feature in country_geo['features'] if feature['properties']['NUTS_ID'] in eu_country_codes]
}

# Initialize the maps centered around a point
m = folium.Map(location=[54, 15], zoom_start=4)

folium.Choropleth(
    geo_data=eu_country_geo,  
    name='choropleth',
    data=FinalDF,  
    columns=['Country_Code', 'GDPChange'],  
    key_on='feature.properties.NUTS_ID', 
    fill_color='YlOrRd_r', 
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Another Value'
).add_to(m)


folium.LayerControl().add_to(m)

# Display the map
m