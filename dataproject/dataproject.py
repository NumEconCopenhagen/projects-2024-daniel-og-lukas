import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib_venn import venn2

# user written modules
import folium
import pycountry
import json

# We load in our datasets 
LoadMain = pd.read_csv('TradeandGDP2021.csv', sep=';')
LoadInf = pd.read_csv('Priceindex.csv', sep=';')
LoadGDP = pd.read_csv('GDP.csv', sep=';')

# Keeping the origial dataframe for presentation
Main=LoadMain.copy()
# Replace commas with periods and convert to float for Main
columns_to_convert = Main.columns[1:]
for column in columns_to_convert:
    Main[column] = Main[column].str.replace(',', '.').astype(float).round(1)
# Calculate RussianDependency2021
Main['RussianDependency2021'] = (Main['ImportFromRussia2021'] / Main['GDP2021']*100).round(1)

# Keeping the origial dataframe for presentation
Inf=LoadInf.copy()
# Replace commas with periods and convert to float for Inf
columns_to_convert = Inf.columns[1:]
for column in columns_to_convert:
    Inf[column] = Inf[column].str.replace(',', '.').astype(float)

# Calculate the percentage change in inflation for each country into a new Dataframe and dropping the first row as that becomes Nan
inflation_change = Inf.iloc[:, 1:].pct_change().multiply(100)
InfChange = pd.concat([Inf['TIME'], inflation_change], axis=1)
InfChange = InfChange.drop(InfChange.index[0])

# We want to calculate an avg pre war and post war for each country and fin the difference
# Define the time periods
start_period_1 = '2015-02'
end_period_1 = '2022-02'
start_period_2 = '2022-03'
end_period_2 = '2023-08'

# Filter the DataFrame for the first time period and calculate the mean
first_period = InfChange[(InfChange['TIME'] > start_period_1) & (InfChange['TIME'] <= end_period_1)]
avg_change_period_1 = first_period.iloc[:, 1:].mean()  # Exclude 'TIME' column

# Filter the DataFrame for the second time period and calculate the mean
second_period = InfChange[(InfChange['TIME'] > start_period_2) & (InfChange['TIME'] <= end_period_2)]
avg_change_period_2 = second_period.iloc[:, 1:].mean()  # Exclude 'TIME' column


AVGInfChange = pd.DataFrame({
    'Country': InfChange.columns[1:],  # The first column is 'TIME' and should be excluded
    'AVGPreWarInf': avg_change_period_1.values,
    'AVGPostWarInf': avg_change_period_2.values
})

AVGInfChange['PostWarINFChange']= (AVGInfChange['AVGPostWarInf'] - AVGInfChange['AVGPreWarInf']).round(2)

# Merging the two dataframes
Main = pd.merge(Main, AVGInfChange, on='Country', how='inner')

# Keeping the origial dataframe for presentation
GDP=LoadGDP.copy()
columns_to_convert = GDP.columns[1:]
for column in columns_to_convert:
    GDP[column] = GDP[column].str.replace('.', '')  # Remove thousand separator
    GDP[column] = GDP[column].str.replace(',', '.').astype(float) 

# Calculate the percentage change in inflation for each country into a new Dataframe and dropping the first row as that becomes Nan
gdp_change = GDP.iloc[:, 1:].pct_change().multiply(100)
GDPChange = pd.concat([GDP['TIME'], gdp_change], axis=1)
GDPChange = GDPChange.drop(GDPChange.index[0])

# We want to calculate an avg pre war and post war for each country and fin the difference
# Define the time periods
gdpstart_period_1 = '2015-2'
gdpend_period_1 = '2022-2'
gdpstart_period_2 = '2022-3'
gdpend_period_2 = '2023-2'

# Filter the DataFrame for the first time period and calculate the mean
gdpfirst_period = GDPChange[(GDPChange['TIME'] > gdpstart_period_1) & (GDPChange['TIME'] <= gdpend_period_1)]
gdpavg_change_period_1 = gdpfirst_period.iloc[:, 1:].mean()  # Exclude 'TIME' column

# Filter the DataFrame for the second time period and calculate the mean
gdpsecond_period = GDPChange[(GDPChange['TIME'] > gdpstart_period_2) & (GDPChange['TIME'] <= gdpend_period_2)]
gdpavg_change_period_2 = gdpsecond_period.iloc[:, 1:].mean()  # Exclude 'TIME' column


AVGGDPChange = pd.DataFrame({
    'Country': GDPChange.columns[1:],  # The first column is 'TIME' and should be excluded
    'AVGPreWarGDP': gdpavg_change_period_1.values,
    'AVGPostWarGDP': gdpavg_change_period_2.values
})

AVGGDPChange['PostWarGDPChange']= (AVGGDPChange['AVGPostWarGDP'] - AVGGDPChange['AVGPreWarGDP']).round(2)

# Merging the two dataframes
Main = pd.merge(Main, AVGGDPChange, on='Country', how='inner')

# Adding country codes to our dataframe
def get_country_code(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return None 

Main['Country_Code'] = Main['Country'].apply(get_country_code)

# Removing columns in a new dataframe for presentation

columns_to_drop = ['ImportFromRussia2021', 'ExportToRussia2021', 'GDP2021', 
                   'AVGPreWarInf', 'AVGPostWarInf', 'AVGPreWarGDP', 'AVGPostWarGDP']

# Create a new DataFrame without the specified columns
CleanMain = Main.drop(columns=columns_to_drop)

def barchart(df):
    # Create figure and axes for the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # List of variables to plot
    variables = ['RussianDependency2021', 'PostWarINFChange', 'PostWarGDPChange']
    
    # Loop through each variable and create a bar plot
    for i, variable in enumerate(variables):
        sorted_df = df.sort_values(by=variable)
        ax = axes[i]
        ax.bar(sorted_df['Country'], sorted_df[variable], color='skyblue')
        ax.set_xlabel('Country')
        ax.set_ylabel(variable)
        ax.set_title(f'{variable} by Country')
        ax.tick_params(axis='x', rotation=90)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def scatterplot(df):
    # Extracting data
    x = df['RussianDependency2021']
    y1 = df['PostWarINFChange']
    y2 = df['PostWarGDPChange']

    # Fit linear regression models
    m1, b1 = np.polyfit(x, y1, 1)  # For PostWarINFChange
    m2, b2 = np.polyfit(x, y2, 1)  # For PostWarGDPChange

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plotting the first subplot for PostWarINFChange
    axes[0].scatter(x, y1, color='blue')  # Scatter plot
    axes[0].plot(x, m1 * x + b1, color='red')  # Regression line
    axes[0].set_title('Russian Dependency vs Post-War Inflation Change')
    axes[0].set_xlabel('Russian Dependency 2021')
    axes[0].set_ylabel('Post-War Inflation Change')

    # Plotting the second subplot for PostWarGDPChange
    axes[1].scatter(x, y2, color='green')  # Scatter plot
    axes[1].plot(x, m2 * x + b2, color='red')  # Regression line
    axes[1].set_title('Russian Dependency vs Post-War GDP Change')
    axes[1].set_xlabel('Russian Dependency 2021')
    axes[1].set_ylabel('Post-War GDP Change')

    # Improve layout and display the plot
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
m1 = folium.Map(location=[54, 15], tiles="OpenStreetMap", zoom_start=4)

# Function to create a popup for each country
def make_popup(country_row):
    return folium.Popup(f"Country: {country_row['Country']}<br>"
                        f"RussianDependency2021: {country_row['RussianDependency2021']}<br>"
                        f"PostWarINFChange: {country_row['PostWarINFChange']}<br>"
                        f"PostWarGDPChange: {country_row['PostWarGDPChange']}", max_width=300)

# Add markers to the map
for idx, row in Main.iterrows():
    country = row['Country']
    coordinates = country_coordinates.get(country)
    if coordinates:
        folium.Marker(
            location=coordinates,
            popup=make_popup(row),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m1)

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
m2 = folium.Map(location=[54, 15], zoom_start=4)

# Create a Choropleth map where the color intensity is based on the inflation change (INFChange)
folium.Choropleth(
    geo_data=eu_country_geo,  
    name='choropleth',
    data=Main,  
    columns=['Country_Code', 'RussianDependency2021'],  
    key_on='feature.properties.NUTS_ID',  
    fill_color='YlOrRd',  
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='RussianDependency2021'
).add_to(m2)

# Initialize the maps centered around a point
m3 = folium.Map(location=[54, 15], zoom_start=4)

folium.Choropleth(
    geo_data=eu_country_geo,  
    name='choropleth',
    data=Main,  
    columns=['Country_Code', 'PostWarINFChange'],  
    key_on='feature.properties.NUTS_ID', 
    fill_color='YlOrRd', 
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Inflation Change'
).add_to(m3)
