import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import plotly.express as px
import altair as alt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import dodge
from bokeh.plotting import figure, show, output_notebook
from bokeh.palettes import Spectral11
import geopandas as gpd
from pyproj import Proj, transform
import bar_chart_race as bcr

# Read the CSV file
data = pd.read_csv("effectifs.csv", sep=";")

# Filtrer pour une tranche d'âge spécifique (par exemple, 20-24 ans)
selected_age_group = "20-24"
filtered_data = data[data["libelle_classe_age"] == selected_age_group]

print(filtered_data)
