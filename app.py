import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px
import altair as alt
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Spectral11
from PIL import Image


# Read the CSV file
data = pd.read_csv("effectifs.csv", sep=";")

# Chargez votre image
image = Image.open("sante.png")
# Ajoutez l'image à la sidebar
st.sidebar.image(image, use_column_width=True)


st.sidebar.text("Médéric ZHOU SUN")
st.sidebar.text("EFREI - Promo 2025")

# Progress bar
progress_bar = st.empty()

# Application title
st.title("Analysis of Patient Demographics and Prevalence for Various Medical Conditions in France (2015-2021)")

image = Image.open("image.jpg")
st.image(image, use_column_width=True)

# Selection bar to choose the category
category = st.sidebar.selectbox("Select a category", ["Data Exploration", "Seaborn Charts", "Matplotlib Charts", "Plotly Express Charts", "Altair Charts", "Bokeh Charts", "Advanced Time Series Charts", "Graphiques à Secteurs"])

# Sidebar
st.sidebar.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-MedPrice-blue?style=for-the-badge&logo=github)](https://github.com/AeroBaze/MedPrice)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Médéric%20ZHOU%20SUN-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mederic-zhousun)
    """
)


# Simulated loading function
def simulate_loading():
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)


def time_series_trend_chart(data, x, y, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x], data[y], marker='o', linestyle='-', markersize=8, color='b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    st.pyplot(plt)


def create_pie_chart(data, title):
    fig = px.pie(data, names='patho_niv1', title=title)
    return fig


# Home page
if category == "Data Exploration":
    progress_bar.text("Loading in progress...")
    simulate_loading()

    st.subheader("Data Overview")
    st.write(data.head())

    st.subheader("Statistical Description")
    st.write(data.describe())

    st.subheader("Available Columns")
    st.write(data.columns)

    # - Correlation matrix for numeric columns
    st.subheader("Correlation Matrix for Numeric Columns")
    numeric_data = data.select_dtypes(include='number')
    corr_matrix = numeric_data.corr()
    st.write(corr_matrix)

    # - Filtering and sorting options for a single column
    st.subheader("Filtering and Sorting Options")

    column_to_sort = st.selectbox("Select a column to sort by", data.columns)
    ascending = st.checkbox("Sort in ascending order")

    # Apply filtering and sorting
    # Filter by column
    filtered_data = data  # Initialize with the original data

    # Sorting
    if ascending:
        sorted_data = filtered_data.sort_values(by=column_to_sort, ascending=True)
    else:
        sorted_data = filtered_data.sort_values(by=column_to_sort, ascending=False)

    # Display limited results
    # Display the first 20 rows for demonstration purposes
    st.write(sorted_data.head(20))

    # - Summary statistics for specific columns
    st.subheader("Summary Statistics for Specific Columns")
    selected_columns = st.multiselect("Select columns for summary statistics", data.columns)
    if selected_columns:
        summary = data[selected_columns].describe()
        st.write(summary)


elif category == "Seaborn Charts":
    progress_bar.text("Loading in progress...")
    simulate_loading()

    # Gender distribution
    st.subheader("Gender Distribution")
    gender_counts = data['libelle_sexe'].value_counts()
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax)
    st.pyplot(fig)

    # Age distribution
    st.subheader("Age Distribution")
    age_counts = data['libelle_classe_age'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=age_counts.index, y=age_counts.values, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Year distribution
    st.subheader("Year Distribution")
    year_counts = data['annee'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Exclude non-numeric columns for correlation analysis
    numeric_data = data.select_dtypes(include=['number'])

    # Compute correlation matrix
    correlation_matrix = numeric_data.corr()

    # Create a heatmap of the correlation matrix
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot(plt)

    # Filter for specific years (remove NaN values and convert to integers)
    years = data['annee'].dropna().astype(int).unique()
    years = sorted(years)  # Sort years in ascending order
    selected_year = int(st.sidebar.selectbox("Select a year", years))
    filtered_data = data[data["annee"] == selected_year]

    # Filter data for the selected year and age group
    selected_age_group = st.selectbox("Select an age group", data['libelle_classe_age'].unique())
    filtered_data_age_group = filtered_data[filtered_data['libelle_classe_age'] == selected_age_group]

    # Group data by disease and calculate the sum of 'npop' for each disease
    disease_distribution = filtered_data_age_group.groupby('patho_niv1')['npop'].sum().reset_index()

    # Sort the diseases by population size in descending order
    disease_distribution = disease_distribution.sort_values(by='npop', ascending=False)

    # Create a bar plot of disease distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(data=disease_distribution, x='npop', y='patho_niv1')
    plt.xlabel('Population')
    plt.ylabel('Disease')
    plt.title(f'Disease Distribution for {selected_age_group} in {selected_year}')
    st.pyplot(plt)


elif category == "Matplotlib Charts":
    progress_bar.text("Loading in progress...")
    simulate_loading()

    # Filter for specific years (remove NaN values and convert to integers)
    years = data['annee'].dropna().astype(int).unique()
    years = sorted(years)  # Sort years in ascending order
    selected_year = int(st.sidebar.selectbox("Select a year", years))
    filtered_data = data[data["annee"] == selected_year]

    # Gender distribution for the selected year
    st.subheader(f"Gender Distribution for {selected_year}")
    gender_counts = filtered_data['libelle_sexe'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(gender_counts.index, gender_counts.values)
    plt.xticks(rotation=0)
    st.pyplot(plt)

    # Age distribution for the selected year
    st.subheader(f"Age Distribution for {selected_year}")
    age_counts = filtered_data['libelle_classe_age'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(age_counts.index, age_counts.values)
    plt.xticks(rotation=90)
    st.pyplot(plt)

    # Year distribution (you may choose to update or remove this section as it's not specific to the selected year)
    st.subheader("Year Distribution")
    year_counts = filtered_data['annee'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(year_counts.index, year_counts.values)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Scatter Plot to analyze relationships between numerical variables
    st.subheader("Scatter Plot between 'npop' and 'prev'")
    plt.figure(figsize=(8, 6))
    plt.scatter(data['npop'], data['prev'], alpha=0.5)
    plt.xlabel('npop')
    plt.ylabel('prev')
    st.pyplot(plt)

    # Filter data for the selected age group
    selected_age_group = st.selectbox("Select an age group", data['libelle_classe_age'].unique())
    filtered_data_age_group = data[data['libelle_classe_age'] == selected_age_group]

    # Group data by year and disease and calculate the sum of 'npop' for each combination
    disease_evolution = filtered_data_age_group.groupby(['annee', 'patho_niv1'])['npop'].sum().unstack().T

    # Create a stacked bar plot for disease evolution
    plt.figure(figsize=(12, 6))
    disease_evolution.plot(kind='bar', stacked=True)
    plt.xlabel('Disease')
    plt.ylabel('Population')
    plt.title(f'Disease Evolution Over the Years for {selected_age_group}')
    st.pyplot(plt)


elif category == "Plotly Express Charts":
    progress_bar.text("Loading in progress...")
    simulate_loading()

     # Example 3: Line Chart
    st.subheader("Example 3: Line Chart - Year Distribution")
    year_counts = data['annee'].value_counts().reset_index()
    year_counts.columns = ['Year', 'Count']
    fig = px.line(year_counts, x='Year', y='Count', labels={'Year':'Year', 'Count':'Count'})
    st.plotly_chart(fig)


elif category == "Altair Charts":
    progress_bar.text("Loading in progress...")
    simulate_loading()

    # Filter for specific years (remove NaN values and convert to integers)
    years = data['annee'].dropna().astype(int).unique()
    years = sorted(years)  # Sort years in ascending order
    selected_year = int(st.sidebar.selectbox("Select a year", years))
    filtered_data = data[data["annee"] == selected_year]

    st.subheader("Chart 1: Example Altair Chart")
    chart1 = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X("patho_niv1:N", title="Disease Category"),
        y=alt.Y("mean(prev):Q", title="Average Prevalence"),
        color=alt.Color("libelle_sexe:N", title="Gender")
    ).properties(width=600, height=400)
    st.altair_chart(chart1)

    st.subheader("Chart 2: Example Altair Chart")
    chart2 = alt.Chart(filtered_data).mark_circle().encode(
        x=alt.X("cla_age_5:N", title="Age Group"),
        y=alt.Y("mean(prev):Q", title="Average Prevalence"),
        color=alt.Color("libelle_sexe:N", title="Gender"),
        size=alt.Size("mean(npop):Q", title="Average Population"),
        tooltip=["cla_age_5", "mean(prev)", "mean(npop)"]
    ).properties(width=600, height=400)
    st.altair_chart(chart2)


elif category == "Bokeh Charts":
    progress_bar.text("Loading in progress...")
    simulate_loading()

    # Filter data for the selected age group and disease
    selected_age_group = st.selectbox("Select an age group", data['libelle_classe_age'].unique())
    selected_disease = st.selectbox("Select a disease", data['patho_niv1'].unique())
    filtered_data_age_group = data[data['libelle_classe_age'] == selected_age_group]
    filtered_data_disease = filtered_data_age_group[filtered_data_age_group['patho_niv1'] == selected_disease]

    # Group data by year and calculate the sum of 'npop' for each year
    disease_evolution = filtered_data_disease.groupby('annee')['npop'].sum().reset_index()

    # Create a line plot for disease evolution
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=disease_evolution, x='annee', y='npop')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Disease Evolution for {selected_disease} in {selected_age_group}')
    st.pyplot(plt)


# Create a section for advanced time series charts
elif category == "Advanced Time Series Charts":
    progress_bar.text("Loading in progress...")
    simulate_loading()

    st.subheader("Time Series Trend for Prevalence")

    # Group data by year and calculate the mean prevalence
    yearly_prevalence = data.groupby('annee')['prev'].mean().reset_index()

    time_series_trend_chart(
        data=yearly_prevalence,
        x='annee',
        y='prev',
        title='Time Series Trend of Prevalence Over the Years',
        x_label='Year',
        y_label='Mean Prevalence'
    )

# Page de graphiques à secteurs
elif category == "Graphiques à Secteurs":
    progress_bar.text("Loading in progress...")
    simulate_loading()

    st.subheader("Distribution des Patients par Catégorie de Pathologie")
    pie_chart_data = data.groupby('patho_niv1').size().reset_index(name='count')
    pie_chart = create_pie_chart(pie_chart_data, "Répartition des Patients par Catégorie de Pathologie")
    st.plotly_chart(pie_chart)


# Comments section
st.subheader("Leave Your Comments")
comment = st.text_area("Leave your comments here", "")
if st.button("Submit Comment"):
    st.write("Comment submitted: ", comment)

