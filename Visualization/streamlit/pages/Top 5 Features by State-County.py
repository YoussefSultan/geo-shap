import streamlit as st
import pandas as pd
import os
import pickle
import plotly.express as px
import json
import numpy as np

global_path = os.path.dirname(os.getcwd()) if os.getcwd() != '/mount/src' else '/mount/src'
st.set_page_config(layout="wide")

with open(global_path + '/visual_datasets_updated.pkl', 'rb') as fp:
    visual_datasets = pickle.load(fp)
with open('geojson-counties-fips.json', 'r') as f:
    geojson = json.load(f)

unique_years = list(visual_datasets.keys())
state_counts = {year: visual_datasets[year]['shap_values']['State'].unique() for year in unique_years}
county_counts = {year: {state: visual_datasets[year]['shap_values'].loc[visual_datasets[year]['shap_values']['State'] == state]['County'].unique() for state in state_counts[year]} for year in unique_years}

st.markdown("### Exploring the Top 5 Features Influencing County-Level Diabetes Prevalence")

with st.expander('## More Information:'):
    st.markdown('''
    We **introduce** a nuanced view of contributors to diabetes prevalence at the 
    geographic level and preserve spatial variance when imputing missing values through
    KNN imputation by conversion of categorical county level to latitude and longititude 
    continuous features before the Euclidean norm calculations:

    - **Data Collection:** We combine the NHANES dataset—which provides information on food security and 
                demographics—with the CDC's dataset on Diabetes. These datasets, publicly accessible through government 
                portals, have multi-year county-level information. Data for each indicator was collected separately for each 
                year from 2012 to 2021. Each dataset is merged based on a common key. This gives us diverse indicators across years, 
                yet does missing values year by year which are imputed using KNNImputation which is a Euclidean approach that is applied 
                with latitude and longitude in the calculation instead of ordinally or onehot encoded geographic features.

    - **Annual Model Training:** For each year an XGBoost Model is trained on the imputed and post-processed dataset with key features 
                selected based on an in-depth analysis. 
                This enables a nuanced understanding of trends and patterns that may evolve over time.

    - **SHAP Value Extraction:** Each model is then put in a SHAP explainer and SHAP values are captured. 
                This shows the contribution of each feature to the model's predictions, however it is at county level 
                so we can see how a specific feature may impact prevalence in one county in the model higher than another.


    The main idea is to understand not only how features impact diabetes prevalence overall in the US for each year, but how certain features impact the prevalence 
                differently based on the geographic location. This geographic variance is preserved by the conversion of categorical geographic data to continuous 
                latitude and longitude values for the imputation of missing values. The models are not trained on latitude or longitude as features.
                
                
    The location is preserved because we reappend/concatenate the county level data on the SHAP values data since the indices are preserved, 
                thus this allows us to understand for that specific record where the location was. 
                
                ''')

c1, c2 = st.columns([3, 1])

with c2:
    selected_year = st.selectbox('Select Dataset Year', unique_years, index=0)
    selected_state = st.selectbox('Select a State', sorted(state_counts[selected_year]))
    selected_county = st.selectbox('Select a County', sorted(county_counts[selected_year][selected_state]))

with c1:
    custom_color_scale = [
        [0, 'rgb(30, 136, 229)'],
        [1, 'rgb(255, 13, 87)']
    ]

    data = visual_datasets[selected_year]['shap_values']
    data_selected_state = data[data['State'] == selected_state]
    data_selected_county = data_selected_state[data_selected_state['County'] == selected_county]

    selected_state_code = data_selected_state['County_FIPS'].str[:2].iloc[0]

    geojson_selected_state = {
        'type': 'FeatureCollection',
        'features': [feature for feature in geojson['features'] if feature['properties']['STATE'] == selected_state_code]
    }

    fig = px.choropleth(
        data_selected_state,
        geojson=geojson_selected_state,
        locations='County_FIPS',
        color='DiagnosedDiabetes(Percentage)',
        color_continuous_scale=custom_color_scale,
        fitbounds="geojson",
        labels={'Top_Features': 'Top 5 Features'},
        hover_data=['County', 'Top_Features']
    )

    fig.update_layout(
        height=1100,
        width=1100,
        margin={"r": 0, "t": 0, "l": 0, "b": 500},
    coloraxis_colorbar=dict(
        len=0.9,
        x=1,
        y=0.5,
        xpad=25,
        ypad=0
        )
    )

    st.plotly_chart(fig)

with c2:
    top_features = data_selected_county['Top_Features'].iloc[0][:5]
    st.markdown(f"#### Top 5 Features for {selected_county}")
    for i, feature in enumerate(top_features, start=1):
        st.write(f"{i}. {feature}")
