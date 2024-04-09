import streamlit as st
import pandas as pd
import os
import pickle
import plotly.express as px
import json 
import numpy as np

# finds the global path that contains the data files
global_path = os.path.dirname(os.getcwd()) if os.getcwd() != '/mount/src' else '/data'
st.set_page_config(layout="wide")


# load data
with open(global_path + '/visual_datasets_040624.pkl', 'rb') as fp:
    visual_datasets = pickle.load(fp)
with open('/geojson-counties-fips.json', 'r') as f:
    geojson = json.load(f)

# unique years list
unique_years = [year for year in visual_datasets.keys()]

st.markdown("### Exploring SHAP Values contributing to Diabetes Prevalence at the County Level and comparing with raw data")

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

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)


with c7:
    # Year selection in sidebar
    selected_year = st.selectbox(
        'Select Dataset Year',
        unique_years,
        index=0  # Defaults to the first year in the list
    )

    # Use the selected year to load the corresponding dataset
    data = visual_datasets[selected_year]['shap_values']
    min_shap_value = np.array(data.iloc[:,:-6].min().to_list()).min()
    max_shap_value = np.array(data.iloc[:,:-6].max().to_list()).max()


    # columns to select for drop down
    columns_of_choice = visual_datasets[selected_year]['shap_values'].columns[:-6].to_list()
    max_values_df = data[columns_of_choice].max().reset_index()
    max_values_df.columns = ['Feature', 'SHAP Value']
    max_values_df = max_values_df.sort_values(by='SHAP Value', ascending=False)
    max_values_df.Feature.to_list()


    # Factor selection in sidebar
    selected_factor = st.selectbox(
        'Select a factor contributing to diabetes to display its SHAP value',
    max_values_df.Feature.to_list()) # drop down has the features sorted from max shap to least value first



# min and max shap value for current year
with c7:
    st.metric("Minimum Value", round(min_shap_value,2), "SHAP Value", delta_color="inverse")
    st.metric("Maximum Value", round(max_shap_value,2), "SHAP Value", delta_color="inverse")


with c1:
    st.markdown('Top Features with Highest SHAP Value For Selected Year')
    st.dataframe(max_values_df)

with c2:
# replicate shap color scale
    custom_color_scale = [
    [0, 'rgb(30, 136, 229)'],
    [1, 'rgb(255, 13, 87)'] # Dark red
      # Dark blue
    ]
# Create choropleth map using the selected factor and data for the chosen year
    fig = px.choropleth(
        data,
        geojson=geojson,  
        locations='County_FIPS',
        color=selected_factor,
        color_continuous_scale=custom_color_scale,
        scope="usa",
        labels={selected_factor: selected_factor},
        hover_data=['County', 'State', selected_factor, 'DiagnosedDiabetes(Percentage)'],
        width=1400, height=500)

    color_bar_title = "SHAP Value: " + selected_factor

    # feel free to edit this
    fig.update_layout(
    title={
        'text': f"{selected_factor} SHAP Value County Level Impact on Diabetes Prevalence",
        'y':.97,
        'x':0.44,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'color':'black',
            'size':14
        }
    },
    margin={"r":0,"t":0,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=color_bar_title,
        x=.93, # Adjusts the horizontal position (try values between -0.1 and 1.1 to see what works best for your layout)
        y=0.5, # Adjusts the vertical position (0.5 is the middle of the plot)
        xpad=22, # Adjusts the padding from the plot on the x-axis
        ypad=0  # Adjusts the padding from the plot on the y-axis
    )
)

    # Display the plotly choropleth map in the main page area
    st.plotly_chart(fig)

with c7:
    # Use the selected year to load the corresponding dataset
    data2 = visual_datasets[selected_year]['processed_data']
    min_value = np.array(data2.loc[:,selected_factor].min()).min()
    max_value = np.array(data2.loc[:,selected_factor].max()).max()


    # columns to select for drop down
    columns_of_choice = visual_datasets[selected_year]['processed_data'].columns[:-6].to_list()
    max_values_df2 = data2[columns_of_choice].max().reset_index()
    max_values_df2.columns = ['Feature', 'Value']
    max_values_df2 = max_values_df2.sort_values(by='Value', ascending=False)

    st.metric("Minimum Value", round(min_value,2), "Selected Feature Value", delta_color="inverse")
    st.metric("Maximum Value", round(max_value,2), "Selected Feature Value", delta_color="inverse")



with c2:
    
# replicate shap color scale
    custom_color_scale = [
    [0, 'rgb(30, 136, 229)'],
    [1, 'rgb(255, 13, 87)'] # Dark red
      # Dark blue
    ]
# Create choropleth map using the selected factor and data for the chosen year
    fig2 = px.choropleth(
        data2,
        geojson=geojson,  
        locations='County_FIPS',
        color=selected_factor,
        color_continuous_scale=custom_color_scale,
        scope="usa",
        labels={selected_factor: selected_factor},
        hover_data=['County', 'State', selected_factor, 'DiagnosedDiabetes(Percentage)'],
        width=1150, height=350,
    )

    color_bar_title = selected_factor

    # feel free to edit this
    fig2.update_layout(
    title={
        'text': f"{selected_factor} Raw Value",
        'y':.97,
        'x':0.44,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'color':'black',
            'size':12
        }
    },
    margin={"r":0,"t":0,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=color_bar_title,
        x=.88, # Adjusts the horizontal position (try values between -0.1 and 1.1 to see what works best for your layout)
        y=0.5, # Adjusts the vertical position (0.5 is the middle of the plot)
        xpad=22, # Adjusts the padding from the plot on the x-axis
        ypad=0  # Adjusts the padding from the plot on the y-axis
    )
)

    st.plotly_chart(fig2)