import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import os
# xgboost
import xgboost as xgb
# streamlit and folium
import streamlit_folium as st_folium
from streamlit_folium import folium_static
import folium 
#geopandas
import geopandas as gpd 

# make the page wider 
st.set_page_config(layout="wide")

"""
# Airbnb market in Austin, TX
Estimate your income and research the market
"""
st.text('Based on 2023 data')

st.divider()  # horizontal line


## load models and data
xgb_price = pickle.load(open('./../model_files/xgb_price.pkl', "rb"))
xgb_occupancy_rate = pickle.load(open('./../model_files/xgb_occupancy_rate.pkl', "rb"))
home_price = pickle.load(open('./../model_files/home_price.pkl', "rb"))
df = pd.read_csv('./../data/austin_listings_clean.csv').drop(columns=['price', 'occupancy_rate'])


# related lists and variables
zipcodes = [78702, 78704, 78741, 78745, 78703, 78731, 78705, 78727, 78751,
       78722, 78701, 78723, 78758, 78757, 78724, 78746, 78736, 78752,
       78721, 78756, 78733, 78737, 78744, 78726, 78754, 78738, 78735,
       78759, 78732, 78748, 78729, 78753, 78728, 78717, 78734, 78749,
       78750, 78730, 78739, 78725, 78747, 78742, 78719]
zipcodes.sort()
num_bedrooms = range(8)
# num_beds = range(16)
# num_baths = range(9)

data = np.random.randn(10, 1)
a = 5

# HTML Elements used
# Define a function to create colored boxes with text
def colored_box(text, color):
    return f'<div style="background-color: {color};font-size:120%; color:black; text-align:center; padding: 10px; border-radius: 5px; margin-bottom: 10px;">{text}</div>'


#=========> Estimator tool <============
st.header("Estimator tool")
col1, col_1, col2,col_2, col3 = st.columns([1,0.25, 1.5, 0.1, 3])

#==========> column 1
col1.write('#### Home information')
# Create a dropdown menu with some options
col1.write('')
selected_zipcode = col1.selectbox("Zipcode:", zipcodes)
selected_bedroom = col1.selectbox("Bedrooms:", num_bedrooms)

# build a dataframe for the date of interest from our observations
focus_data = df[(df['zipcode'] == selected_zipcode) & (df['bedrooms'] == selected_bedroom)]
number_of_listings = int(focus_data.shape[0]/4)
# Use calculated 
col1.write('###### Number of homes:')
col1.write(f'##### {number_of_listings}')
if number_of_listings < 15: col1.warning('Too few observations for reliable modeling!', icon="⚠️")

#estimate the realistic number of beds and baths
# Also, for other features, to avoid extrapolation and getting none-sense values, use the majority class
selected_bed = focus_data['beds'].mean()
selected_bath = focus_data['bath'].mean()
superhost = focus_data['host_is_superhost'].value_counts().sort_values(ascending=False).index[0]
review = focus_data['review_scores_rating'].mean()
pool = focus_data['has_pool'].value_counts().sort_values(ascending=False).index[0]
petfriendly = focus_data['is_petfriendly'].value_counts().sort_values(ascending=False).index[0]
workspace = focus_data['has_workspace'].value_counts().sort_values(ascending=False).index[0]
parking = focus_data['has_freeparking'].value_counts().sort_values(ascending=False).index[0]
gym = focus_data['has_gym'].value_counts().sort_values(ascending=False).index[0]
# do estimate of the price based on bedrooms, zip and other features (found based on zip and bedrooms)
X_test = []
for i in ['Q1', 'Q2', 'Q3', 'Q4']:
    X_test.append([selected_bedroom, selected_bed, selected_bath, selected_zipcode,
                    superhost, review, i, pool, petfriendly, workspace, parking, gym, 
                    home_price[selected_zipcode][min(max(selected_bedroom, 1),5)]])
# convert the X variables to dataframe
X_test = pd.DataFrame(X_test, columns=['bedrooms', 'beds', 'bath', 'zipcode', 'host_is_superhost',
       'review_scores_rating', 'time_quarter', 'has_pool', 'is_petfriendly',
       'has_workspace', 'has_freeparking', 'has_gym', 'home_price_aprx'])
# convert type to category for categorical features
X_test[['zipcode', 'time_quarter']] = X_test[['zipcode', 'time_quarter']].astype('category')
# estimate the occupancy rate, price, and income
occupancy_rate = xgb_occupancy_rate.predict(X_test)
# set min and max for model prediction
occupancy_rate = np.clip(occupancy_rate, 0, 0.7)
price = xgb_price.predict(X_test)
income = sum(price * occupancy_rate * 90)
income_array = price * occupancy_rate * 90
home_value = home_price[selected_zipcode][min(max(selected_bedroom, 1),5)]
X_test['occupancy_rate'] = occupancy_rate
X_test['income'] = income_array

#==========> column 2
col2.write('')
col2.write('')
col2.write(f'##### Yearly income: ')
col2.markdown(colored_box(f"<b>${income:,.0f}<b>", "white"), unsafe_allow_html=True)
col2.write('')
col2.write('')
col2.write(f'##### Occupancy rate: ')
col2.markdown(colored_box(f"<b>{np.mean(occupancy_rate):.2f}<b>", "white"), unsafe_allow_html=True)
col2.write('')
col2.write('')
col2.write(f'##### Yearly income (per 100k investment): ')
col2.markdown(colored_box(f"<b>${income/home_value*100000:,.0f}<b>", "white"), unsafe_allow_html=True)

#==========> column 3
#col3.write('#### Home information')
plot_var = col3.selectbox("", ['Average income', 'Occupancy rate'])
#Plot the altair chart
if plot_var == 'Average income':
    chart = (
            alt.Chart(
                data=X_test,
                #title="Income at different quarters",
            )
            .mark_bar()
            .encode(
                x= alt.X('time_quarter', title='Time (quarter)'),
                y=alt.Y('income', title='Quarterly income, $')
            )
            .properties(
                width=650,  # Set the width of the plot
                height=350 
            )
            .configure_axis(
                labelAngle=0  # Set the angle of the x-axis labels
            )
    )
else: 
    chart = (
            alt.Chart(
                data=X_test,
            #    title="Your title",
            )
            .mark_bar()
            .encode(
                x= alt.X('time_quarter', title='Time (quarter)'),
                y=alt.Y('occupancy_rate', title='Occupancy rate')
            )
            .properties(
                width=650,  # Set the width of the plot
                height=350 
            )
            .configure_axis(
                labelAngle=0  # Set the angle of the x-axis labels
            )
    )

col3.altair_chart(chart)

st.divider()  # horizontal line

#=========> Estimator tool <============
df_all = pd.read_csv('./../data/austin_listings_clean.csv')
df_all['quarterly_income'] = df_all['price'] * df_all['occupancy_rate'] * 90
df_all['quarterly_income_per100'] = df_all['price'] * df_all['occupancy_rate'] * 90 / df_all['home_price_aprx'] * 100_000

st.header("Market research tool")

col1, col_1, col2,col_2, col3, col3_ = st.columns([2,0.15, 1.5, 0.5, 2.5, 0.75])

#==========> column 1
#col1.write('#### Zip code:')
# Create a dropdown menu with some options
selected_zipcode = col1.selectbox("Zip code", zipcodes)
# filter the data to the zip code of interest
df_all = df_all[df_all['zipcode'] == selected_zipcode]

## city map
map = folium.Map(location=[30.29, -97.74], zoom_start=10.4)
city = gpd.read_file('./../data/neighbourhoods.geojson')
city = city[city['neighbourhood'] == str(selected_zipcode)]
folium.Choropleth(
    geo_data=city,
    # create the data that should be used as an overlay
    #data=city,
    # more on how to plot https://python-visualization.github.io/folium/latest/user_guide/geojson/choropleth.html
    #columns=["zipcode", "retrun_per100k"],
    # by looking inside the json file, see where zipcode data is located and address it here
    #key_on="feature.properties.neighbourhood",
    # set the colors OrangeRed
    # more info on colors https://python-visualization.github.io/folium/latest/advanced_guide/colormaps.html 
    # https://python-visualization.github.io/folium/latest/user_guide/geojson/geojson.html
    fill_opacity=0.3,
    line_weight=2,
).add_to(map)


# use with to set the folium map in a column
with col1:
    output = folium_static(map, width=380, height=400)
number_of_homes = int(df_all.shape[0]/4)
col1.write('###### Number of homes:')
col1.write(f'##### {number_of_homes}')
if number_of_homes < 50: col1.warning('Too few observations for reliable estimates!', icon="⚠️")


#=========> column 2
### PLOT 1###
# calculations for plotting yearly income with number of beds
# groupby the datafram by the variable of interest for plotting and also the time of year
# to calculate summation of incomes for this variable at all times of the year
bed_quarter_income = df_all.groupby(['host_is_superhost', 'time_quarter'])[['quarterly_income']].mean()
bed_yearly_income = []
# create a list of list with number of bedrooms and summation of incomes for all quarters
# we will be looping through the first index (number of beds and pick up them in groups of 4)
# this approach is used due to a problem with .loc and streamlit.
for i in range(8):
    temp = bed_quarter_income.iloc[4*i:4*(i+1)]
    bed_yearly_income.append([i, temp.sum().values[0]])
bed_yearly_income = pd.DataFrame(bed_yearly_income, columns=['host_is_superhost', 'yearly_income'])
bed_yearly_income['host_is_superhost'] = bed_yearly_income['host_is_superhost'].astype(str)
bed_yearly_income = bed_yearly_income[bed_yearly_income['yearly_income']>0]
chart = (
        alt.Chart(
            data=bed_yearly_income,
            title=f"Income vs. host experience",
        )
        .mark_bar()
        .encode(
            x= alt.X('host_is_superhost', title='Is superhost'),
            y=alt.Y('yearly_income', title='Yearly income, $')
        )
        .properties(
            width=350,  # Set the width of the plot
            height=350 
        )
        .configure_axis(
            labelAngle=0  # Set the angle of the x-axis labels
        ).configure_title(
            fontSize=20,  # Set the font size of the title
            anchor='middle'
        )
        
)
col2.altair_chart(chart)

### PLOT 2###
# calculations for plotting yearly income with number of beds
# groupby the datafram by the variable of interest for plotting and also the time of year
# to calculate summation of incomes for this variable at all times of the year
bed_quarter_income = df_all.groupby(['has_pool', 'time_quarter'])[['quarterly_income']].mean()
bed_yearly_income = []
# create a list of list with number of bedrooms and summation of incomes for all quarters
# we will be looping through the first index (number of beds and pick up them in groups of 4)
# this approach is used due to a problem with .loc and streamlit.
for i in range(8):
    temp = bed_quarter_income.iloc[4*i:4*(i+1)]
    bed_yearly_income.append([i, temp.sum().values[0]])
bed_yearly_income = pd.DataFrame(bed_yearly_income, columns=['has_pool', 'yearly_income'])
bed_yearly_income['has_pool'] = bed_yearly_income['has_pool'].astype(str)
bed_yearly_income = bed_yearly_income[bed_yearly_income['yearly_income']>0]
chart = (
        alt.Chart(
            data=bed_yearly_income,
            title=f"Income vs. having pool",
        )
        .mark_bar()
        .encode(
            x= alt.X('has_pool', title='Has pool'),
            y=alt.Y('yearly_income', title='Yearly income, $')
        )
        .properties(
            width=350,  # Set the width of the plot
            height=350 
        )
        .configure_axis(
            labelAngle=0  # Set the angle of the x-axis labels
        ).configure_title(
            fontSize=20,  # Set the font size of the title
            anchor='middle'
        )
        
)
col2.altair_chart(chart)



#=========> column 3
### PLOT 1###
# calculations for plotting yearly income with number of beds
# groupby the datafram by the variable of interest for plotting and also the time of year
# to calculate summation of incomes for this variable at all times of the year
bed_quarter_income = df_all.groupby(['bedrooms', 'time_quarter'])[['quarterly_income']].mean()
bed_yearly_income = []
# create a list of list with number of bedrooms and summation of incomes for all quarters
# we will be looping through the first index (number of beds and pick up them in groups of 4)
# this approach is used due to a problem with .loc and streamlit.
for i in range(8):
    temp = bed_quarter_income.iloc[4*i:4*(i+1)]
    bed_yearly_income.append([i, temp.sum().values[0]])
bed_yearly_income = pd.DataFrame(bed_yearly_income, columns=['bedrooms', 'yearly_income'])
bed_yearly_income['bedrooms'] = bed_yearly_income['bedrooms'].astype(str)
bed_yearly_income = bed_yearly_income[bed_yearly_income['yearly_income']>0]
chart = (
        alt.Chart(
            data=bed_yearly_income,
            title=f"Yearly income vs. bedrooms",
        )
        .mark_bar()
        .encode(
            x= alt.X('bedrooms', title='Number of bedrooms'),
            y=alt.Y('yearly_income', title='Yearly income, $')
        )
        .properties(
            width=650,  # Set the width of the plot
            height=350 
        )
        .configure_axis(
            labelAngle=0  # Set the angle of the x-axis labels
        ).configure_title(
            fontSize=20,  # Set the font size of the title
            anchor='middle'
        )
        
)
col3.altair_chart(chart)

### PLOT 2###
# calculations for plotting yearly income with number of beds
# groupby the datafram by the variable of interest for plotting and also the time of year
# to calculate summation of incomes for this variable at all times of the year
bed_quarter_inv = df_all.groupby(['bedrooms', 'time_quarter'])[['quarterly_income_per100']].mean()
bed_yearly_inv = []
# create a list of list with number of bedrooms and summation of incomes for all quarters
# we will be looping through the first index (number of beds and pick up them in groups of 4)
# this approach is used due to a problem with .loc and streamlit.
for i in range(8):
    temp = bed_quarter_inv.iloc[4*i:4*(i+1)]
    bed_yearly_inv.append([i, temp.sum().values[0]])
bed_yearly_inv = pd.DataFrame(bed_yearly_inv, columns=['bedrooms', 'quarterly_income_per100'])
bed_yearly_inv['bedrooms'] = bed_yearly_inv['bedrooms'].astype(str)
bed_yearly_inv = bed_yearly_inv[bed_yearly_inv['quarterly_income_per100']>0]
chart = (
        alt.Chart(
            data=bed_yearly_inv,
            title=f"Income per 100k investment",
        )
        .mark_bar()
        .encode(
            x= alt.X('bedrooms', title='Number of bedrooms'),
            y=alt.Y('quarterly_income_per100', title='Yearly income, $')
        )
        .properties(
            width=650,  # Set the width of the plot
            height=350 
        )
        .configure_axis(
            labelAngle=0  # Set the angle of the x-axis labels
        ).configure_title(
            fontSize=20,  # Set the font size of the title
            anchor='middle'
        )
        
)
col3.altair_chart(chart)

st.text('This app is developed by Masoud "Massi" Alfi to help the investors find the best investment opportunities in Austin Texas.')
st.text('data source:')
st.text('- www.insideairbnb.com')
st.text('- www.zillow.com/data')

