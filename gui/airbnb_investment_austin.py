import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import os
# xgboost
import xgboost as xgb

# make the page wider 
st.set_page_config(layout="wide")

"""
# Airbnb market in Austin (2023 data)

- Use the 'Estimator tool' to estimate airbnb income and occupancy rate for your home.
- Use the 'Market research tool' to learn more about Austin's airbnb market.
"""


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
# selected_zipcode = 78759
# selected_bedroom = 1

selected_bedroom = col1.selectbox("Bedrooms:", num_bedrooms)
# selected_bed = col1.selectbox("Beds:", num_beds)
# selected_bath = col1.selectbox("Baths:", num_baths)

focus_data = df[(df['zipcode'] == selected_zipcode) & (df['bedrooms'] == selected_bedroom)]
number_of_listings = focus_data.shape[0]
col1.write('###### Number of listings:')
col1.write(f'##### {number_of_listings}')
if number_of_listings < 50: col1.warning('Too few observations for reliable modeling', icon="⚠️")

# do estimations low
selected_bed = focus_data['beds'].mean()
selected_bath = focus_data['bath'].mean()
X_test = []
for i in ['Q1', 'Q2', 'Q3', 'Q4']:
    X_test.append([selected_bedroom, selected_bed, selected_bath, selected_zipcode,
                    0, 4.9, i, 0, 0, 0, 0, 0, 
                    home_price[selected_zipcode][min(max(selected_bedroom, 1),5)]])
# convert the X variables to dataframe
X_test = pd.DataFrame(X_test, columns=['bedrooms', 'beds', 'bath', 'zipcode', 'host_is_superhost',
       'review_scores_rating', 'time_quarter', 'has_pool', 'is_petfriendly',
       'has_workspace', 'has_freeparking', 'has_gym', 'home_price_aprx'])
# convert type to category for categorical features
X_test[['zipcode', 'time_quarter']] = X_test[['zipcode', 'time_quarter']].astype('category')


occupancy_rate = xgb_occupancy_rate.predict(X_test)
price = xgb_price.predict(X_test)

income_low = sum(price * occupancy_rate * 90)

#==========> column 2
col2.write(f'##### Yearly income (low): ')
col2.markdown(colored_box(f"<b>${income_low}<b>", "#ADD8E6"), unsafe_allow_html=True)

col2.write(f'##### Yearly income (high): ')
col2.markdown(colored_box(f"<b>${a}<b>", "#ADD8E6"), unsafe_allow_html=True)
col2.write(f'##### Occupancy rate: ')
col2.markdown(colored_box(f"<b>${a}<b>", "#ADD8E6"), unsafe_allow_html=True)

col2.write(f'##### Yearly income (per 100k investment): ')
col2.markdown(colored_box(f"<b>${a}<b>", "#ADD8E6"), unsafe_allow_html=True)

#col2.info("Income and occupancy rate")

#==========> column 3
#col3.write('#### Home information')
col3.write('some text')
col3.line_chart(data)

st.text('This app is developed by Masoud "Massi" Alfi to help the investors find the best investment opportunities in Austin Texas.[Inside airbnb](www.insideairbnb.com) was used for get quarterly data from airbnb. Home price data was extracted from [Zillow](www.zillow.com/data).')

st.write(f"Selected Option:    {a}" )
st.write(f"Selected Option:    {a}" )
# Set the title of the app
st.title("Colored Boxes with Text Example")


# Display colored boxes with text
st.markdown(colored_box("<b>This is a blue box<b>", "#ADD8E6"), unsafe_allow_html=True)
st.markdown(colored_box("This is a green box", "green"), unsafe_allow_html=True)
st.markdown(colored_box("This is a yellow box", "yellow"), unsafe_allow_html=True)
st.markdown(colored_box("This is a red box", "red"), unsafe_allow_html=True)






st.subheader("Market research tool")

with st.container():
   st.write("This is inside the container")

   # You can call any Streamlit command, including custom components:
   st.bar_chart(np.random.randn(50, 3))

st.write("This is outside the container")

col1, col2, col3 = st.columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))
