# airbnb_investment

# assumptions 

- our model will focus on price range less than 2000 per night (consider your Target audience mention in your problem statement)
- Assumptions about the occupancy rates and minimum booking
- Model would not perform well in some zips without that much data

## Recommendations:
- get more accurate data on occupancy rates or find ways to better model it
- more detailed data on the actual house price for listing 
- models having a hard time predicting high occupancy rates, either dont use 0.7 cutoff or find a better way of estimating the occupancy rate. maybe we say for modeling, it is better to not have any cuttoff. 

about objectives:
XGBoost has shown the best performance in out models so far. In this section, we will see whether we can help the algorithm perform better by removing the outliers or data that might be problematic for our model. This is alongside the objectives of this study, which is to provide a reliable model to predict airbnb home price for a typical property in Austin. In our prediction, we are interested in data that are not outside the normal and popular airbnb home options (e.g. listings with number of beds in the range of 1-5 beds). Houses with outstanding features or outliers in prices or occupancy rates are not the primary subjective of this study. For this reason, removing outstanding data or outliers is in accordance with this study's objectives.  
In this section, we will first use DBSCAN to cluster the data and identify outliers. After that, we will use our best estimator to model the data to see if the model performs better on the data with less noise. In this section we will just consider the features we believe are important for our modeling (bedrooms, beds, bath, zipcode, host_is_superhost, has_pool, time_quarter, home_price_aprx) so that we limit the number of outliers due to features that are not helping with our models that much.  