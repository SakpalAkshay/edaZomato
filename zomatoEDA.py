import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.graph_objs as grp
from geopy.geocoders import Nominatim
import folium
from folium.plugins import HeatMap
from streamlit_folium import  folium_static
from wordcloud import WordCloud, STOPWORDS

#reading the data
def read_file():
    filename = 'zomato.csv'
    readFile = pd.read_csv(filename, error_bad_lines=False)
    return readFile

#Overview of data
def topData(df,n):
    return df.head(n)

def lastData(df,n):
    return df.tail(n)

#null value data
def nullFeatureList(df):
    featureNull = [feature for feature in df.columns if df[feature].isnull().sum() > 0]
    return featureNull

#ratevaluespillter
def splitRate(x):
    return x.split('/')[0]

#dealing with Rate column
def rateUpdate():
    df.replace("NEW",0, inplace=True)
    df.replace('-',0, inplace=True)
    df['rate'] = df['rate'].astype(float)

#Average restaurant Rating
def averageRestaurentRating():
    df_rate = df.groupby('name')['rate'].mean().to_frame().reset_index()
    df_rate.columns = ['Restaurant', 'Average Rating']
    return df_rate

#top restra chains
def getTopRestaurants():
    resChains = df['name'].value_counts()[0:20]
    return resChains

#types of Restaurants
def typesOfRestaurants():
    df['rest_type'].dropna(inplace=True)
    resTypes = grp.Bar( 
        x = df['rest_type'].value_counts().nlargest(20).index,
        y = df['rest_type'].value_counts().nlargest(20),
        name= 'rest_type')
    return resTypes

#highest rated restautrants with votes
def topRatedRestaurants():
    topRated =grp.Bar( 
        x = df.groupby('name')['votes'].max().nlargest(10).index,
        y = df.groupby('name')['votes'].max().nlargest(10),
        name= 'name')
    return topRated

#locations with highest Restaurants density
def locRestDensity():
    rest=[]
    loc=[]
    for key,location_df in df.groupby('location'):
        loc.append(key)
        rest.append(len(location_df['name'].unique()))
    dfTotal= pd.DataFrame(zip(loc,rest))
    dfTotal.columns=['location','restaurant']
    dfTotal.set_index('location',inplace=True)
    dfTotal.sort_values(by='restaurant').tail(10)
    resDensity = grp.Bar( 
        x = dfTotal['restaurant'].nlargest(10).index,
        y = dfTotal['restaurant'].nlargest(10),
        name= 'Priority')
    return resDensity

#cost of two people

def approxCostForTwo():
    df.dropna(axis='index',subset=['approx_cost(for two people)'],inplace=True)
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(lambda x: x.replace(',',''))
    df['approx_cost(for two people)']=df['approx_cost(for two people)'].astype(int)
    return df['approx_cost(for two people)']

#scatterplot for cost vs rating
def my_scatterplot(df, xlim1, xlim2, ylim1, ylim2):
    plt.figure(figsize=(7,7))
    sns.set_style('darkgrid')
    acc = sns.scatterplot(x = df['rate'], y = df['approx_cost(for two people)'], hue=df['book_table'])
    acc.set(xlim=(xlim1,xlim2), ylim=(ylim1,ylim2))
    st.pyplot(plt)

#finding budget restaurants
def budgetRestaurants(num):
    df_budget=data[data['approx_cost(for two people)']<=num].loc[:,('approx_cost(for two people)')]
    df_budget=df_budget.reset_index()
    return df_budget.head(10)

#finding budget restaurants
def return_budget(location="BTM",restaurant="Quick Bites"):
    st.write("Location selected {}, and restaurant type is {}.".format(location,restaurant))
    budget=df[(df['approx_cost(for two people)']<=700) & (df['location']==location) & 
                     (df['rate']>4) & (df['rest_type']==restaurant)]
    budgetRes = budget['name'].unique()
    dfBudgetRes = pd.DataFrame(budgetRes, columns = ['Restaurant Name'])
    #budgetRes.columns = "Restraunt Name"
    return dfBudgetRes

#lat-long provider

def latLongFinder(locations):
    lat_lon=[]
    geolocator=Nominatim(user_agent="app")
    for location in locations['Name']:
        location = geolocator.geocode(location)
        if location is None:
            lat_lon.append(np.nan)
        else:    
            geo=(location.latitude,location.longitude)
            lat_lon.append(geo)
    return lat_lon
#Streamlit Coding #Overview
st.title("EDA and Predictive Analysis of Restaurant Data")
df = read_file()

#Overview of data
st.header("Overview of our Restaurant Data")
st.write("The dataset has {r} rows and {c} columns".format(r=df.shape[0], c=df.shape[1]))

#top N terms of Dataframe
headCount = st.slider("Select Number of rows you want to see", 3,50)
st.write("This is top {count} rows.".format(count=headCount))
headDataFrame = topData(df,headCount)
st.dataframe(headDataFrame)

#preparing data for cleaning
st.header("Data Cleaning and Preproccessing")
st.subheader("Count of Null Values in each columns")
nullValuesDf = df.isnull().sum()
st.table(nullValuesDf)

st.subheader("Percentage of Null Values in each columns")
featureNull = nullFeatureList(df)
for feature in featureNull:
    st.write("{} has {} percent of missing values".format(feature, np.round(df[feature].isnull().sum()/len(df)*100,3 )))

#dropping null values
df.dropna(axis='index', subset=['rate'], inplace = True)
df['rate'] = df['rate'].apply(splitRate)
st.subheader("Dataset shape after data cleaning and preprocessing: {}".format(df.shape))
st.dataframe(df.head(3))

#dealing with NEW and - in rate column
rateUpdate()

#droping duplicate values
df.drop_duplicates(inplace = True)
print(df.shape)


#showing Average Restaurant Rating
st.subheader("Average Restaurant Rating")
avgRestaurantRating = averageRestaurentRating()
st.table(avgRestaurantRating.head(10))
sns.distplot(avgRestaurantRating['Average Rating'])
st.pyplot(plt)


#Top restaurant chains
st.subheader("Most Famous Restaurant Chains")
resChains = getTopRestaurants()
sns.barplot(x=resChains, y=resChains.index)
plt.title("Most Famous Restaurants By Outlets!!!")
plt.xlabel("Number of Restaurant Chains")
plt.ylabel(" ")
st.pyplot(plt)

#types of Restaurant
st.subheader("Famous Restaurants based upon food offered")
resTypes = typesOfRestaurants()
st.plotly_chart([resTypes])

#highest rated restaurants
st.subheader("Top Restaurants Based Upon Votes")
topRated = topRatedRestaurants()
st.plotly_chart([topRated])

#top locations based upon restaurant density
st.subheader("Top Locations based upon highest Restaurants Density")
resDensity = locRestDensity()
st.plotly_chart([resDensity])

#most popular cusines in banglore
st.subheader("Most Popular Cuisines")
cuisines=df['cuisines'].value_counts()[:10]
sns.barplot(x=cuisines,y=cuisines.index)
plt.xlabel('Count')
plt.title("Most popular cuisines of Bangalore")
st.pyplot(plt)


#approx cost of two people
st.subheader("Approx Two People Cost Distribution")
costForTwo = approxCostForTwo()
plt.figure(figsize=(7,7))
sns.distplot(costForTwo)
st.pyplot(plt)

#Scatter Plot for Relationship Between Cost for two people vs Rating
st.subheader("Cost of Two vs Rating Analysis")
X_values = st.slider(
    'Select a range of Rating',
    0.0, 5.0, (3.0, 5.0))
st.write('XValues:', X_values)

Y_values = st.slider(
    'Select a range of Budget',
    0, 3000, (2000, 2750))
st.write('YValues:', Y_values)
my_scatterplot(df,X_values[0],X_values[1],Y_values[0],Y_values[1])


#Finding out the most expensive as well as least expensive restaurants
data = df.copy()
data.set_index('name', inplace=True)
def mostExpensiveRestList():
    return data['approx_cost(for two people)'].nlargest(30).index.unique()
st.subheader("List of Most Expensive Restaurants")
st.table(mostExpensiveRestList())

st.subheader("List of Most Cheap Restaurants")
def cheapRestList():
    return data['approx_cost(for two people)'].nsmallest(24).index.unique()
st.table(cheapRestList())

st.subheader("Finding Best Budgets Restaurants")
num = st.number_input('Please Enter Your Budget')
num = int(num)
st.dataframe(budgetRestaurants(num))


#Finding budget restaurants in Different location based upon restaurants type
st.subheader("Finding Best budget Restaurants in any location")
locationChoice = df['location'].unique()
resChoice = df['rest_type'].unique()
location = st.selectbox("Please select one Location", locationChoice)
restaurant = st.selectbox("Please select type of Restaurant", resChoice)
st.table(return_budget(location,restaurant))


#finding geolocations 
st.subheader("Geolocations & HeatMaps of Restaurants")
locations=pd.DataFrame({"Name":df['location'].unique()})
locations['new_Name']='Bangalore '+locations['Name'] 
locations['geo_loc'] = latLongFinder(locations)
st.write(locations.head(3))

#for plotting geolocations and basemap+heatmap

def prepForMap(locations):
    restLocations=pd.DataFrame(df['location'].value_counts().reset_index())
    restLocations.columns=['Name','count']
    restLocations=restLocations.merge(locations,on='Name',how="left").dropna()
    return restLocations

restLocations = prepForMap(locations)
st.write(restLocations.head(3))

def generateMyBaseMap(default_location=[12.97, 77.59], default_zoom_start=12):
    myBasemap = folium.Map(location=default_location, zoom_start=default_zoom_start)
    return myBasemap
baseMap = generateMyBaseMap()
#st_folium(basemap, width=700)
folium_static(baseMap, width=800)

#generate heatMap
def generateMyHeatMap(baseMap, restLocations):
    #### unzip it
    lat,lon=zip(*np.array(restLocations['geo_loc']))
    restLocations['lat']=lat
    restLocations['lon']=lon

    return HeatMap(restLocations[['lat','lon','count']].values.tolist(),zoom=20,radius=15).add_to(baseMap)
heatMap = generateMyHeatMap(baseMap,restLocations)
folium_static(baseMap, width=800)

#creating word cloud for dishes based upon different rest types
st.subheader("Most Liked Dishes across different types of Restaurants")
df['update_dish_liked']=df['dish_liked'].apply(lambda x : x.split(',') if type(x)==str else [''])
rest=df['rest_type'].value_counts()[:9].index
df.dropna(axis='index',subset=['rest_type'],inplace=True)
df.dropna(axis='index',subset=['dish_liked'],inplace=True)


def produce_wordcloud(rest):
    
    plt.figure(figsize=(20,30))
    for i,restaurant in enumerate(rest):
        plt.subplot(3,3,i+1)
        dishes=''
        data=df[df['rest_type']==restaurant]
        for word in data['dish_liked']:
            words=word.split()
            # Converts each token into lowercase 
            for i in range(len(words)): 
                words[i] = words[i].lower() 
            dishes=dishes+ " ".join(words)+" "
        wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,stopwords = stopwords,width=1500, height=1500).generate(dishes)
        plt.imshow(wordcloud)
        plt.title(restaurant)
        plt.axis("off")
stopwords = set(STOPWORDS) 
produce_wordcloud(rest)
st.pyplot(plt)