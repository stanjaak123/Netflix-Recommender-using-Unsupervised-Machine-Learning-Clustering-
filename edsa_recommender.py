"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies

import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')



# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","Exploratory Data Analysis", "Team Members"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------

       
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Our recommender engine can be utilized in a variety of areas including movies, music, news, books or e-commerce products. Our data scientists help you to set up the right scenarios to fit your domain specific use cases")
        st.image("person.jpg")
        st.image("Pic1.jpg")      
    
    if page_selection == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")
        st.write("This page was designed to help data analysts to get a better understanding of the data. It focuses specifically on the movies and ratings datasets.")
        st.header("The Movies dataset")
        df = pd.read_csv('resources/data/movies.csv')
        st.write("Displaying the first few entries in the movie dataset")
        st.write(df)
        
        st.write("Displaying basic statistics about the movieId column.")
        st.write(df.describe())

        # show shape
        if st.checkbox("Check to display the shape of the Movies Dataset"):
            data_dim = st.radio("Show Dimensions By ", ("Rows", "Columns"))
            if data_dim == 'Row':
                st.text("Number of Rows")
                st.write(df.shape[0])
            elif data_dim == 'Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])
            else:
                st.write(df.shape[0])

        if st.checkbox("Check to see specific columns"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        # Show values
        if st.button("Display the amount of movies for each genre"):
            st.text("The amount of movies by genre.")
            st.write(df.iloc[:,-1].value_counts())

        

        df1 = pd.read_csv("resources/data/ratings.csv")
        st.header("The Ratings (Train) dataset")
        st.write("Displaying the first few entries in the ratings dataset")
        st.write(df1)


        if st.checkbox("Check to display the shape of the Ratings Dataset"):
            data_dim1 = st.radio("Show Dimensions by ", ("Rows", "Columns"))
            if data_dim1 == 'Rows':
                st.text("Number of Rows")
                st.write(df1.shape[0])
            elif data_dim1 == 'Columns':
                st.text("Number of Columns")
                st.write(df1.shape[1])
            else:
                st.write(df1.shape[0])

        if st.checkbox("Check to see specific columns "):
            all_columns = df1.columns.tolist()
            selected_columns = st.multiselect("Select one or more columns to display", all_columns)
            new_df = df1[selected_columns]
            st.dataframe(new_df)

        # Show values
        #if st.button("Value Counts for Ratings dataset"):
         #   st.text("Value Counts By Target/Class")
         #   st.write(df1.iloc[:,-1].value_counts())

        st.write("Displaying basic statistics of the Ratings dataset")
        st.write(df1.describe())


        st.write("Displaying the ratings distribution accross all users")
        fig, ax = plt.subplots()
        df1.hist(
        bins=8,
        column="rating",
        grid=False,
        figsize=(8, 8),
        color="#86bf91",
        zorder=2,
        rwidth=0.9,
        ax=ax,  
        )
        st.write(fig)
    
    if page_selection == "Team Members":
        st.title("TEAM Classificatioin_JS4_DSFT21")
        
        st.write("TYRONE CARELSE")
        st.image("Tyrone.jpg")

        st.write("JACQUES STANDER")
        st.image("Jacques.jpg", width=300)

        st.write("KHOMOTSO MAAKE")
        st.image("Khomotso.jpg", width=300)

        st.write("DIMAKATSO MONGWEGELWA")
        st.image("wasp.jpg", width=300)

        st.write("MELUSI ZWANE")
        st.image("batman.jpg",width=300)

        st.write("REFILOE DHLAMINI")
        st.image("ref.jpg", width=300)
     

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    


if __name__ == '__main__':
    main()


    
