import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import seaborn as sns
import flask

#Reading the Data


def get_data():
    """
    this function gets the movies and ratings tables from the specified address on the computer.
    :return: movies and ratings table as pandas dataframe. also a dataframe named final_dataset which is a matrix which its indexes are movie ids and its columns are users'
    """
    movies = pd.read_csv("E:/Projects/Protest and Unrest Forcasting/Dataset/Movie_Lens/movies.csv")
    ratings = pd.read_csv("E:/Projects/Protest and Unrest Forcasting/Dataset/Movie_Lens/ratings.csv")
    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating') # Making dataset usable for our purpose
    return movies, ratings, final_dataset


def get_user_rating(movies):
    """
    gets user rating from a csv file, then adds movie ids from the movie table and removes movies which are not in the database.
    :param movies: movies dataframe
    :return: user rating dataframe that has movieId column
    """
    user_ratings = pd.read_csv("E:/Projects/Protest and Unrest Forcasting/Dataset/Beta-Films/Beta - Film.csv", dtype={'Year':str})
    movie_id_list = []
    #print(beta_ratings)
    for i in range(0, len(user_ratings)):
        movie_user_list = user_ratings.iloc[i]['Movie_name']
        year_user_list = user_ratings.iloc[i]['Year']
        index_in_movies = movies['movieId'][movies['title'].str.match(movie_user_list)].tolist()
        titles = movies['title'][movies['title'].str.match(movie_user_list)].tolist()
        new_idx = 0
        for j in range(0, len(index_in_movies)):
            if str(year_user_list) in titles[j]:
                new_idx = index_in_movies[j]
            else:
                new_idx = 0
        movie_id_list.append(new_idx)
    user_ratings['movieId'] = movie_id_list
    user_ratings.drop(user_ratings[user_ratings['movieId'] == 0].index, inplace=True)
    return user_ratings


def add_user_ratings_column(user_ratings, final_dataset, user_name):
    """
    This function gets user ratings and user_name/ids that we want to forecast based on them. then returns the rating
    matrix which contains the user rating.
    :param user_ratings: the rating of the movies with their lens ids. (pandas dataframe)
    :param final_dataset: the rating matrix of movie lens users.
    :param user_name: the username of the person we want to forecast their ratings.
    :return: the rating matrix of movie lens users with contains a column for the "user_name" ratings.
    """
    user_rating_list = len(final_dataset)*[0.0]
    final_dataset['UserRating'] = user_rating_list
    user_ratings.fillna(0, inplace = True)
    for i in range(0, len(user_ratings)):
        movie_id_t = user_ratings.iloc[i]
        movie_id = movie_id_t['movieId']
        user_rating_t = user_ratings.iloc[i]
        user_rating = user_rating_t[user_name]/2
        final_dataset.at[movie_id, 'UserRating'] = user_rating
       # print(movie_user_rating)
    ## 1. see if the movie is present in the user list
    ## 2. if it if present and has rating, put the rating to the list
    ## 3. else, put zero/ null
    return final_dataset


def cleaning_data(dataset):
    """
    It gets the movieID*UserID matrix and remove rows and columns whose number of nulls are large
    :param dataset: tha matrix of ratings which the indexes are movie ids and columns are user ids.
    :return: the same dataset which nulls are zero and some of the rows and columns are removed.
    """
    dataset.dropna(0, thresh=11, inplace=True)
    dataset.dropna(1, thresh=51, inplace=True)
    dataset.fillna(0, inplace=True)
    return dataset


def get_similar_movies(movie_name, n_recs, n_user_clusters=10):
    """
    this function contains the core process of the movie recommendation system. it starts with get_data, then cleans it
    then filter based on genres and finally uses knearest algorithm and cosine distance to suggest similar movies.
    :param movie_name: a string that contains the movie name
    :param n_recs: an integer which are number of recommendations that we want from our sysetm
    :param n_user_clusters: number of users' cluster
    :return: a list that contains the title of suggested movies.
    """
    movies, ratings, f_dataset = get_data()
    fnl_dataset = cleaning_data(f_dataset)
    final_dataset = genre_filter(movie_name, fnl_dataset, movies)
    final_dataset = cluster_users(final_dataset, n_user_clusters)
    # making sparse matrix and checking if the movie is in the movies table or not
    csr_data = csr_matrix(final_dataset.values)  # converting the dataset to sparse matrix
    final_dataset.reset_index(inplace=True)   # reseting the indexes of final_dataset and adding a column named movieId which was the original dataset index
    # nearest
    knn = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='cosine', n_jobs=-1).fit(csr_data)
    # gets the index of the row that has the movie id
    movie_list = movies[movies['title'].str.contains(movie_name)]   # gets the list of movies that we want to suggest movies based on them
    idx = movie_list.iloc[0]['movieId']  # gets the movie ID from the movies list data frame
    idx = final_dataset[final_dataset['movieId'] == idx].index
    if len(idx)>0: # checks if there is enough data to suggest movies or not
        idx = idx[0]
        distances, indices = knn.kneighbors(csr_data[idx],n_neighbors=n_recs + 1 )  # finding 11 nearest neighbors to the
        suggestions = []
        for val in indices[0]:    # putting the movies into a list
            k = final_dataset.iloc[val]['movieId']  # getting the movie id from final dataset using the indices
            a = movies['title'][movies['movieId'] == k].tolist()  # puts the movie title into a list
            suggestions += a   # puts the movie title into a list
        return suggestions
    else:
        print("Data about the movie is not enough")


def recommend_based_on_taste():
    return 0


def forecast_rating_knn_evaluation(n_user_clusters, forecast_column='UserRating'):
    """
    this function forecasts the rating of a user (or group of users in a cluster) to a movie.
    :param n_user_clusters: a string that contains the movie name
    :param forecast_column: an integer which are number of recommendations that we want from our sysetm
    :return: does not return anything! only shows the plot and print the errors of forcasting.
    """
    movies, ratings, final_dataset = get_data()
    final_dataset = cleaning_data(final_dataset)
   # final_dataset = genre_filter(movie_name, fnl_dataset, movies)
   # final_dataset = cluster_users(final_dataset, n_user_clusters)
    user_ratings = get_user_rating(movies)
    final_dataset = add_user_ratings_column(user_ratings, final_dataset, 'ALI')
    final_dataset.replace({0:pd.NA}, inplace = True)
    final_dataset.dropna(0, subset=[forecast_column], inplace=True)
    train, test = train_test_split(final_dataset, test_size=0.2)
    #train.()
    #test.dropna(0, subset=[1], inplace=True)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    train_x = train.drop(forecast_column, axis=1)
    train_y = train[forecast_column]
    test_x = test.drop(forecast_column, axis=1)
    test_y = test[forecast_column]
    model = neighbors.KNeighborsRegressor(n_neighbors=7, weights='distance', algorithm='brute', metric='cosine')
    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)
    error1 = sqrt(mean_squared_error(test_y, y_hat))
    error2 = mean_absolute_percentage_error(test_y, y_hat)
    print(error1, error2)
    test_y = test_y.tolist()
    plt.plot(test_y)
    plt.plot(y_hat)
    plt.legend(["real values", "Forecast"])
    plt.show()

    # making sparse matrix and checking if the movie is in the movies table or not


def forecast_rating_knn(n_user_clusters, forecast_column='UserRating'):
    """
    this function forecasts the rating of a users for the movies they have not rated yet.
    :param n_user_clusters: a string that contains the movie name
    :param forecast_column: an integer which are number of recommendations that we want from our sysetm
    :return: does not return anything! only shows the plot and print the errors of forcasting.
    """
    movies, ratings, final_dataset = get_data()
    final_dataset_t = cleaning_data(final_dataset)
   # final_dataset = genre_filter(movie_name, fnl_dataset, movies)
   # final_dataset = cluster_users(final_dataset, n_user_clusters)
    user_ratings = get_user_rating(movies)
    final_dataset = add_user_ratings_column(user_ratings, final_dataset_t, 'ALI')
    final_dataset.replace({0: pd.NA}, inplace=True)
    train = final_dataset.dropna(0, subset=[forecast_column])
    final_dataset.fillna(0, inplace=True)
    test = final_dataset[final_dataset[forecast_column] == 0]
    train.fillna(0, inplace=True)
    train_x = train.drop(forecast_column, axis=1)
    train_y = train[forecast_column]
    test_x = test.drop(forecast_column, axis=1)
    test_y = test[forecast_column]
    forcasted_movie_ids = test_y.index.tolist()
    model = neighbors.KNeighborsRegressor(n_neighbors=7, weights='distance', algorithm='brute', metric='cosine')
    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)
    forecast_results = pd.DataFrame(data={'movieId': forcasted_movie_ids, 'ratings': y_hat})
    return forecast_results


def cluster_movies(movies):
    """
    this funtion clusters movies based on their genre.
    :param movies: gets the movies data frame whose columns are movieId, title, and genres.
    :return: a clustered list of movie IDs
    """
    n_clusters = 4
    # The first part of the function , makes a dataframe of the movies/genres
    all_genres = []  # the list of all genres, this will be used for making the input of the clustering algorithm
    indexes = []  # the indexes of the dataframe which are movie IDs
    genres_mat = []  # a matrix contains the genres, movie IDs and wether these movie has these genres or not. the
    for i in range(0, len(movies)):  # iterates over the movies to extract the genres and puts it into lists
        movie_id = movies.iloc[i]['movieId']  # gets the movie ID
        movie_genres = movies['genres'][movies['movieId'] == movie_id].tolist()[0].split("|")  # put the genres to a list of genres
        genres_mat.append(len(all_genres)*[0])  # Adds 0 to the matrix in front of the movie id
        indexes.append(movie_id)  # appends movie ID to a list of movie ids.
        for j in range(0, len(movie_genres)):  # adds 1 in the genre column if the movie has it and leave others 0
            if movie_genres[j] in all_genres:
                genres_mat[i][all_genres.index(movie_genres[j])] = 1
            else:  # if the genre weren't in all_genres list (columns) adds it to the list
                all_genres.append(movie_genres[j])
                genres_mat[i].append(1)
    genres_mat_df = pd.DataFrame(genres_mat, index=indexes, columns=all_genres)  # converts the list to dataframe
    genres_mat_df.fillna(value=0, inplace=True )  # Fills nulls with 0 because if it's zero the movie does not have that genre

    # The second part of the function makes clusters the movies based on genres
    kmeans_obj = KMeans(n_clusters=n_clusters, random_state=0).fit(genres_mat_df)  # defining the object that clusters the data using k-means
    kmeans_results = kmeans_obj.labels_   # brings the results of the cluster

    # note that we can give these results to a mongoDB for lower the calculation time for the next runs.
    return indexes, kmeans_results  # returns an array consists of movie IDs and their cluster


def cluster_users(dataset, n_clusters):
    """
    this clusters the users based on their rating to the movies using K-means.
    :param dataset: a dataframe that its indexes are movie ids and columns are user id
    :param n_clusters: this variable are number of clusters
    :return: user ids and their cluster id
    """
    transposed_data = dataset.transpose()  #transposes the dataframe
    #print(transposed_data)
    kmeans_obj = KMeans(n_clusters=n_clusters, random_state=0).fit(transposed_data)  # defining the object that clusters the data using k-means
    kmeans_results = kmeans_obj.labels_   # brings the results of the cluster
    transposed_data = transposed_data.replace({0:pd.NA})  # replace zeros with null for calculating the mean without 0
    transposed_data['clusterID'] = kmeans_results  # add a new column (cluster ID) to the dataframe
    cluster_rating_mean = []
    for i in range(0, n_clusters):  #gets the mean of ratings for each cluster and puts in a list
        cluster_rating_mean.append(transposed_data[transposed_data['clusterID'] == i].mean(axis=0).tolist())
    movie_ids = dataset.index.tolist()  #gets the movie ids
    dataset = pd.DataFrame(cluster_rating_mean, columns=movie_ids + ['clusterID']).transpose().rename_axis('movieId')
    # (last line): puts the list into a dataframe, transposes it and changes the index name from 'index' tp 'movieId'.
    dataset.drop(index='clusterID', inplace=True)  # drops cluster id
    dataset.fillna(0, inplace=True)  # fill null values with 0
    return dataset


def genre_filter(movie_name, dataset, movies):
    """
    this function filters the movies based on the requested movie name.

    :param movie_name: the movie that we want to recomment based on it
    :param dataset: it is the final dataset, its rows are movies and the columns are ratings
    :param movies: the movies dataset which is used for finding the movie ID of the movie that this function got.
    :return: a dataset that movies are in a particular cluster
    """
    # it gets the movie ids of the entered movie from the movie table.
    movie_id = movies['movieId'][movies['title'].str.contains(movie_name)].tolist()
    if len(movie_id) > 0:  # examines if the movie id is available or not
        movie_id = movie_id[0]
        indexes, clusters = cluster_movies(movies)  # uses cluster_movies function to cluster the movies and gives the results.
        clustered_movies_df = pd.DataFrame({'MovieID':indexes,'ClusterID': clusters})  # turn the results to a dataframe
        cluster_id = clustered_movies_df['ClusterID'][clustered_movies_df['MovieID'] == movie_id].tolist()[0]  #ges the cluster id of the entered movie
        dataset_movie_id = dataset.index.tolist()  #gets the indexes of the movies in pivot dataset. (final dataset)
        dataset_cluster_id = []  # a list which will be filled by the cluster id of the indexed movies.
        for id in dataset_movie_id:  # repeats over movie ids which are in the pivot data set (dataset object)
            cluster_ind_id = clustered_movies_df['ClusterID'][clustered_movies_df['MovieID'] == id].tolist()  #finds the cluster id of each individual movie
            dataset_cluster_id += cluster_ind_id  #puts it into the list
        dataset['ClusterID'] = dataset_cluster_id  #puts the cluster id into the dataset object
        dataset.drop(dataset[dataset.ClusterID != cluster_id].index, inplace=True)  # drops the movies which are not in the cluster
        dataset.drop(columns='ClusterID', inplace=True )  #drops the cluster id column
        return dataset
    else:
        print("the movie is not in our database.")
        quit()


#a, b, c = get_data()
#c = cleaning_data(c)
#print(len(cluster_users(c)))
#sug = get_movie_recommendation(input('please name a movie to get similar movies:'), int(input('How many recommendations do you want?')))
#print(sug)
#movies, ratings, final_dataset = get_data()
#forecast_rating_knn(20, 1)

    # next step: outputting the function as a list. (Done)
# next step: clustering the movies before suggestion based on genres. (Done)
# documentation and debugging (Done)
# next step: clustering the users based on their ratings to the movies. (Done)
# next step: using current user's rating

# next step: working on a larger dataset
# making web service
forecast_rating_knn(10)
#forecast_rating_regression(10)
#a = get_user_rating(movies)
#print(a)