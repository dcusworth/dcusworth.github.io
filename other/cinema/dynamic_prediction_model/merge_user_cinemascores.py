#######################################################################
#
# MERGE USER INPUTTED CINEMASCORES WITH EXISTING DATASET
#
# Script reads films_with_no_cinemascores.csv to get user inputs
# checks to see if they are all in correct formatting,
# then updates cinemascore/release date dataframes
#
#######################################################################


import pandas as pd
import numpy as np
import itertools
import json
import datetime
from class_func import *


#Open CinemaScore dataframe
cinemascore = pd.read_csv('./ancillary_data/cinemascore_data.csv', index_col=0)

#Open release dates
with open('./ancillary_data/better_release_dates.json', 'r') as fp:
    release_dict = json.load(fp)
    
good_primary_genres = ['Sci-Fi', 'Anthology', 'Sports', 'Crime',                               'Romance', 'Supernatural', 'Comedy', 'War',                               'Teen', 'Biopic', 'Horror', 'Sequel', 'Western',                               'Thriller', 'Adventure', 'Mystery', 'Foreign',                                'Drama','Historical', 'Remake', 'Action', 'Animated',                               'Documentary','Musical', 'Spy', 'Family',                                'Romantic Comedy', 'Period', 'Fantasy','Adaptation', 'Children']

good_cscores = ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"]

#Open Rotten Tomatoes training set
good_rot = rot_to_df('rotdat')
rot_titles = list(set(good_rot.title))


#Read user-inputted CinemaScores/Release Date
user_cs = pd.read_csv('./ancillary_data/films_with_no_cinemascore.csv', index_col=0)

#Check for rows that have filled in primary genre & cinemascore
updated_films = user_cs.dropna(axis=0)


#Loop over user inputted films
for irow in updated_films.iterrows():
    idict = dict(zip(irow[1].index, irow[1].tolist()))
    
    title_in = idict['title'] not in rot_titles
    proper_genre = idict['primary_genre'] not in good_primary_genres
    proper_cs = idict['cinemascore'] not in good_cscores

    try:
        idate = datetime.datetime.strptime(idict['release'], '%Y-%m-%d')
        proper_date = False
    except:
        proper_date = True
        
    log_array = [title_in, proper_genre, proper_cs, proper_date]
    name_array = ['title', 'primary_genre', 'cinemascore', 'release']
    if np.sum(log_array) > 0:
        print idict['title'], 'not added'
        print 'Check following fields:'
        print name_array[np.where(log_array)[0][0]]
        print ''
        
    else:
        
        #Add to cinemascore dataframe
        new_array = [idict['title'],idict['primary_genre'], None, idict['cinemascore'], None ]
        new_cs = dict(zip(cinemascore.columns, new_array))
        new_cinemascore = cinemascore.append(new_cs, ignore_index=True)
        
        #Add date to release dataframe
        new_dict = {}
        new_dict[idict['title']] = datetime.datetime.strftime(idate, '%b %d, %Y')
        release_dict.update(new_dict)
        
        
        #Save to disk
        new_cinemascore.to_csv('./ancillary_data/cinemascore_data.csv')

        with open('./ancillary_data/better_release_dates.json', 'w') as fp:
            json.dump(release_dict, fp)
            
            
        print idict['title'], 'added to training set'
        print ''



#Resave films_with_no_cinemascore to only include blank films
updated_film_titles = updated_films.title
remaining_films = user_cs[~user_cs.title.isin(updated_film_titles)]
remaining_films.to_csv('./ancillary_data/films_with_no_cinemascore.csv')




