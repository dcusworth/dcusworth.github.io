#############################################################
#
# ADD CINEMASCORE & RELEASE DATES to DATASET
#
# This script is intended to add individual CinemaScores
# and release dates to new films that were scraped from
# Rotten Tomatoes previously. The Rotten Tomatoes scraper
# only retrieves critic reviews for a particular film. Thus
# for each new film, this script should be run so that the
# CinemaScore and release date can be added (and the film
# can be added to the training set.
#
# This script also exports a text film that lists all
# scraped Rotten Tomatoes films that do not have a CS 
# associated. You can consult this list to add CS's to films
#
# Daniel Cusworth 5/9/17
#
#############################################################


import numpy as np
import pandas as pd
import json
import datetime
from difflib import SequenceMatcher
import sys, getopt
import argparse
from class_func import *

#Load user input
parser = argparse.ArgumentParser(description='Run dataset expander')
parser.add_argument('-i','--manual_input', help='Do you want to input new CinemaScores (0) or just output films with no CinemaScores (1) - default is 1',required=False)

args = parser.parse_args()

man_input = args.manual_input
if man_input is None:
    man_input = 1
else:
    man_input = args.manual_input


#################################

#Define similarity function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#Open CinemaScore dataframe
cinemascore = pd.read_csv('./ancillary_data/cinemascore_data.csv', index_col=0)

#Open release dates
with open('./ancillary_data/better_release_dates.json', 'r') as fp:
    release_dict = json.load(fp)

#Open critic reviews    
good_rot = rot_to_df('rotdat')
rot_titles = list(set(good_rot.title))

#Export list of films that do not have CinemaScores but have RT reviews
#Sort by number of RT reviews
no_cs = set(rot_titles) - set(cinemascore.title)
nocs_df = good_rot[good_rot.title.isin(no_cs)].copy()
count_rev = nocs_df.groupby('title').count().sort('critic', ascending=False)
count_out = pd.DataFrame(zip(count_rev.index, count_rev['critic'].tolist()), \
                         columns = ['title','num_reviews'])
count_out = count_out[count_out.num_reviews > 50].copy()
count_out['title'] = [q.encode('utf-8') for q in count_out['title']]
count_out['primary_genre'] = [None] * count_out.shape[0]
count_out['cinemascore'] = [None] * count_out.shape[0]
count_out['release'] = [None] * count_out.shape[0]

out_df = count_out[['title','primary_genre','cinemascore', 'release']]
out_df.to_csv('./ancillary_data/films_with_no_cinemascore.csv')


if man_input == 0:

    #User inputs title
    new_title = input('Enter title of new film (in quotes): ')


    ####Interact with user to verify entries are in correct format#####

    #If this film already exists with a CinemaScore, exit program
    if new_title in cinemascore.title.tolist():
        print 'This CinemaScore has already been entered, exiting'

    #If there is not CinemaScore for this film AND the film has been
    #scraped from Rotten Tomatoes, update release date/cinemascore dataframes
    elif new_title in rot_titles:
        
        #Primary Genre
        good_primary_genres = ['Sci-Fi', 'Anthology', 'Sports', 'Crime',\
                               'Romance', 'Supernatural', 'Comedy', 'War',\
                               'Teen', 'Biopic', 'Horror', 'Sequel', 'Western',\
                               'Thriller', 'Adventure', 'Mystery', 'Foreign', \
                               'Drama','Historical', 'Remake', 'Action', 'Animated',\
                               'Documentary','Musical', 'Spy', 'Family', \
                               'Romantic Comedy', 'Period', 'Fantasy','Adaptation', 'Children']
        
        prim_genre = input('Enter Primary Genre (in quotes): ')
        
        #Make sure they inputted an accepted genre
        if prim_genre not in good_primary_genres:
            print 'Incorrect genre supplied - must be one of the following:'
            print ''
            print good_primary_genres
            print ''
            isbad = True
            while isbad:
                prim_genre = input('Enter Primary Genre (in quotes): ')
                
                if prim_genre not in good_primary_genres:
                    print 'Still incorrect genre, try again'
                else:
                    isbad = False
                    print 'Acceptable genre supplied'
         
        #CinemaScore input
        good_cscores = ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"]
        cinscore = input('Enter CinemaScore A+ to F (in quotes): ')
        
        #Make sure a correct CinemaScore was inputted
        if cinscore not in good_cscores:
            print 'Incorrect CinemaScore supplied - by be one of the following:'
            print good_cscores
            isbad = True
            while isbad:
                cinscore = input('Enter CinemaScore A+ to F (in quotes): ')
                if cinscore not in good_cscores:
                    print 'Still incorrect CinemaScore, try again'
                else:
                    isbad = False
                    print 'Acceptable CinemaScore entered'
        
        #Release date
        reldate = input('Enter release date %b %d, %Y (in quotes) - e.g. "Jan 15, 2011": ')
        isbad = True
        
        #Make sure the release date is parseable
        while isbad:
            try:
                idate = datetime.datetime.strptime(reldate, '%b %d, %Y')
                isbad = False
            except:
                print 'Incorrect date specification, try again'
                reldate = input('Enter release date %b %d, %Y (in quotes) - e.g. "Jan 15, 2011": ')
                
        
        #Add inputs to existing datasets
        new_dict = {}
        new_dict[new_title] = reldate
        release_dict.update(new_dict)
        
        new_cs = dict(zip(cinemascore.columns,[new_title, prim_genre, prim_genre, cinscore, None]))
        new_cinemascore = cinemascore.append(new_cs, ignore_index=True)
        
        #Save to disk
        new_cinemascore.to_csv('./ancillary_data/cinemascore_data.csv')
        
        with open('./ancillary_data/better_release_dates.json', 'w') as fp:
            json.dump(release_dict, fp)
            
        print 'New CinemaScore & release date updated succesfully'
        

    #If the film they entered is not in Rotten Tomatoes scrape, then exit
    #Output potential matches that the user could retry (in case spelling was incorrect)
    #Otherwise, need to rescrape Rotten Tomatoes
    else:
        print 'Film not found in Rotten Tomatoes dataset'
        sim_met = [similar(q, new_title) for q in rot_titles]
        ipdf = pd.DataFrame(zip(sim_met, rot_titles), columns=['sim','title'])
        ipdf_sort = ipdf.sort('sim', ascending=False)
        print ''
        print 'Closest 5 matches:'
        print ipdf_sort.title.tolist()[0:5]
        print ''
        print 'Try again with one of these titles, or rescrape Rotten Tomatoes reviews'
        print 'Exiting...'





