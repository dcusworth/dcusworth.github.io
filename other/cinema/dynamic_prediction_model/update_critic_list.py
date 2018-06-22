########################################################################
#
# DESCRIBE CRITICS in DATASET
#
# This script queries all critics that have been scraped from 
# RottenTomatoes and outputs number of reviews and correlation
# to CinemaScore
#
# dcusworth 5/9/17
########################################################################


import numpy as np
import pandas as pd
import time
import json
import unittest, re
import itertools
import string
    

#Load in ancillary functions
from class_func import *

#Load pre-scraped Rotten Tomatoes data
good_rot = rot_to_df('rotdat')

#Combine films with CinemaScores
crot = combine_data_with_cinemascore(good_rot, 'updated_rscore', 'date', True)

#Add short-form genre to feature data
crot_date, ngenre_maps = add_short_genre(crot)

#Do Groupby to get total number of reviews and correlation to cinemascore in dataset
crot_critics = crot_date.copy()
critic_sort = pd.DataFrame(crot_critics.groupby('critic').count().sort('title', ascending=False)['title'])
critic_corr = crot_critics.groupby('critic')[['full_cs', 'updated_rscore']].corr().ix[0::2,'updated_rscore']
corr_df = pd.DataFrame(np.abs(np.asarray(critic_corr)),index=critic_corr.index.levels[0], columns=['abs_cor']) 
corr_sort = corr_df.sort('abs_cor', ascending=False)

#Combine count and correlation dataframes
critic_sort['ii'] = critic_sort.index
corr_sort['ii'] = corr_sort.index
merge_count = pd.merge(critic_sort, corr_sort, on='ii')
merge_count.columns = ['num_reviews', 'name', 'abs_cor']

summary_dat = merge_count[['name', 'num_reviews', 'abs_cor']].sort('num_reviews', ascending=False)
summary_dat['name'] = [q.encode('utf-8') for q in summary_dat.name]

#Write to disk
summary_dat.to_csv('./ancillary_data/critics_in_dataset.csv')