import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import json
import random
random.seed(200)
from datetime import date, timedelta, time
import unittest, time, re
import itertools
from scipy.stats.stats import pearsonr
import collections
import string
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
import operator as op
import sys
import pickle


##############################################################
#
# CinemaScore Prediction Functions
#
# This script loads ancillary functions needed to run
# CinemaScore prediction.
#
#
# Created by Daniel Cusworth - 2/21/2017
#
##############################################################


#RMSE function
rms_func = lambda y_hat, y: np.sqrt(mean_squared_error(y, y_hat))


#Combine dataframe with cinemascore
def combine_data_with_cinemascore(choose_dat, score_name='mscore', date_name='date', dostrip=False):
    
    #Load cinemascore data
    merge_cinemascore_title = pd.read_csv('./ancillary_data/cinemascore_data.csv', index_col=0)

    #Strip titles of punctuation to make lists more comparable
    if dostrip:
        cinema_titles = merge_cinemascore_title.title
        rot_titles = choose_dat.title
        rot_strip = [q.replace(':','').replace('-','').replace(' ','').replace('.','').replace("'","").lower() for q in rot_titles]
        cin_strip = [q.replace(':','').replace('-','').replace(' ','').                     replace('.','').replace("'","").lower().replace(',the','') for q in cinema_titles]
        rot_strip2 = [q.replace('the','') if q[0:3]=='the' else q for q in rot_strip ]

        merge_cinemascore_title['strip_title'] = cin_strip
        choose_dat['strip_title'] = rot_strip2


    else:
        merge_cinemascore_title['strip_title'] = merge_cinemascore_title.title
        choose_dat['strip_title'] = choose_dat.title
    
    merge_cinemascore_title = merge_cinemascore_title[merge_cinemascore_title.                                                      columns[~merge_cinemascore_title.columns.isin(['title'])]]
    

    merge_cinema_meta = pd.merge(merge_cinemascore_title, choose_dat, on='strip_title')
        
    ldat = merge_cinema_meta[merge_cinema_meta.cinemascore.notnull()                 & merge_cinema_meta.title.notnull()                 & merge_cinema_meta.primary_genre.notnull()].copy()

    if date_name:
        sel_cols = ['primary_genre', 'cinemascore', 'title', 'critic', score_name, date_name]
    else:
        sel_cols = ['primary_genre', 'cinemascore', 'title', 'critic', score_name]

    return ldat[sel_cols]

#Choose dataset and create usable dataframe
def choose_dataset(which_data):

    #Choose Rotten Tomatoes scrape as it gives us the most reviews with cinemascores
    ddat = which_data.copy()

    #Make cinemascore map - One for all classes, one for collapsed classes
    cinema_map = dict(zip(["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"],range(13)))
    cinema_map_three = dict(zip(["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"],                               [0,0,0,1,1,1,2,2,2,2,2,2,2]))
    cinema_rev = dict(zip([0,1,2], ["A","B","C-F"]))

    #Make genre map
    all_genres = list(set(ddat.primary_genre))
    genre_map = dict(zip(all_genres, range(len(all_genres))))
    genre_list = [genre_map[q] for q in ddat.primary_genre]
    ddat['ngenre'] = genre_list

    ddat['full_cinemascore'] = [float(cinema_map[q]) for q in ddat.cinemascore]
    ddat['collapsed_cinemascore'] = [cinema_map_three[q] for q in ddat.cinemascore]

    #Make datetime object
    date_stripped = [q.replace('Posted ','') for q in ddat.date]
    date_converted = [datetime.datetime.strptime(q, '%b %d, %Y') for q in date_stripped]
    ddat['better_date'] = date_converted
    ddat['yy'] = [q.year for q in date_converted]
    ddat['mm'] = [q.month for q in date_converted]
    ddat['dd'] = [q.day for q in date_converted]

    #Get mean by year and month across all reviews
    dmean = ddat.groupby(['yy','mm']).mean()
    dstd = ddat.groupby(['yy','mm']).std()
    dcount = ddat.groupby(['yy','mm']).count()

    #Summary of score means - combine
    dsummary = pd.concat([dmean['updated_rscore'], dstd['updated_rscore'], dcount['updated_rscore'],                         dmean['full_cinemascore'], dstd['full_cinemascore'], dcount['full_cinemascore']], axis=1)
    dsummary.columns = ['smean','sstd','scount', 'cmean', 'cstd', 'ccount']
    dsummary['yy'] = [q[0] for q in dsummary.index]
    dsummary['mm'] = [q[1] for q in dsummary.index]
    dsummary = dsummary[1:].copy()

    #Add datetime object for later plotting
    timeobj = [datetime.datetime.strptime(str(q[0]) + '-' + str(q[1]) + '-15', '%Y-%m-%d')                for q in zip(dsummary.yy, dsummary.mm)]

    dsummary['date_time'] = timeobj

    return ddat, dsummary


#Make feature matrix
def make_feature_matrix(which_process, which_data, r=50, d=20, cthres=500, fthres = 10, n=500):
    
    ddat, dsummary = choose_dataset(which_data = which_data)
    
    ####Variables for each process type
    #Process 0: No pre-selection - create pivot table of full reviewer/film set
    #Process 1: cthres = Number of reviews we require a reviewer to have reviewed,
    #fthres = Number of reviewers that must exist per film
    #Process 2&3: d = number of reviewers (features), n = number of observations to use
    #Process 3: r = number of initial reviewers

    fmodel_mat = ddat.pivot_table(index='title', columns='critic', values='updated_rscore', aggfunc='mean')
    
    ######### PROCESS 0 ##############
    if which_process == 0:        
        
        return fmodel_mat

    
    ######### PROCESS 1 ##############
    if which_process == 1:
        #Want reviewers as features
        #Look at distribution of reviews per reviewer
        rgroup = ddat.groupby('critic').count()

        #Select critics that have above a threshold of reviews
        #This will hopefully make imputation more sound
        rcritics = rgroup[rgroup.title > cthres].copy()
        tdat = ddat[(ddat.critic.isin(rcritics.index)) & (ddat.critic != 'undefined')].copy()

        #Group by title to see how many unique critic reviews exist per title
        #Choose a threshold, so that we guarantee a minimal number of non-imputed reviews per film
        tgroup = tdat.groupby('title').count()
        gfilms = tgroup[tgroup.critic > fthres].copy()
        ndat = tdat[tdat.title.isin(gfilms.index)].copy()

        #Make predictor columns and fill with reviews
        pred_cols = list(set(ndat.critic))

        print 'Number of Reviewers as Features (d):', len(pred_cols)
        print 'Number of Films (n)', len(list(set(ndat.title.tolist())))
        
        return fmodel_mat[pred_cols]
        

    ######### PROCESS 2&3 ##############
    if which_process == 2 | which_process == 3:
        #Get critic with most reviews
        review_counts = ddat.groupby('critic').count().sort('title', ascending=False)
        top_reviewer = review_counts.index[0]

        print 'Top reviewer by count:', top_reviewer

        #Want to minimize number of missing films between reviewers 
        #Find the disjoint of reviews between reviewer_i and top reviewer
        #Select top d reviewers - these will be the features
        top_reviewers_films = ddat[ddat.critic==top_reviewer]['title'].tolist()
        critic_list = list(set(ddat[ddat.critic != top_reviewer]['critic'].tolist()))

        common_films = []
        for icritic in critic_list:
            icritic_films = ddat[ddat.critic==icritic]['title'].tolist()
            intersection_of_films = set(icritic_films) & set(top_reviewers_films)
            common_films.append(len(intersection_of_films))

        common_film_df = pd.DataFrame(zip(critic_list, common_films), columns=['critic','ncommon_films'])
        common_sorted = common_film_df.sort('ncommon_films', ascending=False).copy()
        

    ######### PROCESS 2 ##############
    if which_process == 2:
        #Choose number of features/critics we consider
        pred_cols = [top_reviewer] + common_sorted['critic'][0:(d-1)].tolist()

        #Make model matrix
        model_mat = fmodel_mat[pred_cols]

        #With this common matrix, get films that need least imputation
        number_reviews_film = model_mat.isnull().sum(axis=1)
        sorted_number_reviews = number_reviews_film.sort(inplace=False)

        #Choose number of observations you want - this will determine extent of imputation
        data_names = sorted_number_reviews.index[0:n]

        #Update model matrix
        mdat = model_mat[model_mat.index.isin(data_names)]

        #Imputation summary statistics
        num_row_nans = mdat.count(axis=1)
        print 'Number of values to impute:', (d*n) - np.sum(num_row_nans)
        print 'Number of values in model matrix:', d*n
        print 'Percent of values needed to impute:', round(100*((d*n - np.sum(num_row_nans))/float(d*n)),1), '%'
        
        return mdat
    
        
    ######### PROCESS 3 ##############        
    if which_process == 3:
        #Find correlation with each reviewer to cinemascore. Thus need to map metascores to cinemascores
        grades = ['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F']
        grade_scores = list(reversed(np.linspace(0,10, len(grades))))
        letter_map2 = dict(zip(grades,np.round(grade_scores,0)))
        critic_corr = {}

        critic_reviews = ddat.groupby(['critic','title']).first()
        critics = [q[0] for q in critic_reviews.index.tolist()]
        critic_reviews['critic'] = critics

        for icritic in list(set(critics)):

            cscore = [letter_map2[q] for q in critic_reviews[critic_reviews.critic == icritic]['cinemascore']]
            rscore = [np.round(q*10,0) for q in critic_reviews[critic_reviews.critic == icritic]['updated_rscore']]

            critic_corr[icritic] = pearsonr(cscore, rscore)[0]

        cor_df = pd.DataFrame(critic_corr.items(), columns=['critic', 'corr'])
        
        #From r features, combine with correlation to cinemascore
        common_corr = pd.merge(common_sorted.head(r), cor_df, on='critic')
        common_corr_sorted = common_corr.sort('corr', ascending=False).copy()

        #From these top r, pick the d best correlated
        pred_cols = [top_reviewer] + common_corr_sorted['critic'][0:(d-1)].tolist()

        #Make model matrix
        model_mat = fmodel_mat[pred_cols]

        #With this common matrix, get films that need least imputation
        number_reviews_film = model_mat.isnull().sum(axis=1)
        sorted_number_reviews = number_reviews_film.sort(inplace=False)

        #Choose number of observations you want - this will determine extent of imputation
        data_names = sorted_number_reviews.index[0:n]

        #Update model matrix
        mdat = model_mat[model_mat.index.isin(data_names)]

        #Imputation summary statistics
        num_row_nans = mdat.count(axis=1)
        print 'Number of values to impute:', (d*n) - np.sum(num_row_nans)
        print 'Number of values in model matrix:', d*n
        print 'Percent of values needed to impute:', round(100*((d*n - np.sum(num_row_nans))/float(d*n)),1), '%'
        
        return mdat

#Imputation scheme
def impute_data(mdat, which_imputation):

    from sklearn.preprocessing import Imputer

    ###### IMPUTATION SCHEME 0 ########
    if which_imputation == 0:
        return mdat

    ###### IMPUTATION SCHEME 1 ########
    if which_imputation == 1:
        imputation_scheme = Imputer(axis=1)
        mdat_np = imputation_scheme.fit_transform(mdat)
        mdati = pd.DataFrame(mdat_np, index=mdat.index, columns = mdat.columns)
        return mdati
    
    ###### IMPUTATION SCHEME 2 ########    
    if which_imputation == 2:
        imputation_scheme = Imputer(axis=0)
        mdat_np = imputation_scheme.fit_transform(mdat)
        mdati = pd.DataFrame(mdat_np, index=mdat.index, columns = mdat.columns)
        return mdati
    
    ###### IMPUTATION SCHEME 3 ########    
    if which_imputation == 3:
        
        cinema_titles =  mdat.index.tolist()
        pred_cols = mdat.columns
        movie_rows = {}
        
        for imovie, ireviews in mdat.iterrows():
            jcol = []
            
            #See if there is a critic review per critic
            for jcritic, jreview in zip(mdat.columns, ireviews):
                if not np.isnan(jreview):
                    jcol.append(jreview)
                
                else:
                    
                    #Make correlation function that changes with jcritic
                    def corr_func(x):
                        idf = pd.DataFrame(zip(mdat[jcritic], x, mdat.index))
                        share_this_movie = not np.isnan(idf[idf[2] == imovie][1].tolist()[0])
                        cordf = idf.dropna()

                        if (cordf.shape[0] > 5) and (share_this_movie):
                            return pearsonr(cordf[0], cordf[1])[0]
                        else:
                            return 0

                    cor_vals = mdat.apply(corr_func)
                    cor_vals = cor_vals[cor_vals.index != jcritic]
                    
                    #If no correlation, take mean of row (film)
                    if np.max(cor_vals) == 0:
                        jcol.append(np.mean(ireviews))
                        
                    #Otherwise find critic who correlates best
                    #Get their deviation of this film vs. all reviewed films
                    #Apply to missing reviewer
                    else:
                        best_cor = cor_vals[cor_vals == np.max(cor_vals)].index[0]
                        mean_best_cor = np.mean(mdat[best_cor])
                        best_dev = mdat[best_cor][mdat.index == imovie] - mean_best_cor
                        jcritic_mean = np.mean(mdat[jcritic])
                        jcritic_imputed = best_dev + jcritic_mean
                        
                        if jcritic_imputed > 1:
                            jcritic_imputed = 1
                        if jcritic_imputed < 0:
                            jcritic_imputed = 0
                            
                        jcol.append(jcritic_imputed)
            movie_rows[imovie] = jcol
            
            
    ###### IMPUTATION SCHEME 4 ########
    if which_imputation == 4:
        return mdat.fillna(-9999)
    
#Making training/test matrix
def make_train_test(cdat, res_col='collapsed_cinemascore', pred_cols=['A.A Dowd']):

    #Split into testing and training
    from sklearn.cross_validation import train_test_split
    itrain, itest = train_test_split(xrange(cdat.shape[0]), train_size=0.8, random_state=0)
    train_data = cdat.iloc[itrain]
    test_data = cdat.iloc[itest]
    Xtrain = train_data[pred_cols]
    Ytrain = train_data[res_col]
    Xtest = test_data[pred_cols]
    Ytest = test_data[res_col]
    
    return Xtrain, Ytrain, Xtest, Ytest


#Get Wikipedia features from previously scraped/processed Wiki data
def get_wiki3():
    with open('wiki_features_blocking.json', 'r') as fp:
        wiki_dat = json.load(fp)

    wiki_df_index = wiki_dat.keys()
    wdat = wiki_dat.values()
    wmon = [q[0] for q in wdat]
    wthurs = [q[1] for q in wdat]
    wiki_df = pd.DataFrame(zip(wmon, wthurs), index=wiki_df_index, columns=['mon_wiki', 'thurs_wiki'])

    return wiki_df


#Load release dates of
def get_release_df():
    
    with open('./Wikipedia/better_release_dates.json', 'r') as fp:
        release_dict = json.load(fp)
    release_df = pd.DataFrame(release_dict.values(), index = release_dict.keys(), columns=['date'])
    idates = []
    for ifilm in release_df.date.tolist():
        try:
            idate = datetime.datetime.strptime(ifilm, '%b %d, %Y')
        except:
            try:
                idate = datetime.datetime.strptime(ifilm, '%B %d, %Y')
            except:
                idate = datetime.datetime.strptime(ifilm, '%d %b, %Y')
        idates.append(idate)
    imonths = [q.month for q in idates]
    release_df['month'] = imonths

    return release_df

#Add short genre to feature matrix
def add_short_genre(crot):
    
    #Collapse genres into shorter clustered genre categories
    #Done arbitralily, can change sub-categories
    action_genre = ['Action', 'Adventure', 'Sci-Fi', 'Fantasy', 'Spy']
    family_genre = ['Animated', 'Family', 'Children', 'Teen', 'Sports']
    pulp_genre = ['War', 'Thriller', 'Crime', 'Mystery', 'Western']
    prestige_genre = ['Drama', 'Foreign', 'Biopic', 'Historical', 'Period', 'Romance']
    comedy_genre = ['Comedy', 'Romantic Comedy', 'Musical']
    horror_genre = ['Horror', 'Supernatural']
    sequel_genre = ['Sequel', 'Remake', 'Adaptation']
    other_genre = ['Documentary', 'Anthology']
    
    short_genre = []
    nshort_genre = []
    for igenre in crot.primary_genre:
        if igenre in action_genre:
            short_genre.append('action')
            nshort_genre.append(0)
        elif igenre in family_genre:
            short_genre.append('family')
            nshort_genre.append(1)
        elif igenre in pulp_genre:
            short_genre.append('pulp')
            nshort_genre.append(2)
        elif igenre in prestige_genre:
            short_genre.append('prestige')
            nshort_genre.append(3)
        elif igenre in comedy_genre:
            short_genre.append('comedy')
            nshort_genre.append(4)
        elif igenre in horror_genre:
            short_genre.append('horror')
            nshort_genre.append(5)
        elif igenre in sequel_genre:
            short_genre.append('sequel')
            nshort_genre.append(6)
        elif igenre in other_genre:
            short_genre.append('other')
            nshort_genre.append(7)

    crot['short_genre'] = short_genre
    crot['nshort_genre'] = nshort_genre
    
    ngenre_maps = dict(zip(['action','family','pulp','prestige','comedy','horror','sequel','other'], range(8)))
    
    #Add release dates to training set films
    #These release dates were scraped offline and stored in a JSON
    #If additional films are scraped from RottenTomatoes, they will need to have
    #a corresponding update to the release date JSON
    with open('./ancillary_data/better_release_dates.json', 'r') as fp:
        release_dict = json.load(fp)
    
    release_df = pd.DataFrame(release_dict.items(), columns=['title', 'idate'])
    release_date = []
    for ikey in release_dict.keys():
        try:
            release_date.append(datetime.datetime.strptime(release_dict[ikey], '%b %d, %Y'))
        except:
            try:
                release_date.append(datetime.datetime.strptime(release_dict[ikey], '%B %d, %Y'))
            except:
                release_date.append(datetime.datetime.strptime(release_dict[ikey], '%d %b, %Y'))

    release_df['release'] = release_date
    release_df['month'] = [q.month for q in release_date]
    crot_date_all = pd.merge(crot, release_df, on='title')
    cinema_map = dict(zip(["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"],range(13)))
    crot_date_all['full_cs'] = [cinema_map[q] for q in crot_date_all.cinemascore]
    
    
    #Exclude Legendary movies from dataset - save out
    legendary_films = ['Warcraft', 'Krampus', 'Crimson Peak', 'Steve Jobs', 'Straight Outta Compton', 'Jurassic World', 'Blackhat', 'Unbroken', 'Seventh Son', 'Interstellar', 'Dracula Untold', 'As Above, So Below', 'Godzilla', '300: Rise of an Empire', 'Pacific Rim', 'Man of Steel', 'The Hangover Part III', '42', 'Jack the Giant Slayer', 'The Dark Knight Rises', 'Wrath of the Titans', 'The Hangover Part II', 'Sucker Punch', 'Due Date', 'The Town', 'Inception', 'Jonah Hex', 'Clash of the Titans', 'Ninja Assassin', 'Where the Wild Things Are', 'Trick r Treat', 'The Hangover', 'Observe and Report', 'Watchmen', 'The Dark Knight', '10,000 BC', '300', 'We Are Marshall', 'Beerfest', 'The Ant Bully', 'Lady in the Water', 'Superman Returns', 'Batman Begins']
    
    #If you want to remove legendary films from training, uncomment below
    
    #crot_date = crot_date_all[~crot_date_all.title.isin(legendary_films)].copy()
    crot_date = crot_date_all.copy()
    
    return crot_date, ngenre_maps


def create_model_matrix_wiki_block(which_data, choose_process=0, choose_imputation=0, choose_col='full_cinemascore'):

    #Choose imputation and processing scheme
    choose_model = make_feature_matrix(which_process = choose_process, which_data = which_data)
    impute_model = impute_data(choose_model, which_imputation = choose_imputation)

    #Combine model matrix with cinemascore data
    ddat, dsummary = choose_dataset(which_data = which_data)
    cinema_group = ddat.groupby('title').first()
    feat_mat = cinema_group[['collapsed_cinemascore', 'full_cinemascore', 'ngenre', 'nshort_genre','month']].copy()
    total_mat0 = impute_model.join(feat_mat, how='inner') #Join imputation to CS
    total_mat = total_mat0.copy()

    #Get the predictior/feature/critics
    pred_cols = list(set(total_mat.columns) - set(['collapsed_cinemascore', 'full_cinemascore']))
    
    #Make training and testing data
    Xtrain, Ytrain, Xtest, Ytest = make_train_test(total_mat, res_col = choose_col, pred_cols=pred_cols)

    return Xtrain, Ytrain, Xtest, Ytest


#Weighting functions to average output of prediction
def weighted_av(preds, verr):
    N = len(preds.values())
    
    if N != 0:
        #Inverse distance weights
        inv_dist = [(1/q)**2 for q in verr.values()]
        weights = [q/np.sum(inv_dist) for q in inv_dist]
        
        #Weighted average
        wav = np.sum([q*r for q,r in zip(weights, preds.values())])
        
        return wav
    else:
        return None

def min_val(preds, verr):
    if len(preds.values()) !=0:
        mind = np.where(verr.values()==np.min(verr.values()))[0][0]
        return preds.values()[mind]
    
    else:
        return None

def perc_pred(preds, verr):
    if len(preds.values()) != 0:
        perc10 = np.percentile(verr.values(), .3)
        mind = [p for p,q in zip(preds.values(), verr.values()) if q <= perc10]
        if len(mind) != 0:
            return np.mean(mind)
        else:
            return None
    else:
        return None

def av_pred(preds, verr):
    if len(preds.values()) != 0:
        return np.mean(preds.values())
    else:
        return None


#Combinatorial calculation
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

#Run XGBOOST
def run_dynamic_prediction_xg(pred_df, crot_date, train_df, Ytrain, meta_cols,  critic_name_list, max_iter=10, mod_thres=100):
    
    start = time.time()
    

    #Make cost function to determine which critics will be used in what order
    which_critics = critic_name_list
  
    
    #Get number of reviews by critic in training set
    #Get correlation to CinemaScore on training set
    crot_critics = crot_date[crot_date.critic.isin(which_critics)].copy()
    critic_sort = pd.DataFrame(crot_critics.groupby('critic').count().sort('title', ascending=False)['title'])
    critic_corr = crot_critics.groupby('critic')[['full_cs', 'updated_rscore']].corr().ix[0::2,'updated_rscore']
    corr_df = pd.DataFrame(np.abs(np.asarray(critic_corr)),index=critic_corr.index.levels[0], columns=['abs_cor']) 
    corr_sort = corr_df.sort('abs_cor', ascending=False)


    #Make cost function
    norm_func = lambda x: (x - np.mean(x)) / np.std(x)
    critic_sort['ii'] = critic_sort.index
    corr_sort['ii'] = corr_sort.index
    merge_count = pd.merge(critic_sort, corr_sort, on='ii')
    merge_count['std_count'] = norm_func(merge_count['title'])
    merge_count['std_corr'] = norm_func(merge_count['abs_cor'])

    #Get critic list sorted by score
    merge_count['score'] = merge_count['std_count'] + 2*merge_count['std_corr']
    merge_count_above = merge_count[merge_count.title > 50].copy()
    which_critics = merge_count_above.sort('score', ascending=False).ii.tolist()[0:int(max_iter)]
    
    
    #Initialize lists/variables
    subpred = {}
    valerror = {}
    subpred_all  = {}
    valerror_all = {}
    actual_models = 0
    potential_models = 0
    
    #Intialize training data/model
    param = {'max_depth':5, 'eta':.3, 'objective':'reg:linear', 'eval_metric':'rmse' }
    Dtrain = train_df.copy()
    Dtrain['resp'] = Ytrain
    h_i = XGBRegressor()
    
    #Do Genre-only Prediction
    dtrain_i = Dtrain[meta_cols + ['resp']].dropna()
    h_i.fit(dtrain_i[meta_cols], dtrain_i['resp'])
    g_pred = h_i.predict(pred_df[meta_cols])
    genre_error = rms_func(h_i.predict(dtrain_i[meta_cols]), dtrain_i['resp'].tolist())
    
    good_critics = which_critics
    jdx = 1
    
    #Sample from training set to see how many concurrent reviews exist for this set of critics
    while jdx <= len(good_critics):
        
        jcomb = itertools.combinations(good_critics, jdx)
        igood_critics = []
        
        if len(good_critics) != 0:
            potential_models += ncr(len(good_critics), jdx)
    
        for kcomb in jcomb:
            dtrain_i = Dtrain[meta_cols + list(kcomb) + ['resp']].dropna()
            xdtrain_i = xgb.DMatrix(dtrain_i[meta_cols + list(kcomb)], label=dtrain_i['resp'])
            
            #Make sure there is sufficient data to train
            if dtrain_i.shape[0] > mod_thres:
                
                if dtrain_i.shape[0] > mod_thres+10:
                    igood_critics = list(set(igood_critics + list(kcomb)))
            
                #Cross validate XGBOOST to find what round to stop training
                h_mod = xgb.cv(param, xdtrain_i, 100, metrics='rmse', early_stopping_rounds=1)
                
                #Retrieve round to stop training - get error on validation
                min_val_error = np.min(h_mod['test-rmse-mean'])
                min_val_round = np.where(h_mod['test-rmse-mean'] == np.min(h_mod['test-rmse-mean']))[0][0]
                
                #Fit XGBoost model
                h_i.fit(dtrain_i[meta_cols + list(kcomb)], dtrain_i['resp'])
                test_pred = h_i.predict(pred_df[meta_cols + list(kcomb)])
                
                #Save results if they are better than baseline genre prediction
                if min_val_error < genre_error:
                    subpred['+'.join(list(kcomb))] = float(test_pred)
                    valerror['+'.join(list(kcomb))] = min_val_error
                    subpred_all['+'.join(list(kcomb))] = float(test_pred)
                    valerror_all['+'.join(list(kcomb))] = min_val_error
                    actual_models += 1
                else:
                    subpred_all['+'.join(list(kcomb))] = float(test_pred)
                    valerror_all['+'.join(list(kcomb))] = min_val_error


        good_critics = igood_critics
        jdx += 1
    
    
    elapsed = time.time() - start

    
    return subpred, valerror, subpred_all, valerror_all, g_pred[0], actual_models, potential_models, elapsed, which_critics





##############################################################
#
# ROTTEN TOMATOES CLEANING FUNCTIONS
#
# This script loads all the data from a rottentomatoes scrape
# and cleans the review output into a uniform format
# - e.g. reviews are converted to 0-1 scale
#
#
#
##############################################################


#String to evaluate scores that are in an evaluable format
#e.g, '3 out of 4' is 0.75
def splitter(split_string, input_list):
    f_rscores2 = []
    for irscore in input_list:
        try:
            sscore = irscore.split(split_string)
            if len(sscore) == 2:
                ifscore = float(sscore[0]) / float(sscore[1])
                f_rscores2.append(ifscore)
            else:
                f_rscores2.append(irscore)
        except:
            f_rscores2.append(irscore)
    return f_rscores2

#Function to map a list of irregular scores to regular scores
#need to supply a mapper
def score_mapper(map_dict, input_list):
    f_rscores2 = []
    for irscore in input_list:
        if irscore in map_dict.keys():
            f_rscores2.append(map_dict[irscore])
        else:
            f_rscores2.append(irscore)
    return f_rscores2


def rot_to_df(rot_url):
    #Load Rottentomatoes functions
    #Each dictionary is saved in a dictionary in 'data' folder under 'critic_XXX.json'
    critic_series = []
    date_series = []
    movie_names = []
    score_series = []
    tmeter = []
    
    #Open each JSON and retrieve reviewer and reviews (and date, publication)
    for iletter in list(string.ascii_lowercase):
        try:
            critic_data_name = rot_url + '/rcritic_reviews_' + iletter + '.json'
            with open(critic_data_name, 'r') as fp:
                allcritic_dict = json.load(fp)
            for icritic in allcritic_dict.keys():
                icritic_dict = allcritic_dict[icritic]
                for imovie in icritic_dict.keys():
                    if icritic_dict[imovie]:
                        ifilm = icritic_dict[imovie]
                        tmeter.append(ifilm['tmeter'])
                        score_series.append(ifilm['rscore'])
                        movie_names.append(imovie)
                        date_series.append(ifilm['posted'].strip())
                        critic_series.append(icritic)
        except:
            ""

    #Make pd dataframe of reviews
    rot_df = pd.DataFrame(zip(critic_series, score_series, date_series, movie_names, tmeter), \
                      columns=['critic','rscore','date','title','tmeter'])
    
    #Evaluate scores, e.g. if string is '3/4', the rscore should be .75
    f_rscores = []
    import __future__
    for irscore in rot_df.rscore.tolist():
        try:
            if irscore[0] == '-':
                irscore = irscore[1:]
            irscore = irscore.replace('-','/')
            float_score = eval(compile(irscore, '<string>', 'eval', __future__.division.compiler_flag))
            f_rscores.append(float_score)
        except:
            f_rscores.append(irscore)

    #Do further evaluations using splitter function
    f_rscores2 = splitter(' out of ', f_rscores)
    f_rscores3 = splitter(' of ', f_rscores2)
    
    #Map letter to scores
    #Here 0 = F and 1 = A+
    grades = ['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F']
    grade_scores = reversed(np.linspace(0,1, len(grades)))
    letter_map = dict(zip(grades, grade_scores))
    f_rscores4 = score_mapper(letter_map, f_rscores3)
    
    #Some grades have a trailing space
    grades = [q+ ' ' for q in grades]
    grade_scores = reversed(np.linspace(0,1, len(grades)))
    letter_map = dict(zip(grades, grade_scores))
    f_rscores4a = score_mapper(letter_map, f_rscores4)
    
    #Now lowercase grades
    grades = ['a+','a','a-','b+','b','b-','c+','c','c-','d+','d','d-','f']
    grade_scores = reversed(np.linspace(0,1, len(grades)))
    letter_map = dict(zip(grades, grade_scores))
    f_rscores5 = score_mapper(letter_map, f_rscores4a)
    
    #Map percentages
    perc_str = [str(q) + '%' for q in range(0,110,10)]
    perc_map = dict(zip(perc_str, np.arange(0,1.1,.1)))
    f_rscores6 = score_mapper(letter_map, f_rscores5)
    
    #Map fresh/rotten to 1,0
    #fresh_map = dict(zip(['fresh', 'rotten', 'FRESH', 'ROTTEN', 'Fresh', 'Rotten'], [1,0,1,0,1,0]))
    #f_rscores5 = score_mapper(fresh_map, f_rscores4)
    
    #Add to original dataframe
    rot_df['updated_rscore'] = f_rscores6
    
    #Select only datapoints that are floats
    is_float = [(type(q) is float) | (type(q) is np.float64) for q in rot_df.updated_rscore]
    float_rot = rot_df[is_float].copy()
    
    #Select only reasonable scores, i.e. can't be larger than 1 or less than 0
    is_reasonable = [(q >= 0) & (q <= 1) for q in float_rot.updated_rscore]
    good_rot = float_rot[is_reasonable].copy()
    
    good_rot['updated_rscore'] = [float(q) for  q in good_rot.updated_rscore]
    
    return good_rot

###################################################################
#
# WRAPPER FOR RUNNING MODEL IN IPYTHON
#
#
###################################################################


def ipython_run_model(max_critics=15, model_thres=50, which_impute=0, which_process=0):

    #Load pre-scraped Rotten Tomatoes data
    good_rot = rot_to_df('rotdat')

    #Combine films with CinemaScores
    crot = combine_data_with_cinemascore(good_rot, 'updated_rscore', 'date', True)

    #Add short-form genre to feature data
    crot_date, ngenre_maps = add_short_genre(crot)

    #Make genre map
    all_genres = list(set(crot_date.primary_genre))
    genre_map = dict(zip(all_genres, range(len(all_genres))))

    #Make train/test split
    Xtrain, Ytrain, Xtest, Ytest = create_model_matrix_wiki_block(crot_date, which_process, which_impute, choose_col='full_cinemascore')


    ########################################
    # Load User-Provided Reviews and Metadata

    #Load metadata
    meta_csv = pd.read_csv('./user_input/metadata_input.csv')
    meta_cols = meta_csv.columns.tolist()
    meta_dat = [genre_map[meta_csv['ngenre'].tolist()[0]], \
                ngenre_maps[meta_csv['nshort_genre'].tolist()[0]], \
                meta_csv['month'].tolist()[0]]

    #Get critic data
    critic_csv = pd.read_csv('./user_input/review_input.csv')

    #Exclude critics that aren't in the training set
    all_critics = critic_csv['critic'].tolist()
    contained_critics = list(set(all_critics).intersection(set(crot_date.critic)))
    non_contained_critics = list(set(all_critics) - set(contained_critics))

    critic_csv2 = critic_csv[critic_csv.critic.isin(contained_critics)]
    critic_name_list = critic_csv2['critic'].tolist()
    critic_review_list = critic_csv2['val'].tolist()

    #Turn user inputs into a dataframe that can be used as input to prediction function
    pred_df = pd.DataFrame(np.asarray(meta_dat + critic_review_list).reshape(1,-1), columns = meta_cols + critic_name_list)


    ############################################
    #Run prediction model

    #First show user who which critics are not contained in training set
    #Wait for them to press enter to continue


    #Compute the similarity of the non-contained critics to all critics in the dataset
    #This checks for the simple spelling mistakes
    from difflib import SequenceMatcher
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    closest_match = []
    critic_names = list(set(crot_date.critic))
    for bad_critic in non_contained_critics:
        sim_met = [similar(q, bad_critic) for q in critic_names]
        ipdf = pd.DataFrame(zip(sim_met, critic_names), columns=['sim','critic'])
        ipdf_sort = ipdf.sort('sim', ascending=False)
        closest_match.append(ipdf_sort.critic.tolist()[0])

    match_df = pd.DataFrame(zip(non_contained_critics, closest_match), \
                            columns=['User Input', 'Closest Match'])

    print("")
    print("")
    print("-----------------------------------")
    print("")
    print("Some inputted critics are not contained in the dataset. Below are the user inputted critics and closest matches found in the training set.")
    print("")
    print("Please check for spelling inconsistencies and update the 'review_input.csv' file accordingly:")
    print("")
    print("")
    print(match_df)
    print("")
    print("")
    raw_input("To proceed with fitting the model, press Enter to continue...")


    subpred, \
    valerror, \
    subpred_all, \
    valerror_all, \
    g_pred, \
    actual_models, \
    potential_models, \
    elapsed,\
    which_critics = run_dynamic_prediction_xg(pred_df, \
                                              crot_date, \
                                              Xtrain, \
                                              Ytrain, \
                                              meta_cols,  \
                                              critic_name_list, \
                                              max_iter=max_critics, \
                                              mod_thres=model_thres)


    #Print Model results to file
    orig_stdout = sys.stdout
    f = open('./results/Prediction_Results.txt', 'w')
    sys.stdout = f

    print("")
    print("------- CINEMASCORE DYNAMIC PREDICTION RESULTS ------")
    print("")
    print("The critic(s) ")
    print("")
    print(non_contained_critics)
    print("")
    print("are not contained in the training set and were excluded from the model")
    print("")
    print("The following critics were used to during fitting: ")
    print("")
    print(which_critics)
    print("")
    print("Size of training set: " +  str(Xtrain.shape[0]))
    print("")
    print("Potential models: " + str(potential_models))
    print("Time to fit: " + str(np.round(elapsed,2)) + " seconds")
    print("Baseline (Genre + Month of Release) Prediction: " + str(np.round(g_pred,2)))
    print("")
    print("------------ Baseline - beating models --------------")
    print("Models fit: " + str(len(subpred.values())))
    print("Mean CinemaScore prediction: " + str(np.round(np.mean(subpred.values()),2)))
    print("5th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred.values(), 5),2)))
    print("95th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred.values(), 95),2)))
    print("----------------- All models ------------------------")
    print("Models fit: " + str(len(subpred_all.values())))
    print("Mean CinemaScore prediction: " + str(np.round(np.mean(subpred_all.values()),2)))
    print("5th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred_all.values(), 5),2)))
    print("95th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred_all.values(), 95),2)))
    print("-----------------------------------------------------")
    print("")
    print("Translate CinemaScores using the following table:")
    print("")
    print(pd.Series(["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"], index=range(13)))

    sys.stdout = orig_stdout
    f.close()

    #Plot results of both baseline-beating and all models

    #Baseline beating plot
    plt.hist(subpred.values(), bins=range(13), color='dodgerblue')
    plt.axvline(np.percentile(subpred.values(), 5), color='k', alpha=.3, linestyle='--', linewidth=5, label='5-95th percentile')
    plt.axvline(np.percentile(subpred.values(), 95), color='k', alpha=.3, linestyle='--', linewidth=5)
    plt.axvline(np.mean(subpred.values()), color='indianred', alpha=.7, linestyle='--', linewidth=5, label='Mean')
    plt.axvline(g_pred, color='forestgreen', alpha=.7, linestyle='--', linewidth=5, label='Baseline')
    plt.xticks(range(13), ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"])
    plt.ylabel('Number of models')
    plt.xlabel('CinemaScore')
    plt.title('Score Distribution: # baseline beating models= '+str(len(subpred.keys())))
    plt.legend()
    plt.savefig('./results/CinemaScore_above_baseline.png')
    plt.close()

    #All models
    plt.hist(subpred_all.values(), bins=range(13), color='dodgerblue')
    plt.axvline(np.percentile(subpred_all.values(), 5), color='k', alpha=.3, linestyle='--', linewidth=5, label='5-95th percentiles')
    plt.axvline(np.percentile(subpred_all.values(), 95), color='k', alpha=.3, linestyle='--', linewidth=5)
    plt.axvline(np.mean(subpred_all.values()), color='indianred', alpha=.7, linestyle='--', linewidth=5, label='Mean')
    plt.axvline(g_pred, color='forestgreen', alpha=.7, linestyle='--', linewidth=5, label='Baseline')
    plt.xticks(range(13), ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"])
    plt.ylabel('Number of models')
    plt.xlabel('CinemaScore')
    plt.title('Score Distribution: All models= '+str(len(subpred_all.keys())))
    plt.legend()
    plt.savefig('./results/CinemaScore_all_models.png')
    plt.close()

    #Save sub-models and cross-validation errors to disk
    pickle.dump(subpred_all, open( './results/individual_predictions.p', 'wb' ))
    pickle.dump(valerror_all, open( './results/cross_validated_error.p', 'wb' ))

    print('')
    print('---------------------------------------------')
    print('MODEL RUNS HAVE FINISHED')
    print('')
    print('Consult results/Prediction_Results.txt for results of model')
    print('---------------------------------------------')








