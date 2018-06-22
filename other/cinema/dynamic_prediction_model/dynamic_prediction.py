########################################################################
#
# CINEMASCORE PREDICTION DRIVER FUNCTION
#
# This script reads in the user's review & metadata inputs from files
# and calls the XGBoost training functions contained in 'class_func'.
# Prediction output is written to a text file and plots
#
# Daniel Cusworth - 4/24/17
#
########################################################################


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import json
import random
random.seed(200)
import unittest, re
import itertools
import string
import sys, getopt
import argparse
from functools import reduce 
import plotly
import plotly.graph_objs as go
    
#### Load user command line inputs #######
    
parser = argparse.ArgumentParser(description='Arguments to run CinemaScore prediction model')
parser.add_argument('-n','--max_critics', help='Maximum number of critics that the model will consider (default is 15)',required=False)
parser.add_argument('-t','--min_thres',help='Threshold required (i.e. minimum number of common films among critics) to fit a single XGBoost model (default is 50)', required=False)
parser.add_argument('-p','--which_process', help='Pre-processing/removal of critics (suggested/default value is zero)',required=False)
parser.add_argument('-i','--which_impute', help='Imputation on training set (suggested/default value is zero)',required=False)

args = parser.parse_args()

max_critics = int(args.max_critics)
if max_critics is None:
    max_critics = 15
    
model_thres = int(args.min_thres)
if model_thres is None:
    model_thres = 50
    
which_process = args.which_process
if which_process is None:
    which_process = 0
    
which_impute = args.which_impute
if which_impute is None:
    which_impute = 0
    
###################################################

#Load in ancillary functions
from class_func import *

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
    ipdf = pd.DataFrame(list(zip(sim_met, critic_names)), columns=['sim','critic'])
    ipdf_sort = ipdf.sort_values('sim', ascending=False)
    closest_match.append(ipdf_sort.critic.tolist()[0])
    
match_df = pd.DataFrame(list(zip(non_contained_critics, closest_match)), \
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
#raw_input("To proceed with fitting the model, press Enter to continue...")


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
f = open('../dynamic_prediction_model/results/Prediction_Results.txt', 'w')
sys.stdout = f

subpred_val = np.array(list(subpred.values()))
subpred_all_val = np.array(list(subpred_all.values()))

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
print("Models fit: " + str(len(subpred_val)))
print("Mean CinemaScore prediction: " + str(np.round(np.mean(subpred_val),2)))
print("5th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred_val, 5),2)))
print("95th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred_val, 95),2)))
print("----------------- All models ------------------------")
print("Models fit: " + str(len(subpred_all_val)))
print("Mean CinemaScore prediction: " + str(np.round(np.mean(subpred_all_val),2)))
print("5th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred_all_val, 5),2)))
print("95th Percentile CinemaScore prediction " + str(np.round(np.percentile(subpred_all_val, 95),2)))
print("-----------------------------------------------------")
print("")
print("Translate CinemaScores using the following table:")
print("")
print(pd.Series(["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"], index=range(13)))

sys.stdout = orig_stdout
f.close()

#Plot results of both baseline-beating and all models

#Baseline beating plot
plt.hist(subpred_val, bins=range(13), color='dodgerblue')
plt.axvline(np.percentile(subpred_val, 5), color='k', alpha=.3, linestyle='--', linewidth=5, label='5-95th percentile')
plt.axvline(np.percentile(subpred_val, 95), color='k', alpha=.3, linestyle='--', linewidth=5)
plt.axvline(np.mean(subpred_val), color='indianred', alpha=.7, linestyle='--', linewidth=5, label='Mean')
plt.axvline(g_pred, color='forestgreen', alpha=.7, linestyle='--', linewidth=5, label='Baseline')
plt.xticks(range(13), ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"])
plt.ylabel('Number of models')
plt.xlabel('CinemaScore')
plt.title('Score Distribution: # baseline beating models= '+str(len(subpred.keys())))
plt.legend()
plt.savefig('../dynamic_prediction_model/results/CinemaScore_above_baseline.png')
plt.close()

#All models
hy, hx, _ = plt.hist(subpred_all_val, bins=range(13), color='dodgerblue')
plt.axvline(np.percentile(subpred_all_val, 5), color='k', alpha=.3, linestyle='--', linewidth=5, label='5-95th percentiles')
plt.axvline(np.percentile(subpred_all_val, 95), color='k', alpha=.3, linestyle='--', linewidth=5)
plt.axvline(np.mean(subpred_all_val), color='indianred', alpha=.7, linestyle='--', linewidth=5, label='Mean')
plt.axvline(g_pred, color='forestgreen', alpha=.7, linestyle='--', linewidth=5, label='Baseline')
plt.xticks(range(13), ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"])
plt.ylabel('Number of models')
plt.xlabel('CinemaScore')
plt.title('Score Distribution: All models= '+str(len(subpred_all.keys())))
plt.legend()
plt.savefig('../dynamic_prediction_model/results/CinemaScore_all_models.png')
plt.close()

#Make plotly plot
ymax_val = np.max(hy)
hist_val = go.Histogram(x=subpred_all_val, name='Ensemble predictions', opacity=0.75,
	xbins = {'start':0, 'end': 13, 'size':1})
gline = go.Scatter(
	x = [g_pred, g_pred], 
	y = [0, ymax_val],
	name = 'Baseline genre prediction',
	line = dict(color='forestgreen', width=4, dash='dash')
)
mline = go.Scatter(
	x = [np.mean(subpred_all_val), np.mean(subpred_all_val)], 
	y = [0, ymax_val],
	name = 'Mean prediction',
	line = dict(color='black', width=4, dash='dash')
)
fline = go.Scatter(
	x = [np.percentile(subpred_all_val, 5), np.percentile(subpred_all_val, 5)], 
	y = [0, ymax_val],
	name = '5th percentile prediction',
	line = dict(color='indianred', width=4, dash='dash')
)
nline = go.Scatter(
	x = [np.percentile(subpred_all_val, 95), np.percentile(subpred_all_val, 95)], 
	y = [0, ymax_val],
	name = '95th percentile prediction',
	line = dict(color='indianred', width=4, dash='dash')
)

data = [hist_val, gline, mline, fline, nline]

layout = {
	'title':'Ensemble CinemaScore prediction results',
	'xaxis' : {'range':[0,13], 'title': 'CinemaScore, 0=A+, 13=F'},
	'yaxis' : {'range':[0, ymax_val], 'title': 'Number of Predictions'}
	}

#layout = {
#	'xaxis': {'range':[0,12]},
#	'yaxis': {'range':[0, hy.max()]},
#	'shapes': [
#		{
#		'type': 'line',
#		'x0': g_pred,
#		'x1': g_pred,
#		'y0': 0,
#		'y1': hy.max(),
#		'line': {'color':'forestgreen', 'width':3, 'dash':'dash'}
#		}
#		]
#	}


plotly.offline.plot({
	'data': data,
	'layout': layout
	},
	filename = './results/results.html'
	)	

#plt.axvline(np.percentile(subpred_all_val, 5), color='k', alpha=.3, linestyle='--', linewidth=5, label='5-95th percentiles')
#plt.axvline(np.percentile(subpred_all_val, 95), color='k', alpha=.3, linestyle='--', linewidth=5)
#plt.axvline(np.mean(subpred_all_val), color='indianred', alpha=.7, linestyle='--', linewidth=5, label='Mean')
#plt.axvline(g_pred, color='forestgreen', alpha=.7, linestyle='--', linewidth=5, label='Baseline')
#plt.xticks(range(13), ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"])
#plt.ylabel('Number of models')
#plt.xlabel('CinemaScore')
#plt.title('Score Distribution: All models= '+str(len(subpred_all.keys())))
#plt.legend()
#plt.savefig('./dynamic_prediction_model/results/CinemaScore_all_models.png')
#plt.close()

#Save sub-models and cross-validation errors to disk
pickle.dump(subpred_all, open( '../dynamic_prediction_model/results/individual_predictions.p', 'wb' ))
pickle.dump(valerror_all, open( '../dynamic_prediction_model/results/cross_validated_error.p', 'wb' ))
pd_dat = pd.DataFrame(list(zip(subpred_all_val, range(len(subpred_all_val)))), columns=['pred','ind'])
pd_dat['baseline'] = g_pred
pd_dat.to_csv('../dynamic_prediction_model/results/individual_predictions.csv')

print('')
print('---------------------------------------------')
print('MODEL RUNS HAVE FINISHED')
print('')
print('Consult results/Prediction_Results.txt for results of model')
print('---------------------------------------------')





