##############################################################
#
# ROTTEN TOMATOES SCRAPING SCRIPT
#
# This script scrapes rottentomates by critic - i.e.,
# Loops through all critics and retrieves their reviews, scores,
# publication, date, and brief snippet of their review
#
# Output an individual JSON for all critics of certain surname
# (first letter)
#
# Created by Daniel Cusworth - 2/21/2017
#
##############################################################

#Load needed functions
import oauth2
import simplejson
import numpy as np
import scipy as sp
import pandas as pd
import time
from bs4 import BeautifulSoup
import requests
import datetime
import json
import random
from datetime import date, timedelta, time
import unittest, time, re
import itertools
import logging
import sys
import string

#Load Selenium webdriver - need to have this installed
#Need to have phantom webdriver installed if using function open_page()
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException


#Define function that uses phantom browsing to open webpage
#Need to have phantom browser installed - i.e. must  call function Phat
def open_page(url):
    
    try:

        driver = webdriver.PhantomJS() # or add to your PATH
        driver.set_window_size(1024, 768) # optional
        driver.get(url)

        ipage = driver.page_source.encode('utf-8')
        driver.close()

        return BeautifulSoup(ipage, 'html.parser')
    
    except:
        
        return "NOPE"

    
#Define function that uses requests to open webpage
#Rotten tomatoes is nice to requests - so try this one first!
def open_page2(url):
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'}
    red_page=requests.get(url, headers=headers)
    soup = BeautifulSoup(red_page.text, "html.parser")
    
    return soup



####### DO SCRAPING #########

#If you want to do regular + 'legacy' critics - i.e. critics who are dead or are no longer
#reviewing films, set this to True
do_legacy = False


#Loop over all letters of alphabet and scrape critics
#This gets the list of critics rottentomatoes has, and saves their critic url slug
#We save this data to input into another data scraping function
base_url = 'https://www.rottentomatoes.com/critics/authors?letter='
legacy = 'https://www.rottentomatoes.com/critics/legacy_authors?letter='
critics = {}
for iletter in list(string.ascii_lowercase):
    time.sleep(1)
    
    #First do current critics of letter
    critic_page = open_page2(base_url + iletter)
    critic_table = critic_page.find_all('p', attrs={'class':'critic-names'})

    for jcritic in critic_table:
        icritic = jcritic.find_all('a', href=True)[0]
        critic_name = icritic.get_text()
        critic_url = icritic['href']

        critics[critic_name] = critic_url
        
    #Now do legacy critics of certain letter
    if do_legacy:
        critic_page = open_page2(legacy + iletter)
        critic_table = critic_page.find_all('p', attrs={'class':'critic-names'})

        for jcritic in critic_table:
            icritic = jcritic.find_all('a', href=True)[0]
            critic_name = icritic.get_text()
            critic_url = icritic['href']

            critics[critic_name] = critic_url
        
    print 'Did letter', iletter


#Save output
with open('rotdat/current_rcritics.json', 'w') as fp:
    json.dump(critics, fp)

#If you already saved critic data, uncomment here
#with open('rotdat/current_rcritics.json', 'r') as fp:
#    critics = json.load(fp)


#Using beautiful soup, get the actual data given a critic url page
def get_reviews(critic_reviews, film_data):
    try:
        big_table = critic_reviews.find('table', attrs={'class':'table table-striped'})
        body_table = big_table.find_all('tr')[1:]
        for ifilm in body_table:

            irscore = ifilm.find('span', attrs={'class': 'icon'}).get('title')
            if irscore=='':
                icon_text = ifilm.find('span', attrs={'class': 'icon'}).get('class')
                if 'fresh' in icon_text:
                    irscore = 1
                elif 'rotten' in icon_text:
                    irscore = 0
                else:
                    irscore = ''

            itscore = ifilm.find('span', attrs={'class': 'tMeterScore'}).get_text()        
            iname = ifilm.find('a', attrs={'class': 'movie-link'}).get_text()
            itime = ifilm.find('div', attrs={'class': 'subtle'}).get_text()
            itime = itime.replace('\nRead More | ','')
            #itext = ifilm.find_all('td')[3]
            itext = ''
            ipub = ifilm.find_all('a', attrs={'class': 'unstyled articleLink'})[0].get_text()


            film_data[iname] = dict(rscore=irscore, tmeter = itscore, posted=itime, text=itext, pub=ipub)

    except:
        ""
    return film_data


#Given a critic and their URL, loop over all rottentomatoes pages they have
#Retrieve all review data
def get_critic_reviews_archaic(critic_i):
    
    critic_i_url = critics[critic_i]
    base_url2 = 'https://www.rottentomatoes.com'
    ipage = 1
    end_url = '/movies?page=' + str(ipage)
    critic_reviews = open_page2(base_url2 + critic_i_url + end_url)

    film_data = {}

    #See if button for next page exists
    is_next = critic_reviews.find('span', attrs={'class': 'glyphicon glyphicon-chevron-right'})
    if is_next:
        keep_going = True
    else:
        keep_going = False

    while keep_going:
        time.sleep(1)
        ipage += 1
        end_url = '/movies?page=' + str(ipage)
        critic_reviews = open_page2(base_url2 + critic_i_url + end_url)

        film_data = get_reviews(critic_reviews, film_data)
        is_next = critic_reviews.find('span', attrs={'class': 'glyphicon glyphicon-chevron-right'})
        if is_next:
            keep_going = True
        else:
            keep_going = False
            
    return film_data

def get_critic_reviews(critic_i):
    
    critic_i_url = critics[critic_i]
    base_url2 = 'https://www.rottentomatoes.com'
    ipage = 1
    end_url = '/movies?page=' + str(ipage)
    critic_reviews = open_page2(base_url2 + critic_i_url + end_url)

    is_next = critic_reviews.find('ul', attrs={'class': 'pagination'}).find_all('a')
    end_page = is_next[-1].get('href').replace('?page=','')

    film_data = {}
    
    for ipage in range(1,int(end_page)+1):
        try:
            time.sleep(1)

            end_url = '/movies?page=' + str(ipage)
            critic_reviews = open_page2(base_url2 + critic_i_url + end_url)
            film_data = get_reviews(critic_reviews, film_data)

            ipage += 1

        except:
            ''

    return film_data


#Loop over all critic names and get data
#Save data to JSON file
allcritics = critics.keys()
critic_letter = [q[0].lower() for q in allcritics]
counter=1
for iletter in list(string.ascii_lowercase):
    sel_critic = [q for q in allcritics if q[0].lower() == iletter]
    critics_reviews = {}
    for icritic in sel_critic:
        time.sleep(1)
        try:
            ireviewer_data = get_critic_reviews(icritic)
            critics_reviews[icritic] = ireviewer_data
        except:
            ""
        print 'Did', icritic, 'Num', counter, 'of', len(allcritics)
        counter += 1
    df_name = 'rotdat/rcritic_reviews_' + iletter + '.json'
    with open(df_name, 'w') as fp:
        json.dump(critics_reviews, fp)
    



