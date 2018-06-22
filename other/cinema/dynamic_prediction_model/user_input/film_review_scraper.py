import numpy as np
import pandas as pd
import time
from bs4 import BeautifulSoup
import requests
import datetime
import unittest, time, re
import itertools
import sys, getopt
import argparse


#Input argument - URL -  to scraping script

parser = argparse.ArgumentParser(description='Arguments to run RT scraper for individual film')
parser.add_argument('-u','--url', help='URL of film to scrape. Must be of form https://www.rottentomatoes.com/m/FILM_SLUG_HERE',required=True)

args = parser.parse_args()

input_url = args.url
    
    
#Define function to get page data from RottenTomatoes
def open_page2(url):
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'}
    red_page=requests.get(url, headers=headers)
    soup = BeautifulSoup(red_page.text, "html.parser")
    
    return soup

#Scraping function to get reviews of individual page
def get_page_reviews(page_dat):
    
    #Get all reviews on page
    review_table = page_dat.find_all('div', attrs={'class': 'row review_table_row'})
    
    reviews = {}
    for idx in range(len(review_table)):

        #Get Critic Name
        name_element = review_table[idx].find('div', attrs={'class':'col-sm-13 col-xs-24 col-sm-pull-4 critic_name'})
        critic_name1 = name_element.find('a', attrs={'class':'unstyled bold articleLink'}).get_text()
        critic_name = critic_name1.encode('utf-8')

        #Get binary - whether RT considered review fresh or rotten
        rev_icon = review_table[idx].find('div', attrs={'class':'col-xs-16 review_container'})            .find('div', attrs={'class', 'review_icon'})            .get('class')

        if 'fresh' in rev_icon:
            icon_score = 1
        elif 'rotten' in rev_icon:
            icon_score = 0
        else:
            icon_score = None

        #See what Original Score is, evaluate if on X/X scale or A-F scale
        full_score = review_table[idx].find('div', attrs={'class': 'small subtle'}).get_text()
        if 'Original Score' in full_score:
            itextall = full_score.split(' Full Review | Original Score: ')
            
            if len(itextall) > 1:
                itext = itextall[1]
            
                if '/' in itext:
                    ivals = itext.split('/')
                    iscore = float(ivals[0]) / float(ivals[1])
                elif itext in grades:
                    iscore = grade_map[itext]

                else:
                    iscore = icon_score
            else:
                iscore = icon_score

        #If binary/assign 0,1 for Fresh/Rotten
        else:
            iscore = icon_score

        #Save to dictionary
        reviews[critic_name] = iscore
    
    return reviews

####### RUN SCRAPER ###########

#Based on user input - get page 1 reviews
#input_url = 'https://www.rottentomatoes.com/m/the_fate_of_the_furious'
page1_url = input_url + '/reviews/?page=1&sort='

#Open page
page_dat = open_page2(page1_url)

#Get total number of pages to scrape
total_pages = int(page_dat.find('span', attrs={'class': 'pageInfo'}).get_text().split('Page 1 of ')[1])

grades = ['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F']
grade_map = dict(zip(grades,reversed(np.linspace(0,1, len(grades)))))

#Initialize review table by getting first page
all_reviews = get_page_reviews(page_dat)

#Loop over all pages
for ipage in range(2,total_pages+1):
    time.sleep(1)
    ipage_dat = open_page2(input_url + '/reviews/?page=' + str(ipage) + '&sort=')
    all_reviews.update(get_page_reviews(ipage_dat))

#Write reviews to disk
reviews_df = pd.DataFrame(all_reviews.items(), columns = ['critic', 'val'])
reviews_df.to_csv('review_input.csv', index=False)

