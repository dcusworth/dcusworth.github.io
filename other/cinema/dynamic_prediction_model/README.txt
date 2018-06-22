——————————————————————————————————————————————————————————————————————
README for running dynamic CinemaScore prediction model
——————————————————————————————————————————————————————————————————————

This model searches the database of RottenTomatoes reviews
and iteratively fits a unique XGBoost model for each 
combination of critics supplied by the user, assuming at 
least 50 films to train on. The model then outputs the 
distribution of all CinemaScore predictions.

developed by Daniel Cusworth (dcusworth@legendary.com; dcusworth@fas.harvard.edu)


——————————————————————————————————————————————————————————————————————

****** CONTENTS OF DIRECTORY ******

	ancillary_data/ - Contains ancillary data used to train/update model
		- better_release_dates.json: JSON containing all films in training set with their release dates
		- cinemascore_data.csv: CSV with film title, CinemaScore, and genre
		- critics_in_dataset.csv: CSV with names of scraped critics, sorted by total number of reviews
		- films_with_no_cinemascore.csv: CSV containing scraped films that have no associated CinemaScore

	class_func/ - Contains scripts to clean, normalize, and train model
		- class_func.py: Functions called my main driver ‘dynamic_prediction.py’ to run model

	dynamic_prediction.py: Main driver of dynamic XGBoost model
		- run from command line to drive model - See section ‘RUNNING THE MODEL’

	expand_dataset.py: Script to compare Rotten Tomatoes scrape with existing CinemaScores and
				saves non-overlapping films in ‘films_with_no_cinemascores.csv’

	merge_user_cinemascore.py - Script that reads contents of ‘films_with_no_cinemascores.csv’ and
				merges with existing cinemascore and release date databases. To be run
				after the user has manually updated films_with_no_cinemascores.csv

	Model_Driver.ipynb: Jupyter interface for running dynamic prediction model. 

	README: README (this document) explaining files contained in model

	results/ - Contains output text/plots from running dynamic prediction model

	rotdat/ - Contains Rotten Tomatoes scraping results.
		- current_rcritics.json: Names and URL slugs for all Rotten Tomatoes critics
		- rcritic_reviews_*.json: JSON with critic reviews, keyed by critic name
	
	rottentomatoes_scraper.py: Rotten Tomatoes scraping function. Updated as needed as
				Rotten Tomatoes periodically updates its html structure.

	update_critic_list.py - Run after rescraping Rotten Tomatoes to get summary of contained critics.
				Output is saved in critics_in_dataset.csv

	Update_Data.ipynb - Jupyter processing script to scrape Rotten Tomatoes dataset and add CinemaScore.

	user_input/ - Directory containing files user needs to manipulate before running the model
		-film_review_scraper.py: Script that actively scrapes Rotten Tomatoes for all critic reviews
			given a URL. Saves output into ‘review_input.cv’
		-metadata_input.csv: CSV that user updates to provide genre/release information used in the model.
		-review_input.csv: CSV with critic reviews for particular film - can be edited manually
			or can be output from film_review_scraper.py


——————————————————————————————————————————————————————————————————————

****** RUNNING THE MODEL from COMMAND LINE ******

(1) Make sure following XGBoost for python is installed on your system

(2) Modify inputs in the directory user_input/:

	- Modify metadata_input.csv:
		- (a) long-form genre: the text must exactly match one of the following (without quotes): 'Sci-Fi', 'Anthology', 'Sports', 'Crime', 'Romance', 'Supernatural', 'Comedy', 'War', 'Teen', 'Biopic', 'Horror', 'Sequel', 'Western', 'Thriller', 'Adventure', 'Mystery', 'Foreign', 'Drama', 'Historical', 'Remake', 'Action', 'Animated', 'Documentary', 'Musical', 'Spy', 'Family', 'Romantic Comedy', 'Period', 'Fantasy', 'Adaptation', 'Children'
		- (b) Short genre: The text must exactly match one of the following (without quotes): 'prestige', 'family', 'horror', 'sequel', 'other', 'action', 'comedy', 'pulp'
		- (c) Month of release (scale 1-12)

	- Modify the review_input.csv
		- Option (1): For each critic, make a new line: 
			first column is critic name, second column is review on 0-1 scale
		- Option (2): Run script film_review_scraper.py and provide a URL at command line. Example input:
			python film_review_scraper.py -u https://www.rottentomatoes.com/m/FILM_SLUG_HERE

(3) Run dynamic prediction model

	- Example command: python dynamic_prediction.py -n 15 -t 50 -p 0 -i 0
		-flag -n: MAX CRITIC - what is the maximum number of features you allow.
			If review_input.csv has more critics than MAX CRITIC, then only the “top” n critics are considered.
			Here, “top” is defined as critics with a large number of reviews who correlated by themselves
			with CinemaScore. Default is 15
		-flag -t: Threshold of common films to fit individual model. I.e, a threshold of 50 means that for
			the kth combination of critics, a model will only be fit if there are 50 common reviews among them
		-flag -p: Preprocessing flag. Consult class_func.py for specific notes. Recommended setting is 0, which
			means no critics are a priori excluded from fitting.
		-flag -i: Imputation scheme. Recommended value is 0. Determines whether to not to impute along an axis.
			Consult class_func.py for more information

	- If a critic is provided in review_input.csv, but not contained in the training set, the script will prompt you
	  about the discrepancy. There may be a typo in review_input.csv, or the Rotten Tomatoes data may need to be rescraped.

(4) Check results

	- Output contained in directory results/. Individual model predictions and cross-validated errors are saved (pickled) in ‘individual_predictions.p’ and ‘cross_validated_error.p.’ Histograms of review distributions are saved for the cases where all models are considered, and just models who beat the baseline (model fit with just metadata) on the training set. Potential models is defined as the number of unique combinations of all provided critics provided in review_input.csv. Actual models is defined as the total number of critic combinations that had sufficient common films to fit an individual model


****** IPYTHON INTERFACE ******

There is an additional iPython/Jupyter interface that runs the model. Consult/run the notebook ‘Model_Driver.ipynb’ for instructions and to run from that interface. The script scrapes Rotten Tomatoes, updates metadata, runs model, and visualizes output. 

To add data to the training set, follow the instructions in the Jupytr notebook ‘Update_Data.ipynb’


