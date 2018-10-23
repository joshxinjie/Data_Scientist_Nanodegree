# Disaster Response Pipeline Project

'''
Instructions for udacity. To be deleted later
1. Open a Terminal
2. Enter the command `env | grep WORK` to find your workspace variables
3. Enter the command `cd 1_flask_exercise` to go into the 1_flask_exercise folder.
4. Enter the command `python worldbank.py`
5. Open a new web browser window and go to the web address:
`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values.
'''

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
