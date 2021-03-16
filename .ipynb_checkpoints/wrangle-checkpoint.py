# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.
# The first step will be to acquire and prep the data. Do your work for this exercise in a file named wrangle.py.
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env import host, user, password
from sklearn.model_selection import train_test_split
import acquire

# 1. Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.

def get_churn_data():
    '''
    This function reads telco_churn in SQL and returns a dataframe with
        'customer_id'
        'monhtly_charges'
        'tenure'
        'total_charges'
    with only customers that have a 2 year contract.'''
    sql_query = '''select customer_id, monthly_charges, tenure, total_charges
    from customers
    join contract_types using(contract_type_id)
    where contract_type = "Two Year"
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
    df = pd.read_sql(sql_query, connection)
    return df


# 2. Walk through the in regression acqusition notes above using your new dataframe. You may handle the missing values however you feel is appropriate.

#df.head()
    # no issues spotted so far

#df.shape
    # 1696 rows and 4 columns

#df.describe()
    #looking good still

#df.info()
    # all columns have 1695 non nulls which tells us we dont have any at all

    # But lets just check and make sure there are no nulls to be safe
#print(df.isnull().sum())
    # cool no nulls

#df.total_charges.value_counts()
    # Foudn a problem there is a blank value for total charges

    # lets convert any and all of the white space in into a null value so we can remove it
#df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        # this removes all white spaces and replaces with null for whole df
        # If you wanted to just do the column you can with:
            # df['total_charges'] = df['total_charges'].replace(r'^\s*$', np.nan, regex = True)

    # Now lets check how many nulls we have in total_charges see if it changed
#df.total_charges.isnull().sum()
    # Dang look now there are 10 where there use to be 0

    # do we have any in other columns now?
#df.info()
    # looks like just total_charges's.non-null count changed

    # now lets get rid of the rows with null values!
#df = df.dropna()

#df.info()
    # Now everything has the same amount of Non-Null values
        # 1685

    # convert total_charges to a float
#df['total_charges'] = df.total_charges.dropna().astype('float')

    # Dropping the customer_id column becasue we dont want to cause issues later
        # can always be added back in later on
    # df = df.drop(columns = 'customer_id')

#plt.figure(figsize=(16, 3))

#for i, col in enumerate(['monthly_charges', 'tenure', 'total_charges']):  
    #plot_number = i + 1 # i starts at 0, but plot nos should start at 1
    #series = df[col]  
    #plt.subplot(1,4, plot_number)
    #plt.title(col)
    #series.hist(bins=5, color = "skyblue", ec="black")
    
    # This is just a way to make it look better in my opinion

#plt.subplots(1, 3, figsize=(20,5), sharey=True)
#plt.title('Initial Pokemon - 1st Generation')
#sns.set(style="darkgrid")
    # Monthly Charges
#plt.subplot(1,3,1)
#plt.hist(data=df, x='monthly_charges', color="powderblue")
#plt.title('Distribution of Monthly Charges')

    # Tenure
#plt.subplot(1,3,2)
#plt.hist(data=df, x='tenure', color="skyblue")
#plt.title('Distribution of Tenure')

    # Total Charges
#plt.subplot(1,3,3)
#plt.hist(data=df, x='total_charges', color="steelblue")
#plt.title('Distribution of Total Charges')
    
    
    # Make box plots for each one    
#plt.subplots(1, 3, figsize=(20,15), sharey=True)
#fig.suptitle('Initial Pokemon - 1st Generation')
#sns.set(style="darkgrid")
    # Monthly Charges
#plt.subplot(1,3,1)
#sns.boxplot(data=df.monthly_charges, color="powderblue")
#plt.title('Monthly Charges')

    # Tenure
#plt.subplot(1,3,2)
#sns.boxplot(data=df.tenure, color="skyblue")
#plt.title('Tenure')

    # Total Charges
#plt.subplot(1,3,3)
#sns.boxplot(data=df.total_charges, color="steelblue")
#plt.title('Total Charges')

    # Split the data into train, validate, and test
#train, test = train_test_split(df, train_size=0.8, random_state=1234)
#train, validate = train_test_split(train, train_size=0.7, random_state=1234)

#train.head()
#validate.head()
#test.head()


# 3. End with a python file wrangle.py that contains the function, wrangle_telco(), that will acquire the data and return a dataframe cleaned with no missing values.


## This acquire_telco is the same one found in acquire.py
def acquire_telco():
    '''
    Grab our data from SQL
    '''
    sql_query = '''select customer_id, monthly_charges, tenure, total_charges
    from customers
    join contract_types using(contract_type_id)
    where contract_type = "Two Year"
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
    df = pd.read_sql(sql_query, connection)
    return df


def clean_telco():
    '''
    Pull data form 'acquire_telco'
    Takes in df of students grades and cleans the data
    replace empty cells with null values
    drop null values
    convert 'total_charges' from dtype object to float
    
    returns a clean df as a pandas df
    '''
    # get data from acquire_telco
    df = acquire_telco()
    # replace empty cells with null values
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    # drop null values
    df = df.dropna()
    # convert 'total_charges' from dtype object to float
    df['total_charges'] = df.total_charges.dropna().astype('float')
    return df

def split_telco():
    '''
    split our data
    takes in a pandas dataframe
    returns 3 panda dataframs:
        train
        test
        validate
    '''
    train, test = train_test_split(df, train_size=0.8, random_state=1234)
    train, validate = train_test_split(train, train_size=0.7, random_state=1234)
    return train, validate, test


def wrangle_telco():
    '''
    Pulls data form acuire.py's 'acquire_telco'
    Takes in df of students grades and cleans the data
    replace empty cells with null values
    drop null values
    convert 'total_charges' from dtype object to float
    split our data
    takes in a pandas dataframe
    returns 3 panda dataframs:
        train
        test
        validate
    '''
    df = clean_telco(acquire.acquire_telco())
    return split_telco(df)