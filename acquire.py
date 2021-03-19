from env import host, user, password
import pandas as pd
import numpy as np

def acquire_zillow():
    '''
    Grab our data from SQL
    '''
    sql_query = '''select *
    from properties_2017
    join predictions_2017 using(parcelid)
    where transactiondate like '2017-05-%%' 
    or transactiondate like '2017-06-%%'
    '''
    connection = f'mysql+pymysql://{user}:{password}@{host}/zillow'
    df = pd.read_sql(sql_query, connection)
    return df
