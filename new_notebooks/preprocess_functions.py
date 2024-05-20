import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.metrics.pairwise import rbf_kernel


def is_north_city(df):
    df_columns = df.columns.to_list()
    north_cities = ['Delhi','Kolkata']

    return (
        df.assign(**{
            f"{col}_is_north" : (df
                                 .loc[:,col]
                                .isin(north_cities)
                                .astype(int))
            for col in df_columns
        })
        .drop(columns=df_columns)
    )
    
    
    
def part_of_day(df,morning=6,noon=12,evening=16,night=21):
    columns = df.columns.to_list()

    time_df = df.assign(**{
        col: pd.to_datetime(df.loc[:,col],format='mixed').dt.hour
        for col in columns
    })

    return (
        time_df.assign(**{
            f'{col}_part_of_day' : (
                np.select(condlist=[time_df.loc[:,col].between(morning,noon,inclusive='left'),
                                    time_df.loc[:,col].between(noon,evening,inclusive='left'),
                                    time_df.loc[:,col].between(evening,night,inclusive='left')],
                         choicelist=["morning",
                                     "noon",
                                     "evening"],
                         default="night")
            )
            for col in columns
        })
        .drop(columns=columns)
    )
    
    

def duration_categories(df,short=0,medium=400,long=1000):
    return (
        df.assign(
            duration_category = np.select(
                condlist=[df.loc[:,'duration'].between(short,medium,inclusive='left'),
                         (df.loc[:,'duration'].between(medium,long,inclusive='left'))],
                choicelist=['short','medium'],
                default='long'
            )
        )
        .drop(columns='duration')
    )
    
    
def is_direct_flight(df):
    columns = df.columns.to_list()

    return(
        df.assign(**{
            f"is_drect": np.where(df.loc[:,col] == 0, 1, 0)
            for col in columns
        })
    )
    
    
def have_info(df):
    columns = df.columns.to_list()

    return(
        df.assign(**{
            f"have_info": np.where(df.loc[:,col] == "No Info", 0, 1)
            for col in columns
        })
        .drop(columns=columns)
    )
    
    
class RbfSimilarityScore(TransformerMixin, BaseEstimator, OneToOneFeatureMixin):

    def __init__(self,percentiles,variables=None):
        self.percentiles = percentiles
        self.variables = variables

    def fit(self,X,y=None):
        if not self.variables:
            self.variables = X.columns.to_list()
        self.reference_values_ = (
            {col: X.loc[:,[col]].quantile(self.percentiles).values
            for col in self.variables}
        )
        return self

    def transform(self,X):
        return ( X.assign(**{
                f'{int(percentile * 100)}percentile_rbf_score' : rbf_kernel(X=X.loc[:,self.variables],
                                                                            Y=self.reference_values_[self.variables[0]][ind,:].reshape(-1,1))
                for ind,percentile in enumerate(self.percentiles)})
                .drop(columns=self.variables)
               )
        

