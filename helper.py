from import_libraries import *

class Helper:
    
    def print_reduce_perc(self,old_df,new_df):
        """
        Print out length difference between dataframes
        
        Parameters
        ----------
            old_df: pandas Dataframe
                Dataframe before any manipulation
            old_df: pandas Dataframe
                Dataframe after manipulation
        """
        
        old_df_len = len(old_df)
        new_df_len = len(new_df)
        reduce = round(100 - (new_df_len * 100 / old_df_len), 2)
        display(f'Dataframe length -  before: {old_df_len}, after: {new_df_len}. Size reduction: {reduce}%')
    

    def reduce_cat_num(self,df_col=False,assign=np.nan,num=100):
        """
        Assign np.nan to all rows except `num` most frequent values

        Parameters
        ----------
            df_col: 
                Dataframe column
            num: int, default=100
                How many values to left
            assign: default=np.nan
                Value what would be assigned to other values
            
        Return
        ------
            top_col_names: list
                List of most frequent column values
            assign: 
                Value what was assign
            df_col: pandas Dataframe column
                Processed column
        """
        
        top_col_names = df_col.value_counts()[:num].index.tolist()
        if assign in top_col_names: top_col_names.remove(assign)
        df_col.loc[~df_col.isin(top_col_names)] = assign

        return top_col_names, assign, df_col


    def apply_reduced_cats(self,df_col=False,top_col_names=[],assign=np.nan):
        """
        Usually you want to use this method to apply to test set most frequent values from train set
            
        Parameters
        ----------
            df_col: 
                Dataframe column
            top_col_names: list, default=[]
                List of most frequent column values
            assign: default=np.nan
                Value what would be assigned to other values
            
        Return
        ------
            df_col: pandas dataframe column
                Processed column
        """
        
        if top_col_names:
            df_col.loc[~df_col.isin(top_col_names)] = assign
            
            return df_col
        else:
            raise Exception('top_col_names should not be empty')
            
            
    def plot_bar_vs_labels(self,df,feature_col_name,labels_col_name,title=None,xlabel=None,ylabel=None,legend_title=None):
        """
        Helper function to make bar plot
        
        Parameters
        ----------
            df:
                Pandas dataframe
            feature_col_name: str
                Name of column what will be used
            labels_col_name: str
                Name of column what will be used
            title: str, default=None
                Plot title
            xlabel: str, default=None
                Label for the x-axis
            ylabel: str, default=None
                Label for the y-axis
            legend_title: str, default=None
                Legend title
        """
        
        plt.figure(figsize=(16,8))

        x = np.arange(len(df[feature_col_name].value_counts()))

        data_grouped = df.groupby(labels_col_name)
        label_values = df[labels_col_name].unique().tolist()
        width = 0.2
        for i, col in enumerate(label_values):
            y = data_grouped.get_group(col)[feature_col_name].value_counts().sort_index()
            plt.bar(x-width+(width*i), y, width)

        plt.xticks(x, df[feature_col_name].value_counts().sort_index().index)
        plt.title(title or f'{feature_col_name} vs {labels_col_name}',fontsize=16)
        plt.xlabel(xlabel or feature_col_name,fontsize=14)
        plt.ylabel(ylabel or 'Count',fontsize=14)
        plt.legend(label_values,title=legend_title or labels_col_name,title_fontsize=14)
        plt.show()
        
        
    def selectEstimator(self,estimators,X_train,X_test,y_train,y_test,cv=3,verbose=False):
        """
        Fit list of models and get score and best params(for GridSearchCV) info
        
        Parameters
        ----------
            estimators: list of dictionaries
                {
                    'pipeline': Pipeline(),
                    'param_grid': [{}] # optional for GridSearchCV
                }
            X_train,X_test,y_train,y_test: pandas dataframes
                Train and test data
            cv: int, default=3
                Number of fold for cross validation (read GridSearchCV docs)
            verbose: bool, default=False
                Print info at runtime
                
        Return
        ------
            models:
                Fitted models dictionary
            results: pandas dataframe
                Sorted dataframe with columns=['Classifier','Train score','Test score','Best params']
            
        """
        
        models = {}
        results = []

        for estimator in estimators:
            est_name = estimator['pipeline']['classifier'].__class__.__name__

            if verbose:
                print('\nRunning: ',est_name)

            if 'param_grid' in estimator:
                model = GridSearchCV(
                    estimator['pipeline'],
                    param_grid=estimator['param_grid'],
                    scoring='accuracy',cv=cv)
            else:
                model = estimator['pipeline']

            models[est_name] = model.fit(X_train,y_train)

            train_score = "%.3f" % model.score(X_train,y_train)
            test_score = "%.3f" % model.score(X_test,y_test)

            results.append([
                est_name, train_score, test_score, model.best_params_ if 'param_grid' in estimator else ''
            ])

            if verbose:
                print('\tTrain score: ',train_score)
                print('\tTest score:  ',test_score)

                if 'param_grid' in estimator:
                    print('\tBest params: ',model.best_params_)


        results = pd.DataFrame(results, columns=['Classifier','Train score','Test score','Best params'])
        results = results.sort_values(['Test score'],ascending=False)

        return models, results
