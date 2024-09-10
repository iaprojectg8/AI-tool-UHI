from utils.imports import *
from tools.preprocessing import *

def rdf_grid_search(X_train,y_train):
    """
    Performs grid search on rdf model in order to optimize it.

    Parameters:
        X_train (pd.DataFrame): The dataframe containing the variables
        y_train (pd.DataFrame): The dataframe containing the target variable

    Returns:
        results : Returns the results as a dictionary
    """
    
    rf_regressor = RandomForestRegressor()

    # Define the hyperparameters and their values to search
    param_grid = {
        'n_estimators': [15, 17, 20, 23, 25, 28],  # Number of trees in the forest
        'max_depth': [None, 10, 15, 20],      # Maximum depth of the trees
        'min_samples_split': [2, 3, 5, 7, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4, 6, 8, 10]     # Minimum number of samples required to be at a leaf node
    }

    # Define the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5,n_jobs=-1, scoring=['neg_mean_squared_error',"r2"],verbose=3,refit=False,return_train_score=False)

    # Perform the grid search
    grid_search.fit(X_train, y_train)  # X_train and y_train are your training data and labels

    # Get the results of each test
    results = grid_search.cv_results_

    # Print the results to a text file
    with open('grid_search_results.txt', 'w') as f:
        f.write("Grid Search Results:\n\n")
        f.write("Hyperparameters, Mean Squared Error\n")
        for params, score in zip(results['params'], results['mean_test_score']):
            f.write(f"{params}, {score}\n")

    return results

