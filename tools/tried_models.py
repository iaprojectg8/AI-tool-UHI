
from utils.imports import *
from tools.preprocessing import create_X_y, clean_data, init_df_all_variable
from utils.variables_path import *
from tools.visualization import basic_visualization, importance_vis



def main_pca():
    params_to_take = []
    params_to_drop = ["LS2", "LS3", "LS5", "LS6", "ALB", "OCCSOL", "URB"]
    params_pca1 = ["LS2", "LS3", "LS5", "LS6", "ALB"]
    params_pca2 = ["OCCSOL", "URB"]
    params_list = [params_to_take, params_to_drop]
    
    df = init_df_all_variable()
    
    # Data prep for pca on correlated variables
    df_pca1 = df[params_pca1]
    df_pca2 = df[params_pca2]

    # Applying the PCA process
    pca1_components = pca_process(df_pca1, display=False)
    pca2_components = pca_process(df_pca2, display=False)
    
    # Adding the PCA main component to the dataframe
    pca1_df = pd.DataFrame(pca1_components.T, columns=[f'PC{i+1}' for i in range(pca1_components.shape[0])])
    pca2_df = pd.DataFrame(pca2_components.T, columns=[f'PC{i+1}' for i in range(pca2_components.shape[0])])
    df = pd.concat([df, pca1_df, pca2_df], axis=1)

    # Making the linear regression
    linear_regression(df=df, parameters_list=params_list)

def linear_regression(df, parameters_list):

    # y will always be the same because this is the variable we want to predict
    X,y = create_X_y(df, parameters_list)    
    scaler = StandardScaler()

    # Fit the scaler to your data and transform the data
    X = scaler.fit_transform(X)
    # Split the train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42,)
    

    # Model init and train
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    # Visualisation containing the results of the test
    basic_visualization(X_test,y_test, model)
    



def all_combination_regression(df, result_file):

    df = clean_data(df)
    all_variables = df.columns.tolist()
    target_variable = "LST"  
    variables = [v for v in all_variables if v!=target_variable]

    # Créer un DataFrame pour stocker les résultats
    results_df = pd.DataFrame(columns=["Variables utilisées", "R2 score", "MSE"])

    # Boucle sur toutes les combinaisons de variables
    for i in tqdm(range(1, len(variables) + 1)):
        for comb in itertools.combinations(variables, i):
            # Créer un modèle de régression linéaire
            X = df[list(comb)]
            y = df[target_variable]
            model = LinearRegression()
            
            # Entraîner le modèle
            model.fit(X, y)
            
            # Faire des prédictions
            y_pred = model.predict(X)
            
            # Calculer le R2 score et la MSE
            r2 = round(r2_score(y, y_pred),4)
            mse = round(mean_squared_error(y, y_pred),2)
            
            # Ajouter les résultats au DataFrame
            temp = pd.DataFrame({"Variables utilisées": [', '.join(comb)],
                                "R2 score": [r2],
                                "MSE": [mse]})
            results_df = pd.concat([results_df, temp], ignore_index=True)

    # Écrire les résultats dans un fichier CSV
    results_df = results_df.sort_values(["R2 score","MSE"],ascending=[False,True])
    results_df.to_csv(result_file, index=False,sep=";",encoding="utf-8")


def random_comparison(df):

    # y will always be the same because this is the variable we want to predict
    X,y = create_X_y(df)

    # Split the train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

    print(X_train.shape)

    # Model init and train
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    # Prediction and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred)

    # MSE for Mean Squarred Error
    print("For Y pred:")
    print("R2:",r2,"MSE",mse)

    # Random generation of point in temperature in the zone
    random_generated_temp = np.random.choice(y, size=y_pred.shape[0])   # The R2 is negative it means it is worse to always predict the mean
    # We can try with the mean then
    constant_mean = np.full_like(y_test,np.mean(y_test))
    mse_rand = mean_squared_error(y_test,constant_mean)
    r2_rand = r2_score(y_test,constant_mean)
    print("\nFor Y random:")
    print("R2:",r2_rand,"MSE",mse_rand)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, label=f'R2 = {r2:.2f}\nMSE = {mse:.2f}')
    plt.xlabel('Ground Truth Temperatures')
    plt.ylabel('Predicted Temperatures')
    plt.title('Predicted vs Ground Truth temperature')
    plt.legend()

    # Scatter plot for predicted vs. randomly generated temperatures
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, constant_mean, label=f'R2 = {r2_rand:.2f}\nMSE = {mse_rand:.2f}')
    plt.xlabel('Ground Truth Temperatures')
    plt.ylabel('Temperature Mean')
    plt.title('Comparison of Predicted vs Randomly Generated Temperatures')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlapping labels
    plt.show()




def pca_process(X,display=False):

    # Apply the PCA to the variable distribution
    pca = PCA()
    pca.fit(X)
    X_pca_components = pca.transform(X)
    
    # Visualize the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    print("Explained Variance Ratio:", explained_variance_ratio)


    resulting_components = list()
    for i in range(X_pca_components.shape[1]):
        if cumulative_variance_ratio[i] < 0.95:
            resulting_components.append(X_pca_components[:,i])

        # Get the first component above the 95% explainability
        elif cumulative_variance_ratio[i] >= 0.95:
            resulting_components.append(X_pca_components[:,i])
            break

    resulting_components = np.array(resulting_components)


    if display:

        # Plot the data in the new feature space
        fig = plt.figure(figsize=(13, 6))

        # Plot PCA in 2D
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(X_pca_components[:, 0], X_pca_components[:, 1], alpha=0.8)
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('PCA')
        ax1.grid(True)

        # Plot PCA in 3D
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(X_pca_components[:, 0], X_pca_components[:, 1], X_pca_components[:, 2], alpha=0.8)
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.set_zlabel('Principal Component 3')
        ax2.set_title('PCA in 3D')

        # Plot explained variance ratio
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8)
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title('Explained Variance Ratio for Each Principal Component')
        ax3.set_xticks(range(1, len(explained_variance_ratio) + 1))
        ax3.grid(True)

        # Calculate cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Plot cumulative explained variance ratio
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('Cumulative Explained Variance Ratio')
        ax4.set_title('Cumulative Explained Variance Ratio')
        ax4.set_xticks(range(1, len(cumulative_variance_ratio) + 1))
        ax4.grid(True)

        plt.tight_layout()  # Adjust layout to prevent overlapping labels
        plt.show()

    return resulting_components
    

def lst_vs(df):

    # Preprocess the data
    X = df.drop("LST",axis=1)
    y = df["LST"]
    
    # Get the right number of rows and columns for the figure
    num_vars = len(X.columns)
    rows = int(num_vars**0.5)
    cols = int(num_vars**0.5)
    if rows * cols < num_vars:
        cols += 1

    # Create subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))

    # Plot individual scatter plots for each independent variable
    # This works but i didn't take time to understand it, maybe something to modify or at least coment
    for i, column in enumerate(X.columns):
        ax = axes[i // cols, i % cols] if num_vars > 1 else axes
        ax.scatter(X[column], y, alpha=0.5)
        ax.set_title(f'{column} vs LST')
        ax.set_xlabel(column)
        ax.set_ylabel('LST')

    # Hide empty subplots
    for i in range(num_vars, rows * cols):
        ax = axes[i // cols, i % cols] if num_vars > 1 else axes
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def Knn_regressor(df):

    # Data preprocessing
    X = df.drop("LST",axis=1)
    y = df["LST"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

    # Create and train the model
    knn_model = KNeighborsRegressor(n_neighbors=11,weights="uniform", n_jobs=-1)
    knn_model.fit(X_train, y_train)

    # Make prediction
    basic_visualization(X_test=X_test, y_test=y_test,model=knn_model)


def rdf_regressor(df,parameters_list, force = False):


    X,y = create_X_y(df, parameters_list) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

    # # Load or create the model and save it
    # if FILENAME_RDF in os.listdir(MODELS_FOLDER) and not force:
    #     print("Loading the model...")
    #     rf_model = load(MODEL_PATH)
    # else:
    print("Training the model")
    rf_model = RandomForestRegressor(n_estimators=25, random_state=42,n_jobs=-1,verbose=1)
    rf_model.fit(X_train, y_train)
    
    # importance_vis(X_test=X_test, y_test=y_test, model=rf_model)
    basic_visualization(X_test=X_test, y_test=y_test, model = rf_model)
    # Save the model to a file
    # print("Saving the model...")
    # dump(rf_model, MODEL_PATH)


def adjusted_r2_calc(r2, X_test):
    n,k = X_test.shape
    print(k)
    print(n)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / ( n - k- 1)
    return adjusted_r2