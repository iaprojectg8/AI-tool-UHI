from utils.imports import *
from tools.preprocessing import *

# ------------------------------
# General statistics on data
# ------------------------------

def heat_map(df:pd.DataFrame,wandb_title):
    """
    Generates a heatmap of the correlation matrix of a dataframe and logs it to Weights and Biases).
    
    Args:
    df (pd.DataFrame): Dataframe for which the correlation matrix is computed.
    wandb_title (str): Title for the heatmap image to be logged in wandb.
    
    """
    # Compute the correlation matrix of the dataframe
    C_mat = df.corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(C_mat, cmap="coolwarm", square=True)
    
    # Log the heatmap to wandb
    wandb.log({wandb_title: wandb.Image(plt)})
    # plt.show()
    plt.clf()

def stat_on_data_with_zone(df,classif,classes_to_labels,wandb_title):
    """
    Generates and logs a histogram of Land Surface Temperature (LST) by zone (urban and rural) using Weights and Biases (wandb).

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    classes_to_labels (dict): Dictionary mapping classes to labels.
    wandb_title (str): Title for the wandb logging.

    """

    # Convert classes to labels in the DataFrame
    if classif:
        df = class_to_label(df, classes_to_labels)

    # Separate urban and rural data based on 'URB' column
    urban_data = df[df['URB'] == 1]['LST']
    rural_data = df[df['URB'] == 0]['LST']

    # Define histogram bins
    min = df["LST"].min()
    max = df["LST"].max()
    bins = range(int(min), int(max), 1) 

    plt.hist([urban_data, rural_data], bins=bins, color=['blue', 'green'], label=['Urban', 'Rural'], alpha=0.7)
    plt.xlabel('LST')
    plt.ylabel('Frequency')
    plt.title('Histogram of LST by Zone')
    plt.legend()

    wandb.log({wandb_title : wandb.Image(plt)})
    plt.clf()
    # plt.show()


# --------------------------------------
# Statistics on predictions and models
# --------------------------------------

def stat_on_prediction(pred, y_test, threshold, wandb_push:bool, title: str):
    """
    Generates statistical plots to compare predictions with ground truth values and logs them to Weights and Biases (wandb).
    
    Args:
    pred (np.ndarray): Predicted values.
    y_test (np.ndarray): Ground truth values.
    threshold (float): Threshold to consider a prediction as well-predicted.
    title (str): Title for the plot to be logged in wandb.
    
    """
    # Compute the min and max values for predictions and ground truth
    min_pred = pred.min()
    max_pred = pred.max()
    min_y_test = y_test.min()
    max_y_test = y_test.max()
    
    # Create ranges and bins for the histograms
    pred_range = range(int(min_pred), int(max_pred) + 1, 1)
    y_test_range = range(int(min_y_test), int(max_y_test) + 1, 1)
    bins = sorted(set(pred_range).union(set(y_test_range)))

    # Reshape predictions to match the shape of y_test
    pred = np.reshape(pred, y_test.shape)
    
    # Determine which predictions are within the threshold of the ground truth
    well_predicted = (abs(pred - y_test) <= threshold)
    well_predicted_array = pred[well_predicted]
    others = pred[~well_predicted]

    # Calculate the percentage of well-predicted values
    well_predicted_counts, _ = np.histogram(well_predicted_array, bins=bins)
    total_counts, _ = np.histogram(np.concatenate([well_predicted_array, others]), bins=bins)
    percentages = well_predicted_counts / total_counts
    percentages = [0 if (percentage < 1e-5 or np.isnan(percentage)) else round(percentage, 2) for percentage in percentages]

    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(title)
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Historgram for distribution comparison between ground truth and predicted values
    ax1.hist([pred, y_test], bins=bins, color=['blue', 'green'], label=["Prediction", "Ground Truth"], alpha=0.7)
    ax1.set_xlabel('Intervals')
    ax1.set_ylabel('Count')
    ax1.legend()

    # Calculate the overall percentage of well-predicted values
    well_predicted_all = sum(well_predicted_counts) / sum(total_counts)
    
    # Subplot for histogram of well-predicted and mispredicted values
    ax2.hist([well_predicted_array, others], bins=bins, color=['green', 'red'], label=[
        f"{well_predicted_all:.2f} of well predicted\nwith {threshold} precision", 
        f"{1 - well_predicted_all:.2f} Mispredicted"], alpha=0.7, width=0.7, stacked=True)
    ax2.set_xlabel('Intervals')
    ax2.set_ylabel('Count')
    ax2.legend()

    # Annotate histogram with percentages
    bin_centers = [bin + 0.3 for bin in bins]
    for i, percentage in enumerate(percentages):
        if percentage != 0: 
            ax2.annotate(percentage, xy=(bin_centers[i], total_counts[i]), 
                         xytext=(0, 5), textcoords='offset points', ha='center', color='black', fontsize=10)
        else: 
            ax2.annotate(percentage, xy=(bins[i], total_counts[i]), 
                         xytext=(0, 5), textcoords='offset points', ha='center', color='black', fontsize=10)

    
    
    plt.tight_layout()
    if wandb_push:
        wandb_title = title
        wandb.log({wandb_title: wandb.Image(plt)})
        plt.clf()
    else:
        plt.title(title)
        plt.show()


def comparison_to_csv(X_test, y_test, y_pred):
    y_pred = np.reshape(y_pred,(len(y_test),))
    print(sum(y_test>y_pred))
    scaler : StandardScaler = load(MODEL_SCALER_REG)
    X_test = scaler.inverse_transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['LON', 'LAT', 'LS1', 'LS2', 'LS3', 'LS4', 'LS5', 'LS6', 'OCCSOL', 'URB', 'ALT', 'EXP', 'PENTE', 'NATSOL', 'NATSOL2', 'HAUTA', 'CATHYD', 'ZONECL', 'ALB'])
    y_pred = pd.DataFrame(y_pred, columns=["LST_pred"])
    y_test = pd.DataFrame(y_test,columns=["LST"])
    pd.DataFrame(y_pred, columns=['predicted_temp'])
    df = pd.concat([y_test, y_pred, X_test],axis=1)
    df.to_csv("extend_area_lst_comparison.csv",index=False)


def visualization(classif,classes_to_temps,test_inputs,wandb_title, model): 
    """
    Visualizes the model's performance by predicting on the test set, calculating statistics, and generating plots.

    Parameters:
    model_type (str): The type of model (e.g., "classifier").
    model_path (str): Path to the saved model.
    classes_to_temps (dict): Dictionary mapping class labels to temperatures.
    inputs (tuple): A tuple containing datasets (train, validation, test splits).
    wandb_title (str): Title for the plots to be logged in wandb.
    model (optional): Loaded model object. If None, the model is loaded from the model_path.

    """
    # Unpack the data
    X_test,y_test = test_inputs

    # Make the prediction and the evaluation
    y_pred = model.predict(X_test.copy())
    # comparison_to_csv(X_test, y_test, y_pred)
    
    # Take the labels back
    if classif:
        y_pred = np.argmax(y_pred, axis = 1)
        y_pred = np.array([classes_to_temps[label] for label in y_pred])
        y_test = np.array([classes_to_temps[label] for label in y_test])
        

    stat_on_prediction(y_pred,y_test,threshold=0.5, wandb_push=1,title="Prediction vs Ground Truth")

    # Evaluation of the model
    n,k = X_test.shape
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / ( n - k- 1)
    print("\nFor Y pred:")
    print("R2:",r2,"MSE:",mse, "MAE:",mae)
    print("Adjusted R2:",adjusted_r2)
    
    # Reshaping the y_pred because it is (n,1), instead of (n,) for y_test, which is a problem to broadcast in the substraction operation later
    y_pred = np.reshape(y_pred,y_test.shape)
    # Residuals calculation
    residuals = abs(y_test - y_pred)

    # Intervals definition
    
    intervals = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0,2.0), (2.0,5.0),(5.0,10.0)]  # Liste des intervals des intervalles
    # intervals = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),(0.5, 0.6),(0.6, 0.7),(0.7, 0.8),(0.8, 0.9),(0.9, 1.0), (1.0,1.2), (1.2,1.5), (1.5,2.0), (2.0,5.0),(5.0,10.0)]  # nouvelle liste

    # Residuals interval calculation
    residuals_amount_per_interval = []
    total_residual_amount = len(residuals)

    # Amount of residuals per interval
    for i,interval in enumerate(intervals):
        if i==0:
            nb_residus = sum((residuals >= interval[0]) & (residuals <= interval[1]))
        else: 
            nb_residus = sum((residuals > interval[0]) & (residuals <= interval[1]))
        residuals_amount_per_interval.append(nb_residus)

    
    percentages = [nb / total_residual_amount * 100 for nb in residuals_amount_per_interval]
    
    # Create the figure to plot
    fig = plt.figure(figsize=(13, 6))
    
    # Plot 1
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    colors = cm.nipy_spectral(np.linspace(0.45, 0.95, len(intervals)))
    bars = ax1.bar(range(len(intervals)), residuals_amount_per_interval, color=colors, alpha=0.7)
    

    # Put the percentages on the bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{percentages[i]:.1f}%', ha='center', va='bottom')

    # Put the right label for each interval 
    labels = [f'[{interval[0]}, {interval[1]}]' for interval in intervals]
    ax1.set_xticks(range(len(intervals)), labels)
    ax1.set_xlabel('Residual interval (°C)')
    ax1.set_ylabel('Residual amount')
    ax1.set_title('Residual amounts in each interval')
    for label in ax1.get_xticklabels():
        label.set_rotation(45)

    # Plot 2 
    
    ax2.scatter(y_test, y_pred, label=f'R2 = {r2:.2f}       R2*= {adjusted_r2:.2f}\nMSE = {mse:.2f}    MAE = {mae:.2f}')
    # Make all the line to define the +/- intervals to have a better idea on the distribution
    for i in range(colors.shape[0]-1):
        ax2.plot( y_test,y_test+intervals[i][1], color=colors[i], label=f'+/- {intervals[i][1]}')  # Plotting the line of ground truth temp
        ax2.plot( y_test,y_test-intervals[i][1], color=colors[i])  # Plotting the line of ground truth temp
    ax2.set_xlabel('Ground Truth Temperatures')
    ax2.set_ylabel('Predicted Temperatures')
    ax2.set_title('Predicted vs Ground Truth temperature')
    ax2.legend(loc="upper left")

    plt.tight_layout()
    wandb.log({wandb_title : wandb.Image(plt)})
    # plt.show()

def basic_visualization(X_test,y_test,model): 
    """
    Visualizes the model's performance by predicting on the test set, calculating statistics, and generating plots.

    Parameters:
    model_type (str): The type of model (e.g., "classifier").
    model_path (str): Path to the saved model.
    classes_to_temps (dict): Dictionary mapping class labels to temperatures.
    inputs (tuple): A tuple containing datasets (train, validation, test splits).
    wandb_title (str): Title for the plots to be logged in wandb.
    model (optional): Loaded model object. If None, the model is loaded from the model_path.

    """

    # Make the prediction and the evaluation
    y_pred = model.predict(X_test.copy())
    # comparison_to_csv(X_test, y_test, y_pred)
        

    stat_on_prediction(y_pred,y_test,threshold=0.5,wandb_push=0,title="Prediction vs Ground Truth")

    # Evaluation of the model
    n,k = X_test.shape
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / ( n - k- 1)
    print("\nFor Y pred:")
    print("R2:",r2,"MSE:",mse, "MAE:",mae)
    print("Adjusted R2:",adjusted_r2)
    
    # Reshaping the y_pred because it is (n,1), instead of (n,) for y_test, which is a problem to broadcast in the substraction operation later
    y_pred = np.reshape(y_pred,y_test.shape)
    # Residuals calculation
    residuals = abs(y_test - y_pred)

    # Intervals definition
    
    intervals = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0,2.0), (2.0,5.0),(5.0,10.0)]  # Liste des intervals des intervalles
    # intervals = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),(0.5, 0.6),(0.6, 0.7),(0.7, 0.8),(0.8, 0.9),(0.9, 1.0), (1.0,1.2), (1.2,1.5), (1.5,2.0), (2.0,5.0),(5.0,10.0)]  # nouvelle liste

    # Residuals interval calculation
    residuals_amount_per_interval = []
    total_residual_amount = len(residuals)

    # Amount of residuals per interval
    for i,interval in enumerate(intervals):
        if i==0:
            nb_residus = sum((residuals >= interval[0]) & (residuals <= interval[1]))
        else: 
            nb_residus = sum((residuals > interval[0]) & (residuals <= interval[1]))
        residuals_amount_per_interval.append(nb_residus)

    
    percentages = [nb / total_residual_amount * 100 for nb in residuals_amount_per_interval]
    
    # Create the figure to plot
    fig = plt.figure(figsize=(13, 6))
    
    # Plot 1
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    colors = cm.nipy_spectral(np.linspace(0.45, 0.95, len(intervals)))
    bars = ax1.bar(range(len(intervals)), residuals_amount_per_interval, color=colors, alpha=0.7)
    

    # Put the percentages on the bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{percentages[i]:.1f}%', ha='center', va='bottom')

    # Put the right label for each interval 
    labels = [f'[{interval[0]}, {interval[1]}]' for interval in intervals]
    ax1.set_xticks(range(len(intervals)), labels)
    ax1.set_xlabel('Residual interval (°C)')
    ax1.set_ylabel('Residual amount')
    ax1.set_title('Residual amounts in each interval')
    for label in ax1.get_xticklabels():
        label.set_rotation(45)

    # Plot 2 
    
    ax2.scatter(y_test, y_pred, label=f'R2 = {r2:.2f}       R2*= {adjusted_r2:.2f}\nMSE = {mse:.2f}    MAE = {mae:.2f}')
    # Make all the line to define the +/- intervals to have a better idea on the distribution
    for i in range(colors.shape[0]-1):
        ax2.plot( y_test,y_test+intervals[i][1], color=colors[i], label=f'+/- {intervals[i][1]}')  # Plotting the line of ground truth temp
        ax2.plot( y_test,y_test-intervals[i][1], color=colors[i])  # Plotting the line of ground truth temp
    ax2.set_xlabel('Ground Truth Temperatures')
    ax2.set_ylabel('Predicted Temperatures')
    ax2.set_title('Predicted vs Ground Truth temperature')
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

# ----------------------------------------------
# Metric progression for neural network models
# ----------------------------------------------

def loss_and_metrics_vis(history,wandb_title):
    """
    Visualizes the training history of a model, including loss, MAE/accuracy, and learning rate progression.

    Parameters:
    history (dict): Dictionary containing the training history (loss, mae/accuracy, learning rate).
    wandb_title (str): Title for the wandb logging.

    Returns:
    None
    """
       

    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Training and validation loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss Progression During Training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Check if MAE is present in the history and plot it; otherwise, plot accuracy
    if "mae"in history.keys():
        ax2.plot(history['mae'], label='Training MAE')
        ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE Progression During Training')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MAE')
        ax2.legend()

    else : 
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy Progression During Training')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

    # Learning rate progression
    ax3.plot(history['learning_rate'], label='Learning rate')   
    ax3.set_title('Learning Rate Progression During Training')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning rate')
    ax3.set_yscale('log')
    ax3.legend()

    
    
    
    plt.tight_layout()
    wandb.log({wandb_title : wandb.Image(plt)})
    plt.clf()
    # plt.show()



# --------------------------------------------------
# Importance visualisation for random forest models
# --------------------------------------------------

def importance_vis(X_test, y_test,model):
    """
    Visualizes the feature importance of a random forest regression trained model using a bar chart and a pie chart.

    Args:
    model_path (str): Path to the saved model.
    inputs (tuple): A tuple containing datasets (train, validation, test splits).
    model (optional): Loaded model object. If None, the model is loaded from the model_path.

    """


    print("Start calculating importance")

    feature_names = X_test.columns.tolist()  
    cmap = cm.nipy_spectral(np.linspace(0.1,0.9,len(feature_names)))
    color_dict = dict(zip(feature_names, cmap))
    
    
    # importance_features = model.feature_importances_
    # Change of the importance here
    result_dict = permutation_importance(model, X_test,y_test)
    importance_features = list(result_dict["importances_mean"])
    importance_sum = sum(importance_features)
    importance_rounded = [round(importance/importance_sum, 3) for importance in importance_features]
    # Create a dictionary associating feature names with their rounded importance values
    feature_importance_dict = dict(zip(feature_names, importance_rounded))

    # Alternatively, you can plot a pie chart
    threshold = 0.01

    # Filter out features with importance values below the threshold
    filtered_importance_dict = {feature: importance for feature, importance in feature_importance_dict.items() if importance >= threshold}
    colors = [color_dict[column] for column in filtered_importance_dict.keys()]
    print(filtered_importance_dict)

    fig = plt.figure(figsize=(13, 6))
    
    # Plot 1
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Generate colors from the 'tab10' colormap
    bars = ax1.bar(filtered_importance_dict.keys(), filtered_importance_dict.values(), color=colors)

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Importance')
    ax1.set_title('Feature Importance')
    # Peut-être ajouter une rotation pour les labels de l'a


    # Plot 2
    
    ax2.pie(filtered_importance_dict.values(), labels=filtered_importance_dict.keys(), autopct='%1.1f%%', startangle=140,colors=colors)
    ax2.set_title('Feature Importance')

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()
    plt.show()

def neural_net_importance(model, X_train, X_test):
   
    # Utiliser SHAP pour expliquer les prédictions
    feature_names = ["LON","LAT","LS1","LS2","LS3","LS4","LS5","LS6","OCCSOL","URB","ALT","EXP","PENTE","NATSOL","NATSOL2","HAUTA","CATHYD","ZONECL","ALB"]
    explainer = shap.Explainer(model, X_train,feature_names=feature_names)
    out_file = "explainer_output"  # File path where the explainer will be saved

    with open(out_file, 'wb') as f:
        explainer.save(f)
    
    print("Values will be calculated from here")
    
    shap_values = explainer(X_test)
    with open('shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values, f)

    # Afficher un graphique de résumé
    shap.summary_plot(shap_values, X_test,feature_names=feature_names)

   