from utils.imports import *


# ------------------------------
# Basic file processes
# ------------------------------

def init_df_all_variable():
    """
    Initializes a DataFrame from a CSV file.

    This function loads data from a specified CSV file
    and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    data_csv_file = os.path.join(CSV_INPUT_FOLDER, CSV_INPUT_FILE)
    df = pd.read_csv(data_csv_file)
    return df

def init_df_basic_file():
    """
    Initializes a DataFrame from a CSV file and replace certain values.

    This function loads data from a CSV file, replaces values in the 'SECTOR' column
    with numeric values, and returns the modified DataFrame along with a result file name.

    Returns:
        tuple: A tuple containing the modified DataFrame and the result file name.
    """
    pd.set_option('future.no_silent_downcasting', True)
    data_csv_file = "Cleaned_urb_temp.csv"
    result_file = "Results.csv"
    df = pd.read_csv(data_csv_file)
    df['SECTOR'] = df['SECTOR'].replace({'Urbain': 1, 'Rural': 2, 'Eau': 3})
    return df, result_file


def concatenate_data(filename_list: list = []):
    """
    Concatenates multiple CSV files into a single DataFrame.

    This function loads data from multiple specified CSV files
    and concatenates them into a single pandas DataFrame.

    Args:
        filename_list (list): List of CSV file names to concatenate.

    Returns:
        pd.DataFrame: DataFrame containing the concatenated data from the CSV files.
    """
    df = pd.DataFrame()
    for filename in filename_list:
        data_csv_file = os.path.join(CSV_INPUT_FOLDER, filename)
        df_temp = pd.read_csv(data_csv_file)
        df = pd.concat([df, df_temp], ignore_index=True)
    return df

# ------------------------------
# DBF files process
# ------------------------------

def convert_dbf_to_csv(input_file_path, output_file_path):
    """
    Converts a DBF file to a CSV file.

    This function loads a specified DBF file and converts it to a CSV file.

    Args:
        file_path (str): Path to the DBF file to convert.

    Returns:
        None
    """
    dbf = Dbf5(input_file_path)
    dbf.to_csv(output_file_path)

def dbf_to_df(input_file_path):
    """
    Converts a DBF file to a CSV file.

    This function loads a specified DBF file and converts it to a CSV file.

    Args:
        file_path (str): Path to the DBF file to convert.

    Returns:
        None
    """
    dbf = Dbf5(input_file_path)
    df = dbf.to_dataframe()
    return df



def convert_all_dbf_in_a_folder():

    """
    Converts all the DBF files into CSV file in a specific folder
    
    It writes in CSV format all the DBF file present in the folder and put the file in cleaned output file

    Args:
        file_path (str): Path to the DBF file to convert.

    Returns:
        None
    """
    dbf_folder = DBF_FOLDER
    ouput_folder =  CSV_INPUT_FOLDER
    print("Processing the cleaning and convertion of DBF file into CSV")
    for filename in  os.listdir(dbf_folder):
        print(f"Cleaning dbf file {filename}")
        input_file_path = os.path.join(dbf_folder,filename)
        df = dbf_to_df(input_file_path=input_file_path)
        df = drop_LS7(df)
        df = clean_data(df)
        output_filename = filename.split(".")[0] + ".csv"
        output_file_path = os.path.join(ouput_folder,output_filename)
        df.to_csv(output_file_path,index=False)


# ------------------------------
# Cleaning the data
# ------------------------------

def separate_on_urb(df:pd.DataFrame, parameters_list, urb_frac=0.1):
    """
    Separates data into training and test sets based on urban or rural zones.

    This function separates data into two sets: urban and rural.
    A percentage of urban data is used for the training set,
    the rest of the urban data and all rural data are used for the test set.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        parameters_list (list): List of columns to use for X.
        urb_frac (float): Fraction of urban data to include in the training set. Default is 0.1.

    Returns:
        tuple: A tuple containing the X and y matrices for the training and test sets.
    """
    df_rur = df[df["URB"] == 0]
    df_urb = df[df["URB"] == 1]

    urb_rows = df_urb.sample(frac=urb_frac, random_state=42)
    df_test = df_urb.drop(urb_rows.index)
    df_train = pd.concat([df_rur, urb_rows], ignore_index=True)

    X_train, y_train = create_X_y(df_train, parameters_list=parameters_list)
    X_test, y_test = create_X_y(df_test, parameters_list=parameters_list)

    return X_train, X_test, y_train, y_test


def clean_data(df:pd.DataFrame):
    """
    Cleans the DataFrame by removing NaN values and filtering out rows where the 'LST' column is zero.
    
    Parameters:
    df (DataFrame): The input DataFrame to be cleaned.
    
    Returns:
    DataFrame: The cleaned DataFrame.
    """
    df = df.dropna()
   # temp_mean = np.mean(df["LST"])
    # temp_std = np.std(df["LST"])
    # threshold = 3 * temp_std
    # df = df[abs((df["LST"] - temp_mean)) <= threshold]
    df  = df[df["LST"]!=0]
    
    
    return df

def drop_LS7(df:pd.DataFrame):
    if "LS7" in df.columns:
        df = df.drop("LS7",axis=1)
    return df



def take_right_parameters(df,params_to_take=[],params_to_drop=[]):
    """
    Selects or drops specified parameters (columns) from the DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    params_to_take (list): List of parameters to include in the DataFrame.
    params_to_drop (list): List of parameters to drop from the DataFrame.
    
    Returns:
    DataFrame: The DataFrame with the specified parameters taken or dropped.
    """
    if params_to_take!=[]:
        if "LST" not in params_to_take:
            params_to_take.append("LST")
        df = df[params_to_take]
            
    elif params_to_drop!=[]:
        print(params_to_drop)
        print(df)
        df = df.drop(params_to_drop,axis=1)
        print(df)

    return df

def create_X_y(df,parameters_list):
    """Prepares feature matrix X and target vector y from the DataFrame by selecting the right parameters
    and cleaning the data.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    parameters_list (list): List of parameters to include in the feature matrix.
    
    Returns:
    tuple: Feature matrix X and target vector y.
    """
    # y will always be the same because this is the variable we want to predict
    
    df = take_right_parameters(df,*parameters_list)
    y = df["LST"] 
    y = np.array(y,dtype=np.float16)     
    X = df.drop('LST', axis=1)  # If you want to take all the variables except the dependant one do this, even if it is hard to understand

    return X,y

# ----------------------------------------------------
# Loading, cleaning, splitting and displaying the data
# ----------------------------------------------------

def display_preprocess(train_data, test_data,val_data):
    """
    Displays the size and distribution of the training, testing, and validation datasets.

    Args:
    train_data (DataFrame): Training dataset.
    test_data (DataFrame): Testing dataset.
    val_data (DataFrame): Validation dataset.
    """
    print(f"Train: {train_data.shape[0]} data | {train_data[train_data['URB']==1].shape[0]} urban | {train_data[train_data['URB']==0].shape[0]} rural")
    print(f"Test:  {test_data.shape[0]} data | {test_data[test_data['URB']==1].shape[0]} urban | {test_data[test_data['URB']==0].shape[0]} rural")
    print(f"Val:  {val_data.shape[0]} data | {val_data[val_data['URB']==1].shape[0]} urban | {val_data[val_data['URB']==0].shape[0]} rural")
    print("\n")


def create_empty_dataframe(city_list):
    """
    Create empty dataframes for training, validation, and testing datasets
    with the same columns as the cleaned dataset of the first city in the list.

    Args:
        city_list (list): List of city names.

    Returns:
        df_train (DataFrame): Empty DataFrame for training data.
        df_test (DataFrame): Empty DataFrame for testing data.
        df_val (DataFrame): Empty DataFrame for validation data.
    """
    example = os.path.join(CSV_INPUT_FOLDER,city_list[0])
    df =  pd.read_csv(example)
    df = clean_data(df)
    header = df.columns.tolist()
    df_train = pd.DataFrame(columns=header)
    df_test = pd.DataFrame(columns=header)
    df_val = pd.DataFrame(columns=header)
    return df_train, df_test, df_val

def open_and_clean_df(city):
    """
    Open and clean the CSV file for a given city.

    Args:
        city (str): Name of the city.

    Returns:
        df_cleaned (DataFrame): Cleaned DataFrame for the city.
    """
    data_csv_file = os.path.join(CSV_INPUT_FOLDER,city)
    df_file = pd.read_csv(data_csv_file) # The na_values is for the EXP columns that contains those weird values
    df_cleaned = clean_data(df_file)

    return df_cleaned

def sampling_data(df_cleaned, amount_urban_per_city, amount_rural_per_city):
    """
    Sample a specified number of urban and rural data points from the cleaned DataFrame.

    Args:
        df_cleaned (DataFrame): Cleaned DataFrame of the city.
        amount_urban_per_city (int): Number of urban data points to sample.
        amount_rural_per_city (int): Number of rural data points to sample.

    Returns:
        df_urb (DataFrame): Sampled urban DataFrame.
        df_rur (DataFrame): Sampled rural DataFrame.
    """
    df_urb = df_cleaned[df_cleaned["URB"]==1]
    df_rur = df_cleaned[df_cleaned["URB"]==0]
    n_urb = df_urb.shape[0]
    n_rur = df_rur.shape[0]
    if n_urb > amount_urban_per_city:
        df_urb = df_urb.sample(n=amount_urban_per_city)
    if n_rur > amount_rural_per_city:
        df_rur = df_rur.sample(n=amount_rural_per_city)
    return df_urb, df_rur


def process_train_list_city(df_train, df_val, df_urb, df_city, frac_urb_train):
    """
    Process urban data for a city, splitting it into training and validation datasets.

    Args:
        df_train (DataFrame): DataFrame for training data.
        df_val (DataFrame): DataFrame for validation data.
        df_urb (DataFrame): Urban DataFrame for the city.
        df_city (DataFrame): Full DataFrame for the city.
        frac_urb_train (float): Fraction of urban data to use for training.

    Returns:
        df_train (DataFrame): Updated training DataFrame.
        df_val (DataFrame): Updated validation DataFrame.
    """
    df_temp_val = df_urb.sample(frac = (1 - frac_urb_train),random_state=42)
    df_temp_train = df_city.drop(df_temp_val.index)
    df_temp_test = pd.DataFrame(columns=df_temp_train.columns)

    df_train = pd.concat([df_train,df_temp_train], ignore_index=True)
    df_val = pd.concat([df_val,df_temp_val])

    display_preprocess(df_temp_train,df_temp_test, df_temp_val)

    return df_train, df_val


def process_both_list_city(df_train, df_test, df_val, df_urb, df_rur, frac_urb_train):
    """
    Process urban and rural data for a city, splitting it into training, validation, and testing datasets.

    Args:
        df_train (DataFrame): DataFrame for training data.
        df_test (DataFrame): DataFrame for testing data.
        df_val (DataFrame): DataFrame for validation data.
        df_urb (DataFrame): Urban DataFrame for the city.
        df_rur (DataFrame): Rural DataFrame for the city.
        frac_urb_train (float): Fraction of urban data to use for training.

    Returns:
        df_train (DataFrame): Updated training DataFrame.
        df_test (DataFrame): Updated testing DataFrame.
        df_val (DataFrame): Updated validation DataFrame.
    """
    df_urb_train = df_urb.sample(frac=frac_urb_train,random_state = 42)
    df_temp_test = df_urb.drop(df_urb_train.index)
    df_temp_val = df_urb_train.sample(frac = 0.15,random_state=42) 

    
    df_urb_train = df_urb_train.drop(df_temp_val.index)
    df_temp_train = pd.concat([df_rur,df_urb_train])


    # Put the right data in test and train set
    df_val = pd.concat([df_val,df_temp_val])
    df_test = pd.concat([df_test,df_temp_test],ignore_index=True)
    df_train = pd.concat([df_train,df_temp_train])

    display_preprocess(df_temp_train,df_temp_test,df_temp_val)

    return df_train, df_test, df_val

def process_test_list(city_name, df_urb):
    """
    Process urban and rural data for a city, adding it to the testing dataset.

    Args:
        city_name (str): Name of the city.
        df_urb (DataFrame): Urban DataFrame for the city.
        df_rur (DataFrame): Rural DataFrame for the city.

    Returns:
        df_test (DataFrame): Updated testing DataFrame.
    """
    print(f"{city_name} will serve for testing only:")
    df_test = pd.concat([df_test,df_urb], ignore_index= True)
    # Refaire un display ici si jamais pour le test, car le display pour le train ne va pas être cohérent 
    # compte tenu du peu de valeurs que l'on a

    return df_test


def process_last_city(df_train, df_test, df_val):
    print("All the data dataset are being shuffled")
    df_train = df_train.sample(frac=1)
    df_test = df_test.sample(frac=1)
    df_val = df_val.sample(frac=1)
    print("Finally here are the distribution of the data among the train , test, and val set")
    display_preprocess(df_train,df_test,df_val)

    return df_train, df_test, df_val

def load_and_separate_data_for_train(city_list, classif, step, n_rur, n_urb , train_city_list=[],frac_urb_train=0.8):
    """
    Load and separate data into training, testing, and validation sets based on the provided city list and parameters.

    Args:
    city_list (list): List of city filenames.
    step (float): Step size for discretization.
    n_rur (int): Maximum number of rural data per city.
    n_urb (int): Maximum number of urban data per city.
    train_city_list (list): List of cities to be used for training.
    test_city_list (list): List of cities to be used for testing.
    frac_urb_train (float): Fraction of urban data to be used for training.

    Returns:
    tuple: DataFrames for training, testing, and validation sets.
    """

    df_train, df_test, df_val = create_empty_dataframe(city_list)

    amount_rural_per_city = n_rur
    amount_urban_per_city = n_urb
    print(f"For each city, the maximum amount of line in dataset will be:\n - {amount_rural_per_city} for rural\n - {amount_urban_per_city} for urban")

    for city in tqdm(city_list):

        city_name = city.split(".")[0]
        df_cleaned = open_and_clean_df(city)
        
        if classif:
            df_cleaned = discretize_LST(df_cleaned,step)
        
        # Sample the data if necessary
        df_urb, df_rur = sampling_data(df_cleaned, amount_urban_per_city=amount_urban_per_city, amount_rural_per_city=amount_rural_per_city)
        
        # Display information for the user
        print(f"\n\nFor {city_name} there will be {df_urb.shape[0]} of urban data and {df_rur.shape[0]} of rural data")

        # Concatenates the row we are taking and then shuffles it with the sample function
        df_city = pd.concat([df_rur, df_urb],ignore_index=False)
        df_city = df_city.sample(frac=1)

        if city in train_city_list:
            print(f"{city_name} will serve for training only:")
            df_train, df_val = process_train_list_city(df_train, df_val, df_urb, df_city, frac_urb_train)

        else:
            print(f"{city_name} will serve for training and testing, with {frac_urb_train} of urban used for train ")
            df_train, df_test, df_val = process_both_list_city(df_train, df_test, df_val, df_urb, df_rur, frac_urb_train)
    

        # At the end of the dataset loading, shuffle all the dataset that have been made 
        if city == city_list[-1]:
            df_train, df_test, df_val = process_last_city(df_train, df_test, df_val)

    return df_train,df_test,df_val

# ------------------------------
# Equalizing the data
# ------------------------------

def data_equalizer(df, threshold):
    """
    Balance the data distribution in the DataFrame based on the values in the "LST" column.

    Args:
    df (DataFrame): Input DataFrame.
    threshold (int): Threshold value to determine the maximum number of data points per interval.

    Returns:
    DataFrame: Balanced DataFrame with equalized data distribution.
    """

    # Define histogram bins based on the minimum and maximum values of the "LST" column
    min_value = df["LST"].min()
    max_value = df["LST"].max()
    bins = range(int(min_value), int(max_value) + 1, 1)  # +1 to include the max value in the range
    
    # Create an empty DataFrame to store the balanced data
    balanced_df = pd.DataFrame(columns=df.columns)
    
    # Iterate over each bin interval
    for i in range(len(bins) - 1):
        # Get data in the current interval
        interval_data = df[(df["LST"] >= bins[i]) & (df["LST"] < bins[i + 1])]
        
        # Check if the amount of data exceeds the threshold
        if len(interval_data) > threshold:
            # Randomly sample data from the interval to meet the threshold
            interval_data = interval_data.sample(n=threshold, random_state=42)
        
        # Append the (sampled) interval data to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, interval_data], ignore_index=True)
    
    return balanced_df

# ------------------------------
# Discretizing the data
# ------------------------------

def discretize_LST(df, step):
    """
    Discretizes the "LST" column in the DataFrame by rounding each value to the nearest multiple of `step`.

    Args:
    df (DataFrame): Input DataFrame.
    step (float): Step size for discretization.

    Returns:
    DataFrame: DataFrame with discretized "LST" column.
    """

    df["LST"] = df["LST"].apply(lambda x : round_to_the_step(x,step))
    return df

def round_to_the_step(x,step):
    """
    Rounds the input value `x` to the nearest multiple of `step`.

    Args:
    x (float): Input value.
    step (float): Step size for rounding.

    Returns:
    float: Rounded value.
    """
    return round(x/step)*step

# -----------------------------------------
# Switching classes to labels in both ways
# -----------------------------------------

def class_label_dicts(df):
    """
    Creates dictionaries to map between class labels and their corresponding numerical representations.

    Args:
    df (DataFrame): Input DataFrame.

    Returns:
    dict: Dictionary mapping labels to classes.
    dict: Dictionary mapping classes to labels.
    """
    labels = df["LST"]
    unique_labels = pd.unique(labels)
    unique_labels_sorted = np.sort(unique_labels)
    

    classes_to_labels = dict()
    labels_to_classes = dict()

    for i, label in enumerate(unique_labels_sorted):
        label = round(label,2)
        classes_to_labels[i] = label
        labels_to_classes[label] = i

   
    # classes = [labels_to_classes[round(label,2)] for label in labels]
    # df["LST"] = classes

    return labels_to_classes, classes_to_labels

def label_to_class(df,labels_to_classes):
    """
    Converts class labels in the "LST" column of the DataFrame to their corresponding numerical representations.

    Args:
    df (DataFrame): Input DataFrame.
    labels_to_classes (dict): Dictionary mapping labels to classes.

    Returns:
    DataFrame: DataFrame with class labels converted to numerical representations.
    """
    labels = df["LST"]
    classes = [labels_to_classes[round(label,2)] for label in labels]
    df["LST"] = classes
    return df

def class_to_label(df,classes_to_labels):
    """
    Converts numerical class representations in the "LST" column of the DataFrame back to their corresponding labels.

    Args:
    df (DataFrame): Input DataFrame.
    classes_to_labels (dict): Dictionary mapping classes to labels.

    Returns:
    DataFrame: DataFrame with numerical class representations converted to labels.
    """
    classes = df["LST"]
    labels = [classes_to_labels[clas] for clas in classes]
    df["LST"] = labels
    return df

# ------------------------------
# Dropped parameters formating
# ------------------------------

def create_str_for_drop_parameters(params_to_drop):
    """
    Creates a string representation of parameters to be dropped from a DataFrame. This is very useful
    to display the name of a run in wandb specifying the variables not taken

    Args:
    params_to_drop (list): List of parameters to be dropped.

    Returns:
    str: String representation of dropped parameters.
    """
    str_drop_params = ""  # Just in case the list of parameters to drop is empty
    if params_to_drop != []:

        drop_params = [word.lower() for word in params_to_drop]
        drop_params.insert(0,"_without")
        str_drop_params= "_".join(drop_params)
    return str_drop_params

def create_city_str(city_files):
    """
    Creates a string representation of cities to train. This is very useful
    to display the name of a run in wandb specifying the cities taken

    Args:
    city_files (list): List of cities to take.

    Returns:
    str: String representation of cities to take.
    """

    city_list = [city_name.split(".")[0].lower() for city_name in city_files]
    str_city= "_".join(city_list)
    return str_city

# ------------------------------
# Redundant part of code
# ------------------------------

def scale_data(X_train:pd.DataFrame, X_test:pd.DataFrame, X_val:pd.DataFrame,scaler:StandardScaler,classif):
    """
    Scale data using StandardScaler

    Args:
    X_train (array-like): Training data to be scaled.
    X_test (array-like): Testing data to be scaled.
    X_val (array-like): Validation data to be scaled.

    Returns:
    tuple: Scaled versions of X_train, X_test, and X_val
    """
    if not X_train.empty:
        X_train = scaler.fit_transform(X_train)
        if classif:
            dump(scaler, MODEL_SCALER_PATH_CLASS)
        else:
            dump(scaler,MODEL_SCALER_PATH_REG)
    if not X_test.empty:
        X_test = scaler.transform(X_test)
    if not X_val.empty:
        X_val = scaler.transform(X_val)
    return X_train, X_test, X_val

def create_X_y_for_all(df_train,df_test,df_val,params_list):
    """
    Create feature (X) and target (y) datasets for training, testing, and validation.

    Args:
    df_train (pd.DataFrame): DataFrame containing the training data.
    df_test (pd.DataFrame): DataFrame containing the testing data.
    df_val (pd.DataFrame): DataFrame containing the validation data.
    params_list (list): List of parameters/columns to be used for creating the feature sets.

    Returns:
    tuple: Tuple containing the feature and target sets for training, testing, and validation.
    """
    X_train,y_train = create_X_y(df_train,parameters_list=params_list)
    X_test, y_test = create_X_y(df_test,parameters_list=params_list)
    X_val, y_val = create_X_y(df_val,parameters_list=params_list)
    return X_train, y_train, X_test, y_test, X_val, y_val

def df_labels_to_class_for_all(df_train, df_test, df_val, temps_to_classes):
    """
    Convert labels to classes for training, testing, and validation DataFrames.

    Args:
    df_train (pd.DataFrame): DataFrame containing the training data.
    df_test (pd.DataFrame): DataFrame containing the testing data.
    df_val (pd.DataFrame): DataFrame containing the validation data.
    temps_to_classes (dict): Dictionary mapping temperature values to class labels.

    Returns:
    tuple: Tuple containing the modified DataFrames for training, testing, and validation.
    """
    df_train= label_to_class(df_train,temps_to_classes)
    df_test= label_to_class(df_test,temps_to_classes)
    df_val = label_to_class(df_val,temps_to_classes)
    return df_train, df_test, df_val


# ------------------------------
# Redundant part of code
# ------------------------------

def create_df_test(city_list,classif, step):

    for city in tqdm(city_list):

        data_csv_file = os.path.join(CSV_INPUT_FOLDER,city)
        df = pd.read_csv(data_csv_file) # The na_values is for the EXP columns that contains those weird values
        df = clean_data(df)
        if classif:
            df = discretize_LST(df,step)
        df_test = df [df["URB"]==1]
        
    return df_test, df

def preoprocess_test_data(city_list):

    for city in tqdm(city_list):

        data_csv_file = os.path.join(CSV_INPUT_FOLDER,city)
        df = pd.read_csv(data_csv_file) # The na_values is for the EXP columns that contains those weird values
        df = df[df["layer"]=="Calculé"]
        df["URB"]=1
        df = df.drop(["layer", "path"],axis=1)
    
    return df

def save_config(config, classif):
    # Step 2: Define the filename
    if classif:
        file_path = MODEL_CONFIG_CLASS

    else:
        file_path = MODEL_CONFIG_REG

    # Step 3: Save the configuration dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)  # `indent=4` makes the file readable

def load_config(classif):

    if classif:
        file_path = MODEL_CONFIG_CLASS

    else:
        file_path = MODEL_CONFIG_REG

    with open(file_path, 'r') as f:
        config = json.load(f)  # `indent=4` makes the file readable

    return config

def save_history(history,file):
    with open(file, "w") as f:
        json.dump(history,f, indent=4)