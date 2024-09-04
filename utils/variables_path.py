CSV_INPUT_FOLDER = "csv_input_data/"
CSV_ALL_COMB_FOLDER="all_variable_combination_regression"
CSV_INPUT_FILE = "Bamako.csv"
# CSV_INPUT_FILE = "LST_Points_File.csv"
CSV_ALL_COMB_RESULT_FOLDER = "Results_all_variable_plus.csv"
DBF_FOLDER = "dbf_files/"
MODELS_FOLDER="models/"
FILENAME_RDF = "bamako_lat_lon_03.joblib"
FILENAME_NN = 'temperature_prediction_model_yaounde_eq.keras'
MODEL_NN_PATH = MODELS_FOLDER + FILENAME_NN
MODEL_PATH = MODELS_FOLDER + FILENAME_RDF
CLEANED_DIR = "new_cleaned_files"
DATA_DIR = "new_input_file"


# Model path variables
BEST_MODEL_CLASS = "best_model_class.keras"
BEST_MODEL_PATH_CLASS = MODELS_FOLDER + BEST_MODEL_CLASS
BEST_MODEL_REG = "best_model_reg.keras"
BEST_MODEL_PATH_REG = MODELS_FOLDER + BEST_MODEL_REG

# Scaler path
MODEL_SCALER_CLASS = "scaler_class.joblib"
MODEL_SCALER_PATH_CLASS = MODELS_FOLDER + MODEL_SCALER_CLASS
MODEL_SCALER_REG = "scaler_reg.joblib"
MODEL_SCALER_PATH_REG = MODELS_FOLDER + MODEL_SCALER_REG


# history path
HISTORY_CLASS = "history_class.json"
MODEL_HISTORY_CLASS = MODELS_FOLDER + HISTORY_CLASS
HISTORY_REG = "history_reg.json"
MODEL_HISTORY_REG = MODELS_FOLDER + HISTORY_REG

# config path
CONFIG_CLASS = "config_class.json"
MODEL_CONFIG_CLASS = MODELS_FOLDER + CONFIG_CLASS
CONFIG_REG = "config_reg.json"
MODEL_CONFIG_REG = MODELS_FOLDER + CONFIG_REG

# MODEL_PATH = "rdf_reg_addis_abeba.joblib"

vars = ["LST","LON","LAT","LS1","LS2","LS3","LS4","LS5","LS6","OCCSOL","URB","ALT","EXP","PENTE","NATSOL","NATSOL2","HAUTA","CATHYD","ZONECL","ALB"]