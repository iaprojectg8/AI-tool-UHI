from tools.preprocessing import *
from tools.tried_models import *
from utils.imports import *
from tools.visualization import *
from tools.nn_models import * 



def main():
    params_to_take = ["LAT", "LON"]
    params_to_drop = []
    params_list = [params_to_take, params_to_drop]
    df = init_df_all_variable()
    rdf_regressor(df,parameters_list=params_list)
 

    
if __name__ == "__main__":
    main()