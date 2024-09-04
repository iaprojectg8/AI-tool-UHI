from tools.preprocessing import *
from tools.tried_models import *
from utils.imports import *
from tools.visualization import *
from tools.nn_models import * 




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

def main():
    params_to_take = ["LAT", "LON"]
    params_to_drop = []
    params_list = [params_to_take, params_to_drop]
    df = init_df_all_variable()
    rdf_regressor(df,parameters_list=params_list)
    # print(X_train.shape)
    # model = models.Sequential()

    # # Adding the first group of layers with a dummy layer
    # model.add(layers.Input(shape=(X_train.shape[1],)))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(visualkeras.SpacingDummyLayer(spacing=40))

    # # Adding the second group of layers with a dummy layer
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(0.3))
    # model.add(visualkeras.SpacingDummyLayer(spacing=40))

    # # Adding the third group of layers with a dummy layer
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.25))
    # model.add(visualkeras.SpacingDummyLayer(spacing=40))

    # # Adding the fourth group of layers with a dummy layer
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(visualkeras.SpacingDummyLayer(spacing=40))

    # # Adding the output layer
    # model.add(layers.Dense(1))  # Output layer for continuous prediction

    # # Now you can generate and show the visualization
    # vis_net = visualkeras.layered_view(
    #     model, min_xy=20, max_xy=4000, min_z=20, max_z=4000,
    #     padding=100, scale_xy=5, scale_z=20, spacing=20,
    #     one_dim_orientation='x', legend=True, draw_funnel=True
    # )

    # vis_net.show()    

    
if __name__ == "__main__":
    main()