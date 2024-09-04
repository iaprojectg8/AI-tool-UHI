from utils.imports import *
from utils.variables_path import *
from tools.preprocessing import *
from tools.visualization import *


class BaseModel(tf.keras.Model):
    """
    Base model class for building and training neural network models.

    This class is the base of all denses model that will be made. The architecture and the hyper parameters will be changeable
    in the children classes

    Attributes:
        input_shape (tuple): Shape of the input data.
        config (dict): Configuration dictionary with model parameters.
        model (tf.keras.Model): Keras model built using the provided configuration.
    """

    def __init__(self, input_shape=None, config=None):
        """
        Initializes the BaseModel.

        Args:
            input_shape (tuple): Shape of the input data.
            config (dict): Configuration dictionary with model parameters.
        """
        super(BaseModel, self).__init__()
        self.input_shape = input_shape
        self.config = config

                    

    def build_model(self):
        """
        Builds the neural network model.

        This method creates a sequential model and adds hidden layers based on the given configuration.

        Returns:
            tf.keras.Model: Return base keras model on which layers can be added
        """
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        
        hidden_layers = [value for key, value in self.config.items() if key.startswith('dense')]

        for units in hidden_layers:
            self.add_dense_block(model, units)

        self.model = model
        return model
    
    def loads_model(self):
        """
        Loads a trained model

        Returns:
            tf.keras.Model: Return a trained model 
        """
        model = models.load_model(self.model_path)
        self.model = model
        print("NN model loaded")

        return model
    
    def upload_model_archi(self, model):
        """
        Uploads the model architecture visualization to Weights & Biases (Wandb).

        First a layered_view is processed by the visualkeras module and then the graph is sent to wandb

        Args:
            model (tf.keras.Model): Keras model to visualize.
        """
        vis_net = visualkeras.layered_view(model, min_xy=20, max_xy=4000, min_z=20, max_z=4000, padding=100, scale_xy=5, scale_z=20, spacing=20, one_dim_orientation='x', legend=True, draw_funnel=True)
        wandb.log({"Network architecture": wandb.Image(vis_net)})

    def add_dense_block(self, model, units):
        """
        Adds a dense block to the model.

        A dense block consists of a Dense layer, Batch Normalization, ELU activation, and Dropout.

        Args:
            model (tf.keras.Model): Model to which the dense block will be added.
            units (int): Number of units in the Dense layer.
        """
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.ELU(alpha=self.config.alpha_elu))
        model.add(layers.Dropout(self.config.dropout))
        model.add(visualkeras.SpacingDummyLayer(spacing=40))
    
    
    def get_callbacks(self):
        """
        Gets the list of callbacks for model training.

        Returns:
            list: List of Keras callbacks.
        """
        return [
            callbacks.ReduceLROnPlateau(
                monitor=self.config.monitor,
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr
            ),
            callbacks.EarlyStopping(
                monitor=self.config.monitor,
                mode='min',
                patience=self.config.early_stopping_patience,
                restore_best_weights=False
            ),
            callbacks.ModelCheckpoint(
                filepath= self.model_path,
                monitor=self.config.metrics[0],
                save_best_only=True
            ),
            WandbMetricsLogger()
        ]

    def summary(self):
        """
        Prints the summary of the model.
        """
        self.model.summary()

    def fit(self,X_train, y_train, X_val, y_val):
        """
        Trains the model and returns it with the history logs

        Args:
            X_train (np.array): Training data features.
            y_train (np.array): Training data labels.
            X_val (np.array): Validation data features.
            y_val (np.array): Validation data labels.

        Returns:
            tuple: The trained model and the training history.
        """
        # X_train, y_train, X_val, y_val = train_inputs
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs, batch_size=self.config.batch_size,
            verbose=1,
            callbacks=self.get_callbacks()
        )
        history = history.history
        return self.model, history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on test data.

        Args:
            X_test (np.array): Test data features.
            y_test (np.array): Test data labels.

        Returns:
            list: Evaluation results.
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Makes predictions using the model.

        Args:
            X (np.array): Input data.

        Returns:
            np.array: Model predictions.
        """
        return self.model.predict(X)


class NNRegressionModel(BaseModel):
    """
    Neural network regression model class that extends BaseModel.
    """
    def __init__(self, input_shape = None, config = None):

        super().__init__(input_shape, config)
        self.model_path=BEST_MODEL_PATH_REG  # Overriding the model_path because it changes whether it is a classification model and a regression one

    def build_model(self):
        """
        Builds the regression model.

        This method calls the parent build_model method to create the base model
        and then adds the output layer with one neuron in the last dense layer to make a continuous prediction, 
        which corresponds to a regression.

        Returns:
            tf.keras.Model: Compiled regression model.
        """
        model = super().build_model()
        model.add(layers.Dense(1))
        self.upload_model_archi(model)
        # optimizer = optimizers.Nadam(learning_rate=self.config.initial_lr, weight_decay=self.config.weight_decay)
        optimizer = optimizers.deserialize(self.config.optimizer)
        model.compile(optimizer=optimizer, loss=self.config.loss, metrics=self.config.metrics)
        
        return model


class NNClassificationModel(BaseModel):
    """
    Neural network classification model class that extends BaseModel.
    """

    def __init__(self, input_shape=None, config=None):
        
        super().__init__(input_shape, config)
        self.model_path=BEST_MODEL_PATH_CLASS

    def build_model(self):
        """
        Builds the classification model.

        This method calls the parent build_model method to create the base model
        and then adds the output layer with softmax activation for classification.

        Returns:
            tf.keras.Model: Compiled classification model.
        """
        model = super().build_model()
        model.add(layers.Dense(self.config.softmax_classes, activation="softmax"))
        self.upload_model_archi(model)

        # optimizer = optimizers.Nadam(learning_rate=self.config.initial_lr, weight_decay=self.config.weight_decay)
        optimizer = optimizers.deserialize(self.config.optimizer)
        model.compile(optimizer=optimizer, loss=self.config.loss, metrics=self.config.metrics)
        
        return model


def train(step=None):

    if  step is None:
        classif = 0
    else: 
        classif = 1
    # City parameters initialization    
    city_list = os.listdir(CLEANED_DIR)
    city_list = ["Bamako.csv"]
    str_city = create_city_str(city_files=city_list)
    train_city_list=[]
    params_to_take=[]
    params_to_drop=[]
    params_list = params_to_take, params_to_drop
    str_drop_params = create_str_for_drop_parameters(params_to_drop=params_to_drop)
    temps_to_classes, classes_to_temps = dict(), dict()
    
    # Layers
    dense_1 = 150
    dense_2 = 256
    dense_3 = 150
    dense_4 = 64
    dense_5 = 32
    
    # Hyper parameters
    frac_urb_train = 0.8
    dropout = 0.10
    alpha_elu = 1.0
    initial_lr =  1e-2
    weight_decay = 4e-4
    min_lr  =  1e-9

    epochs = 250
    batch_size = 512
    optimizer = optimizers.Nadam(learning_rate=initial_lr, weight_decay=weight_decay)
    optimizer_dict = optimizers.serialize(optimizer)
    monitor = "val_loss"
    if classif:
        loss =  "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else : 
        loss = "mse"
        metrics = ["mae"]

    # Callback parameters
    early_stopping_patience = 25
    reduce_lr_factor = 0.2
    reduce_lr_patience =  7
    
    # Other parameters
    model = None
    n_rur = 100000
    n_urb = 200000
    if classif:
        step = step
    
    print("\nLoading datasets...")
    df_train, df_test, df_val = load_and_separate_data_for_train(city_list, classif=classif, step=step,n_rur=n_rur, n_urb=n_urb, train_city_list=train_city_list, frac_urb_train=frac_urb_train)
    df_all = pd.concat([df_train,df_test,df_val],ignore_index=True)
    df_all = take_right_parameters(df_all,*params_list)

    # This is the command to prepare test data without taking care of what has been done before 
    # df_test = preoprocess_test_data(city_list=city_list)

    if classif:
        model_type = "class"
        temps_to_classes, classes_to_temps = class_label_dicts(df_all)
        n_classes = len(temps_to_classes)
        df_train, df_test, df_val = df_labels_to_class_for_all(df_train, df_test, df_val, temps_to_classes)
    else:
        n_classes = None
        model_type = "reg" 
    
    config={
        "dense_1": dense_1,
        "dense_2": dense_2,
        "dense_3": dense_3,
        "dense_4": dense_4,
        "dense_5": dense_5,

        "epochs":epochs,
        "batch_size": batch_size,
        "loss": loss, 
        "optimizer" : optimizer_dict,
        "metrics": metrics,
        "monitor": monitor, 

        "frac_urb_train" : frac_urb_train,
        "initial_lr": initial_lr,
        "min_lr" : min_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "alpha_elu": alpha_elu,
        
        "early_stopping_patience": early_stopping_patience,
        "reduce_lr_factor": reduce_lr_factor,
        "reduce_lr_patience": reduce_lr_patience,

        "softmax_classes" : n_classes,
        "classif_step" : step
    }
    save_config(config, classif)
    
    # Wand init
    directory = f"train_{model_type}_{str_city}"
    path = os.path.join("wandb", directory)
    os.makedirs(path,exist_ok=True)
    os.environ["WANDB_DIR"] = path
    wandb.init(
        
        project="Temperature prediction",
        name = f'train_{model_type}_{str_city}_drop_{config["dropout"]}_step_{step}_{config["optimizer"]["class_name"]}_bs_{config["batch_size"]}_ep_{config["epochs"]}_urb_{config["frac_urb_train"]}_layers_{config["dense_1"]}_{config["dense_2"]}_{config["dense_3"]}_{config["dense_4"]}_{config["dense_5"]}{str_drop_params}', #{dense_6}", 
        config=config,
        # id = f"new_drop_{dropout}_step_{step}_nadam_bs_{batch_size}_ep_{epochs}_urb_{frac_urb_train}",
        mode="online"
    )
    config = wandb.config

    # Display some statistique about data
    if train:
        heat_map(df_all,"Heat map of the variable next to LST")
        stat_on_data_with_zone(df_train.copy(),classif=classif,classes_to_labels=classes_to_temps,wandb_title="Train set distribution")
        stat_on_data_with_zone(df_test.copy(), classif=classif,classes_to_labels=classes_to_temps,wandb_title="Test set distribution")

    print("Creating input...") 
    X_train, y_train, X_test , y_test, X_val, y_val = create_X_y_for_all(df_train=df_train,df_test=df_test,df_val=df_val,params_list=params_list)


    # Rdf model train
    # inputs = X_train,y_train, X_test, y_test
    # rdf_regressor(inputs,force=force)
    
    scaler = StandardScaler()
    X_train, X_test, X_val = scale_data(X_train, X_test, X_val, scaler,classif=classif)   
    test_inputs = X_test, y_test
    input_shape = (X_train.shape[1],)
    

    if classif:
        model = NNClassificationModel(input_shape=input_shape, config=config)
        model.build_model()
        print("Training the model...")
        model, history = model.fit(X_train, y_train, X_val, y_val)
        save_history(history,MODEL_HISTORY_CLASS)
        loss_and_metrics_vis(history,"Evolution of accuracy and loss during training")
    
    else:
        model = NNRegressionModel(input_shape=input_shape,config=config)
        model.build_model()
        print("Training the model...")
        model, history = model.fit(X_train, y_train, X_val, y_val)
        save_history(history,MODEL_HISTORY_REG)
        loss_and_metrics_vis(history,"Evolution of mae and loss during training")
    # neural_net_importance(model,X_train,X_test)
    visualization(classif=classif, classes_to_temps=classes_to_temps,test_inputs=test_inputs,wandb_title="Results of the model predictions", model=model)

def test(step=None):

    if  step is None:
        classif = 0
    else: 
        classif = 1

    test_city_list = ["Bamako.csv"]
    str_city = create_city_str(city_files=test_city_list)
    params_to_take=[]
    params_to_drop=[]
    params_list = params_to_take, params_to_drop
    str_drop_params = create_str_for_drop_parameters(params_to_drop=params_to_drop)
    temps_to_classes, classes_to_temps = dict(), dict()
    

    print("\nLoading datasets...")
    df_test, df_all = create_df_test(city_list=test_city_list,classif=classif,step=step)
    df_test = take_right_parameters(df_test,*params_list)

    if classif:
        temps_to_classes, classes_to_temps = class_label_dicts(df_all)

        # Change labels to class categories
        df_test = label_to_class(df_test,temps_to_classes)
        model_type = "class"
    else:
        model_type = "reg"

    directory = f"test_{model_type}_{str_city}"
    path = os.path.join("wandb", directory)
    os.makedirs(path,exist_ok=True)
    os.environ["WANDB_DIR"] = path
    config = load_config(classif=classif)
    wandb.init(
        
        project="Temperature prediction",
        name = f'test_{model_type}_{str_city}_drop_{config["dropout"]}_step_{step}_{config["optimizer"]["class_name"]}_bs_{config["batch_size"]}_ep_{config["epochs"]}_urb_{config["frac_urb_train"]}_layers_{config["dense_1"]}_{config["dense_2"]}_{config["dense_3"]}_{config["dense_4"]}_{config["dense_5"]}{str_drop_params}',
        # Penser à rajouter le chargement de la config id = f"new_drop_{dropout}_step_{step}_nadam_bs_{batch_size}_ep_{epochs}_urb_{frac_urb_train}",
        mode="online"
    )
    wandb.config = config


  
    heat_map(df_test,"Heat map of the variable next to LST on test dataset")
    stat_on_data_with_zone(df_test.copy(), classif=classif,classes_to_labels=classes_to_temps,wandb_title="Test set distribution")


    print("Creating input...") 
    X_test, y_test = create_X_y(df_test,parameters_list=params_list)

    if classif:
        scaler : StandardScaler = load(MODEL_SCALER_PATH_CLASS)
    else:
        scaler : StandardScaler = load(MODEL_SCALER_PATH_REG)

    if not X_test.empty:
        X_test = scaler.transform(X_test)
        
    test_inputs = X_test, y_test

    if classif:
            model = NNClassificationModel()
            model.loads_model()
        
    else:
            model = NNRegressionModel()
            model.loads_model()


    print("Making iterations on test data...")
    visualization(classif=classif, classes_to_temps=classes_to_temps,test_inputs=test_inputs,wandb_title="Results of the model predictions", model=model)

