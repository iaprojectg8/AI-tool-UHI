from utils.imports import *
from utils.variables_path import *

def nn_train(inputs,force,config):
    global history
    X_train,y_train, X_test,y_test, X_val,y_val =  inputs

    if force:
        model = models.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                layers.Dense(config.dense_1),
                layers.BatchNormalization(),
                layers.ELU(alpha=config.alpha_elu,),  # putting the activation layer after the batchnorm because it should be more efficient
                layers.Dropout(config.dropout),
                
                # Hidden layers
                layers.Dense(config.dense_2),
                layers.BatchNormalization(),
                layers.ELU(alpha=config.alpha_elu,), # Putting elu because it help from gradient vanishing
                layers.Dropout(config.dropout),
                
                layers.Dense(config.dense_3),
                layers.BatchNormalization(),
                layers.ELU(alpha=config.alpha_elu,), 
                layers.Dropout(config.dropout),

                layers.Dense(config.dense_4),
                layers.BatchNormalization(),
                layers.ELU(alpha=config.alpha_elu,), 
                layers.Dropout(config.dropout),

                layers.Dense(config.dense_5),
                layers.BatchNormalization(),
                layers.ELU(alpha=config.alpha_elu,), 
                layers.Dropout(config.dropout),

                layers.Dense(config.dense_6),
                layers.BatchNormalization(),
                layers.ELU(alpha=config.alpha_elu,), 
                layers.Dropout(config.dropout),

                layers.Dense(1),  # Output layer for continuous prediction
            ])

        # Compiler le modèle

        initial_learning_rate = config.initial_lr
        weight_decay = config.weight_decay
        optimizer = optimizers.AdamW(learning_rate=initial_learning_rate,weight_decay=weight_decay)
        model.compile(optimizer=optimizer, loss=config.loss, metrics = config.metrics)
        model.summary()

        # Define the ReduceLROnPlateau callback
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=config.monitor,  
            factor=config.reduce_lr_factor,         
            patience=config.reduce_lr_patience,          
            min_lr=1e-9        
        )

        # Define the EarlyStopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor=config.monitor, 
             mode='min',
            patience=config.early_stopping_patience,  
            restore_best_weights=False  
        )

        # Define the ModelCheckpoint callback
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,  # Path where the model will be saved
            monitor='mae',
            save_best_only=True  # Save only the best model
           
        )
        
        # Register the signal handler
        
        history = model.fit(X_train, y_train,
                            epochs=config.epochs, batch_size=config.batch_size,
                            validation_data=(X_val, y_val),
                            verbose=1,
                            callbacks=[reduce_lr, early_stopping, model_checkpoint,WandbMetricsLogger()])
        
        
        history = history.history
        model.save(MODEL_NN_PATH)
        # Save dictionary to a file
    
    

        with open(MODEL_HISTORY, 'w') as file:
            json.dump(history, file)
    else:
        model = models.load_model(BEST_MODEL_PATH)
        print("Model loaded")
        history = None

    # Évaluer le modèle
    loss, mae = model.evaluate(X_test, y_test)
    print(loss,mae)
    
    # Visualize the training and validation loss over epochs

    return model, history
