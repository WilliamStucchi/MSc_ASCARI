import tensorflow as tf
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------

def incremental_update(model, new_data, new_labels, initial_learning_rate=0.001, reduction_factor=0.1, epochs=1,
                       batch_size=1, save_path='../test/test_incremental_learning/model_incremental.h5'):
    """
    Incrementally update the model with new data and save the model after updating.

    Args:
    - model: The trained Keras model.
    - new_data: New input data (numpy array).
    - new_labels: New target data (numpy array).
    - initial_learning_rate: The initial learning rate used for the original training.
    - reduction_factor: Factor by which to reduce the learning rate for incremental updates.
    - epochs: Number of epochs for incremental training (default: 1).
    - batch_size: Batch size for incremental training (default: 1).
    - save_path: Path to save the updated model.
    """
    new_data = new_data.reshape(1, 20)
    new_labels = new_labels.reshape(1, 3)

    # Calculate the reduced learning rate
    new_learning_rate = initial_learning_rate * reduction_factor

    # Create a new optimizer with the reduced learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=new_learning_rate)

    # Compile the model with the new optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Perform incremental update
    model.fit(x=new_data, y=new_labels, epochs=epochs, batch_size=batch_size, verbose=1)

    # Save the updated model
    model.save(save_path)
    print(f'Model saved to {save_path} with learning rate {new_learning_rate}')


# -----------------------------------------------------------------------------------------------------------------------

# load model
model_path = 'scirob_submission/Model_Learning/saved_models/step_1/callbacks/2024_05_17/12_29_37/'
if model_path is not None:
    loaded_model = tf.keras.models.load_model(model_path + 'keras_model.h5')

# load data
data_path = 'scirob_submission/Model_Learning/data/new/train_data_step1_mass.csv'
if data_path is not None:
    data = np.loadtxt(data_path, delimiter=',')

for i, row in tqdm(enumerate(data)):
    incremental_update(loaded_model, np.array(row[:-3]), np.array(row[-3:]), initial_learning_rate=0.001, reduction_factor=0.1,
                       epochs=1, batch_size=5)
