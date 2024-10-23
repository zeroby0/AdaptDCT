import numpy as np
import tensorflow as tf
from tensorflow import keras
# from Model import create_cnn_model
from dataset import make_dataset
from pathlib import Path
# Assume 'pruned_model' is your pruned model


def get_non_zero_indexes(weights):
    return np.where(np.any(weights != 0, axis=tuple(range(1, len(weights.shape)))))[0]


def create_smaller_model(pruned_model):
    new_model = keras.Sequential()
    previous_output_shape = pruned_model.input_shape[1:]  # Start with the input shape

    print(pruned_model.input_shape)
    
    for layer in pruned_model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            weights, biases = layer.get_weights()
            non_zero_filters = get_non_zero_indexes(weights)

            print(f'Layer {layer.name}. Indexes of non-zero-filters: ', non_zero_filters)
            print('Weights shape: ', weights.shape, 'Biases shape:', biases.shape)
            
            if len(non_zero_filters) > 0:
                new_conv = keras.layers.Conv2D(
                    filters=len(non_zero_filters),
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    input_shape=previous_output_shape
                )
                new_model.add(new_conv)
                
                new_weights = weights[:, :, :previous_output_shape[-1], non_zero_filters]
                new_biases = biases[non_zero_filters]
                new_conv.set_weights([new_weights, new_biases])

                print('New weights shape:', new_weights.shape)
                
                # Update the output shape for the next layer
                previous_output_shape = new_conv.compute_output_shape(previous_output_shape)
                print('New Output shape:', previous_output_shape, '\n')
        
        elif isinstance(layer, keras.layers.Dense):
            weights, biases = layer.get_weights()
            non_zero_units = get_non_zero_indexes(weights.T)
            
            print(f'Layer {layer.name}. Indexes of non-zero-units: ', non_zero_units)
            print('Weights shape: ', weights.shape, 'Biases shape:', biases.shape)

            if len(non_zero_units) > 0:
                new_dense = keras.layers.Dense(
                    units=len(non_zero_units),
                    activation=layer.activation
                )
                new_model.add(new_dense)
                
                new_weights = weights[:previous_output_shape[-1], non_zero_units]
                new_biases = biases[non_zero_units]
                new_dense.set_weights([new_weights, new_biases])
                
                # Update the output shape for the next layer
                previous_output_shape = new_dense.compute_output_shape(previous_output_shape)
        
        else:
            print(f'Adding layer: {layer.name}')
            new_model.add(layer)

            previous_output_shape = layer.compute_output_shape((None, *previous_output_shape))[1:]

            if layer.name == 'flatten': previous_output_shape = (None, *previous_output_shape)


            # if 'max_pooling2d' in layer.name:
            # # Update the output shape for the next layer
            # # previous_output_shape = layer.compute_output_shape(previous_output_shape)
            #     previous_output_shape = layer.compute_output_shape((None, *previous_output_shape))[1:]
            
            # else:
            #     previous_output_shape = layer.compute_output_shape((None, *previous_output_shape))
            print('New Output shape:', previous_output_shape, '\n')
    
    return new_model


# model = create_cnn_model()
model = tf.keras.models.load_model('saved_models/tiny-basic/pruned.export.keras')
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss="mean_absolute_error",
    loss_weights=[1] * 64 + [4] * 64,
    metrics="mean_absolute_error",
)

# model.load_weights("vanilla.keras")
# model.load_weights("pruned.export.keras")
model.summary()


# Create the smaller model
smaller_model = create_smaller_model(model)

# Compile the smaller model
smaller_model.compile(
    optimizer=model.optimizer,
    loss=model.loss,
    metrics=model.metrics
)

# Save the smaller model
smaller_model.save('smaller_model.keras')
smaller_model.summary()

test_x, test_y1, test_y2 = make_dataset(range(700, 840))


_, dense_mae = smaller_model.evaluate(
    x=test_x, y=tf.concat([test_y1, test_y2], axis=1), verbose=0
)

print("\nModel MAE:", dense_mae)