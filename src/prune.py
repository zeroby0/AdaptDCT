from pathlib import Path
import tensorflow as tf
from dataset import make_dataset


import numpy as np

prune_percent = 0.999


def print_model_weights_sparsity(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Wrapper):
            weights = layer.trainable_weights
        else:
            weights = layer.weights
        for weight in weights:
            # ignore auxiliary quantization weights
            if "quantize_layer" in weight.name:
                continue
            weight_size = weight.numpy().size
            zero_num = np.count_nonzero(weight == 0)
            print(
                f"{weight.name}: {zero_num/weight_size:.2%} sparsity ",
                f"({zero_num}/{weight_size})",
            )


model = tf.keras.models.load_model(
    "saved_models/tiny-basic/bestvloss.keras"
)
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss="mean_absolute_error",
    # loss_weights=[1] * 64 + [4] * 64,
    metrics="mean_absolute_error",
)

# model.load_weights("vanilla.weights.h5")

model.summary()


# This HAS to be imported AFER loading the model, or the model loading fails
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
        prune_percent, begin_step=0, frequency=100
    )
}


pruned_model = prune_low_magnitude(model, **pruning_params)


import tempfile

logdir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    tf.keras.callbacks.ModelCheckpoint(
        f"saved_models/tiny-basic/{prune_percent}-prune_best_valmae.keras",
        monitor="val_mean_absolute_error",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=25,
        verbose=1,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=50,
    )

]


pruned_model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
    loss="mean_absolute_error",
    loss_weights=[1] * 64 + [4] * 64,
    metrics="mean_absolute_error",
)


train_x, train_y1, train_y2 = make_dataset(range(1, 600))
valid_x, valid_y1, valid_y2 = make_dataset(range(600, 700))
test_x, test_y1, test_y2 = make_dataset(range(700, 840))


pruned_model.fit(
    x=train_x,
    y=tf.concat([train_y1, train_y2], axis=1),
    batch_size=64,
    epochs=200,
    callbacks=callbacks,
    verbose=2,
    validation_data=(test_x, tf.concat([test_y1, test_y2], axis=1))
)

# Evaluate the dense model.
_, pruned_mae = pruned_model.evaluate(
    x=test_x, y=tf.concat([test_y1, test_y2], axis=1), verbose=0
)

_, dense_mae = model.evaluate(
    x=test_x, y=tf.concat([test_y1, test_y2], axis=1), verbose=0
)


print_model_weights_sparsity(pruned_model)
print(dense_mae, pruned_mae)

model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
model_for_export.save('saved_models/tiny-basic/pruned.export.keras')
