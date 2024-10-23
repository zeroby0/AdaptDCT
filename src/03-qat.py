# saved_models/best-tiny-no-bnorm-noreg-square1000-lrsched9
# has the best model so far
# It's the smaller model. Init it from the config,
# load the weights, and start quantising it.
# We don't need to prune anymore, it's the small model.
# No batch norm in this, so hopefully only a few float layers.


from pathlib import Path
import tensorflow as tf
from dataset import make_dataset
from tensorflow_model_optimization.python.core.keras.compat import keras


train_x, train_y1, train_y2 = make_dataset(range(1, 600))
valid_x, valid_y1, valid_y2 = make_dataset(range(600, 700))
test_x, test_y1, test_y2 = make_dataset(range(700, 840))

model_path = Path('saved_models/tiny-basic/bestvloss.keras')

model = keras.models.load_model(model_path)
model.summary()


import tensorflow_model_optimization as tfmot

quantized_model = tfmot.quantization.keras.quantize_model(model)


lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 1e-4) # LR9

# `quantize_model` requires a recompile.
quantized_model.compile(
    optimizer=keras.optimizers.Adamax(learning_rate=lr_schedule),
    loss="mean_absolute_error",
    loss_weights=[1] * 64 + [4] * 64,
    metrics="mean_absolute_error",
)

quantized_model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path.parent / f"q-{model_path.stem}.keras"),
        monitor="val_mean_absolute_error",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),

    # tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0,
    #     patience=25,
    #     verbose=1,
    #     mode="min",
    #     restore_best_weights=True,
    #     start_from_epoch=50,
    # )
]

quantized_model.fit(
    x=train_x,
    y=tf.concat([train_y1, train_y2], axis=1),
    batch_size=64,
    epochs=400,
    callbacks=callbacks,
    verbose=2,
    validation_data=(test_x, tf.concat([test_y1, test_y2], axis=1))
)

# # # Evaluate the dense model.
_, pruned_mae = quantized_model.evaluate(
    x=test_x, y=tf.concat([test_y1, test_y2], axis=1), verbose=0
)

print(pruned_mae)