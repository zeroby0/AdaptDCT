from pathlib import Path
import tensorflow as tf
from dataset import make_dataset
import toml


runtag = 'tiny-basic'

savedir = Path(f'saved_models/{runtag}/')
savedir.mkdir(exist_ok=True, parents=True)

train_x, train_y1, train_y2 = make_dataset(range(1, 600))
valid_x, valid_y1, valid_y2 = make_dataset(range(600, 700))
test_x, test_y1, test_y2 = make_dataset(range(700, 840))


# def create_cnn_model(input_shape=(256, 256, 1)):
#     model = tf.keras.models.Sequential(
#         [
#             tf.keras.layers.Input(shape=input_shape),

#             tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

#             tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

#             tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

#             tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dropout(0.5),

#             tf.keras.layers.Dense(
#                 128,
#             ),
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.Dropout(0.5),

#             tf.keras.layers.Dense(
#                 120,
#             ),
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.Dropout(0.5),

#             tf.keras.layers.Dense(24), # was 8
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.Dropout(0.2),

#             tf.keras.layers.Dense(128),
#         ]
#     )

#     return model


# 1 (4, 4)
# 3 (4, 4)
# 3 (4, 4)
# flatten
# 128
# 24
# 128
# 0.009630

# 3 (4, 4)

def create_cnn_model(input_shape=(256, 256, 1)):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),

            tf.keras.layers.Conv2D(3, (3, 3), padding='same'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

            tf.keras.layers.Conv2D(3, (3, 3), padding='same'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

            tf.keras.layers.Conv2D(3, (3, 3), padding='same'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(128),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(24), # was 8
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(128),
        ]
    )

    return model




# Prepare the datasets
def prepare_dataset(x, y1, y2, batch_size):
    y = tf.concat([y1, y2], axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.shuffle(1000).batch(batch_size)


@tf.function
def fastloss(y_true, y_pred, delta=1.0):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    abs_error = tf.abs(y_true - y_pred)

    quadratic = tf.square(abs_error) / (2 * delta)
    linear = (delta * abs_error)

    return tf.math.reduce_mean(tf.where(abs_error <= delta, linear, quadratic), axis=-1)


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)

        # loss = (
        #     tf.keras.losses.MAE(y, predictions)
        #     + tf.square(tf.keras.losses.MAE(y[:, 64:], predictions[:, 64:]) * 1000)
        # )

        loss = (
            tf.keras.losses.MAE(y, predictions)
            + tf.square(tf.keras.losses.MAE(y[:, :64], predictions[:, :64]) * 1000)
            + tf.square(tf.keras.losses.MAE(y[:, 64:], predictions[:, 64:]) * 1000)
        )

        # loss = tf.keras.losses.MAE(y, predictions) * 100


        # 0.0074 was best with reg and both ssim and size

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions


@tf.function
def valid_step(model, x, y):
    predictions = model(x, training=False)

    # loss = tf.keras.losses.MAE(y, predictions)
    loss = (
        tf.keras.losses.MAE(y, predictions)
        + tf.square(tf.keras.losses.MAE(y[:, 64:], predictions[:, 64:]) * 1000)
    )

    return loss, predictions


def train_model(model, train_data, valid_data, epochs):
    best_vloss = float("inf")
    best_sizemae = float("inf")
    best_ssimmae = float("inf")

    with open(savedir / "config.toml", "a") as conffile:
        toml.dump(model.get_config(), conffile)


    # LR best performer
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 10000, 1e-4) # LR1
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 2e-5) # LR2
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 1e-4) # LR4
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 40000, 1e-5) # LR5
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 60000, 1e-5) # LR6
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 1e-6) # LR6
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 5e-5) # LR7

    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 1e-4) # LR9
    
    optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

    # Metrics
    train_loss_metric = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    valid_loss_metric = tf.keras.metrics.Mean("valid_loss", dtype=tf.float32)
    ssim_metric = tf.keras.metrics.MeanAbsoluteError("ssim_mae")
    size_metric = tf.keras.metrics.MeanAbsoluteError("size_mae")

    for epoch in range(epochs):
        # Training loop
        for x_batch, y_batch in train_data:
            loss, predictions = train_step(model, optimizer, x_batch, y_batch)
            train_loss_metric.update_state(loss)

        # Validation loop
        for x_batch, y_batch in valid_data:
            loss, predictions = valid_step(model, x_batch, y_batch)
            valid_loss_metric.update_state(loss)

            ssim_metric.update_state(y_batch[:, :64], predictions[:, :64])
            size_metric.update_state(y_batch[:, 64:], predictions[:, 64:])

        result = (
            f"Epoch {epoch + 1}, "
            f"Train Loss: {train_loss_metric.result():.4f}, "
            f"Valid Loss: {valid_loss_metric.result():.4f}, "
            f"SSIM MAE: {ssim_metric.result():.4f}, "
            f"Size MAE: {size_metric.result():.4f}, "
            f"Best Vloss: {best_vloss:.4f}, "
            f"Best SSIM MAE: {best_ssimmae:.6f}, "
            f"Best Size MAE: {best_sizemae:.6f}"
        )

        # Print metrics
        print(result)

        with open(savedir / "results.txt", "a") as resfile:
            resfile.write(result + "\n")

        if valid_loss_metric.result() < best_vloss:
            model.save(savedir / "bestvloss.keras")
            best_vloss = valid_loss_metric.result()

        if size_metric.result() < best_sizemae:
            model.save(savedir / "best_sizemae.keras")
            best_sizemae = size_metric.result()

        if ssim_metric.result() < best_ssimmae:
            model.save(savedir / "best_ssimmae.keras")
            best_ssimmae = ssim_metric.result()

        # Reset metrics for next epoch
        train_loss_metric.reset_state()
        valid_loss_metric.reset_state()
        ssim_metric.reset_state()
        size_metric.reset_state()


# Usage example
batch_size = 32
epochs = 2000

# Prepare datasets
train_data = prepare_dataset(train_x, train_y1, train_y2, batch_size)
valid_data = prepare_dataset(valid_x, valid_y1, valid_y2, batch_size)

model = create_cnn_model()
model.summary()

train_model(model, train_data, valid_data, epochs)
