import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.spatial import distance


def myLossFunction(sample):
    distances = []
    bad_class =
    for bad_samp in bad_class:
        dist = distance.cosine(sample, bad_samp)
        distances.append(dist)

    return 1/np.mean(distances)

myLossFunctionVec = np.vectorize(myLossFunction)

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        sample_weight = None
        x = data

        with tf.GradientTape() as tape:
            # y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = myLossFunctionVec(x)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
