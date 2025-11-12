from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers

def build_model(**kwargs):
    defaults = dict(
        hidden_units=[128, 64],
        dropout=0.2,
        batch_size=64,
        epochs=30,
        learning_rate=0.001,
        random_state=42,
        input_dim=None,
    )
    defaults.update(kwargs)
    hu = defaults["hidden_units"]; dp=defaults["dropout"]; lr=defaults["learning_rate"]
    epochs=defaults["epochs"]; bs=defaults["batch_size"]; input_dim=defaults["input_dim"]

    def make_model(meta):
        input_dim = meta["n_features_in_"]
        inp = keras.Input(shape=(input_dim,))
        x = inp
        for units in hu:
            x = layers.Dense(units)(x); x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x); x = layers.Dropout(dp)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
        return m

    return KerasClassifier(model=make_model, epochs=epochs, batch_size=bs, verbose=0)