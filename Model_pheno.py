import keras
from keras import layers
import matplotlib.pyplot as plt
# this file is used to specify and train the model for the phenomenological task
directory = "C:/Users/ferdi/Phenomenological"
batch_size = 16
epochs = 70
seed_train_val = 25
validation_split = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.045,
    decay_steps=1204,
    decay_rate=0.84)

train_ds = keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(299, 299),
    shuffle=True,
    seed=seed_train_val,
    validation_split=validation_split,
    subset='training',
    verbose=True,
)

val_ds = keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(299, 299),
    shuffle=True,
    seed=seed_train_val,
    validation_split=validation_split,
    subset='validation',
    verbose=True,
)


augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
]


def data_augmentation(image):
    for layer in augmentation_layers:
        image = layer(image)
    return image


train_ds = train_ds.map(lambda z, y: (data_augmentation(z), y))

base_model = keras.applications.Xception(
    include_top=False,
    pooling="avg",
    weights="imagenet",
    input_shape=(299, 299, 3),
    classifier_activation="softmax",
)

base_model.trainable = False

inputs = keras.Input(shape=(299, 299, 3))
scale_layer = layers.Rescaling(scale=1 / 255)
x = scale_layer(inputs)
x = base_model(x, training=False)
x = layers.Dense(512, activation="relu")(x)
x = keras.layers.Dropout(0.35)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(4, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.summary(show_trainable=True)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()],
)

print("Fitting the top layer of the model")
history = model.fit(train_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=val_ds)
model.save("C:/Users/ferdi/OneDrive/Masterarbeit/Models/Pheno_Model.keras")

print(history.params)
print(history.history.keys())

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Phenomenological-Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Plots/Phenomenological_Accuracy_plot', bbox_inches='tight')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Phenomenological-Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Plots/Phenomenological_Loss_plot', bbox_inches='tight')
plt.show()
