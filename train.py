import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots as files, don't open windows
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
EPOCHS = 12
FINE_EPOCHS = 5

os.makedirs("model", exist_ok=True)

# 1. LOAD IMAGES (resized to 224x224, 80% train / 20% validation)
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)

class_names = train_ds.class_names
print("Classes:", class_names)

# Save class names so app.py uses the same order
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

# 2. CLASS WEIGHTS (helps classes that have fewer images)
train_labels = np.concatenate([y.numpy() for _, y in train_ds])
weights = compute_class_weight("balanced",
                               classes=np.arange(len(class_names)),
                               y=train_labels)
class_weight = dict(enumerate(weights))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 3. BUILD MODEL (MobileNetV2 transfer learning)
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False  # keep pretrained knowledge for now

inputs = keras.Input(shape=(224, 224, 3))
x = augment(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # pixels -> -1 to 1
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True)

# 4. TRAIN (top layers only)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
               class_weight=class_weight, callbacks=[early_stop])

# 5. FINE-TUNE (unfreeze the last 30 layers, tiny learning rate)
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=FINE_EPOCHS,
               class_weight=class_weight, callbacks=[early_stop])

# 6. SAVE MODEL
model.save("model/fruit_model.keras")
print("Model saved to model/fruit_model.keras")

# 7. ACCURACY / LOSS GRAPH
acc = h1.history["accuracy"] + h2.history["accuracy"]
val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"] + h2.history["loss"]
val_loss = h1.history["val_loss"] + h2.history["val_loss"]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="train")
plt.plot(val_acc, label="validation")
plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label="train")
plt.plot(val_loss, label="validation")
plt.title("Loss"); plt.legend()
plt.savefig("model/training_plot.png")
plt.close()

# 8. EVALUATE ON THE TEST SET (images the model never saw)
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTEST ACCURACY: {test_acc * 100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)
print("\n", classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 8))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
plt.yticks(range(len(class_names)), class_names, fontsize=7)
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.close()
print("Saved training_plot.png and confusion_matrix.png in model/")