import tensorflow as tf
import os
import kagglehub
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Téléchargement
path_brut = kagglehub.dataset_download("alessiocorrado99/animals10")
path_final = os.path.join(path_brut, "raw-img")

# Création du Dataset (Entraînement)
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_final,
    validation_split=0.2, # 20% pour le test
    subset="training",
    seed=123,
    image_size=(64, 64),  # Taille identique à ton code manuel
    batch_size=32         # On traite les images par paquets de 32
)

# Création du Dataset (Test)
test_ds = tf.keras.utils.image_dataset_from_directory(
    path_final,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(64, 64),
    batch_size=32
)

# Construction du Modèle --------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(64, 64, 3)), #Normalisation : on passe de [0, 255] à [0, 1]
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.summary()



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
nom_du_fichier = 'mon_modele_animaux.keras'

# VERIFICATION : Est-ce que j'ai déjà un cerveau entraîné sur mon disque ?
if os.path.exists(nom_du_fichier):
    print("Succès : Modèle trouvé ! Chargement en cours...")
    model = tf.keras.models.load_model(nom_du_fichier)
else:
    print("Modèle non trouvé. Début de l'entraînement (cela peut prendre du temps)...")
    # On entraîne seulement si on n'a pas de sauvegarde
    history = model.fit(train_ds, epochs=10, validation_data=test_ds)

    # SAUVEGARDE : On enregistre pour la prochaine fois
    model.save(nom_du_fichier)
    print(f"Modèle sauvegardé sous le nom : {nom_du_fichier}")


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(test_acc)

class_names = train_ds.class_names






def predire_animal(chemin_image):
    img = tf.keras.utils.load_img(chemin_image, target_size=(64, 64))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Ajouter la dimension de "batch" : (64,64,3) -> (1,64,64,3)

    predictions = model.predict(img_array) # Faire la prédiction

    score = tf.nn.softmax(predictions[0])

    # Trouver l'indice avec la plus haute probabilité
    indice_max = tf.math.argmax(score)
    nom_classe = class_names[indice_max]
    confiance = 100 * tf.math.reduce_max(score)

    print(f"\nRésultat : {nom_classe} ({confiance:.2f}% de confiance)")

    # Affichage de l'image avec le résultat
    plt.imshow(img)
    plt.title(f"Prédiction : {nom_classe}")
    plt.axis('off')
    plt.show()


# TEST SUR UNE IMAGE RÉELLE ---------------------------------------------------------------------

chemin_test = "cat.png"

if os.path.exists(chemin_test):
    predire_animal(chemin_test)
else:
    print(f"\nAttention : Le fichier {chemin_test} n'a pas été trouvé.")