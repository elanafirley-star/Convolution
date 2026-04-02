import numpy as np
import kagglehub
import os
import cv2

# 1. Téléchargement
path_brut = kagglehub.dataset_download("alessiocorrado99/animals10")

# 2. Correction du chemin (Le dataset Kaggle contient un sous-dossier 'raw-img')
path_final = os.path.join(path_brut, "raw-img")


# Note : Si le print de classes affiche une liste vide après,
# vérifie si le dossier s'appelle bien 'raw-img' ou s'il n'y en a pas.

def charger_animaux(path, max_images_par_classe ,taille=(64, 64)):
    images = []
    labels = []

    if not os.path.exists(path):
        print(f"Erreur : Le chemin {path} n'existe pas.")
        return None, None, []

    # On liste les dossiers (un dossier = une classe d'animal)
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    for idx, nom_classe in enumerate(classes):
        dossier_classe = os.path.join(path, nom_classe)

        print(f"Chargement de {nom_classe} (Classe {idx})...")
        compteur = 0

        # On liste les fichiers images
        for fichier in os.listdir(dossier_classe):
            if compteur >= max_images_par_classe:
                break

            chemin_img = os.path.join(dossier_classe, fichier)

            # Lecture
            img = cv2.imread(chemin_img, cv2.IMREAD_COLOR)

            if img is not None:
                # Redimensionnement (Obligatoire pour le CNN)
                img = cv2.resize(img, taille)
                img = img.transpose(2,0,1) #on déplace k'index 2 en position pour avoir (3,64,64) au lieu de (64,64,3)

                # Conversion en float32 pour éviter les erreurs de calcul plus tard
                images.append(img.astype(np.float32))
                labels.append(idx)
                compteur += 1

    return np.array(images), np.array(labels), classes


# --- TEST DU CHARGEMENT ---
images, labels, noms_classes = charger_animaux(path_final, taille=(64, 64), max_images_par_classe=10)

if images is not None:
    print(f"\nChargement terminé !")
    print(f"Nombre total d'images : {len(images)}")
    print(f"Format d'une image : {images[0].shape}")
    print(f"Classes trouvées : {noms_classes}")

def melanger_images(images,labels,ratio= 0.8):#80% entrainement, 20%test
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    limite = int(len(images)*ratio)
    x_train,x_test = images[:limite],images[limite:]
    y_train,y_test = labels[:limite],labels[limite:]

    return x_train,y_train,x_test,y_test












class Convolution():
    def __init__(self, nb_filtres, pas, padding, nb_convolution, kernelsize, pool_size, nb_pooling, image_shape, nb_classes, couches_denses):

        self.nb_filtres = nb_filtres
        self.pas = pas
        self.type_padding = padding
        self.nb_couches_convolution = nb_convolution
        self.taille_noyau = kernelsize
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.nb_classes = nb_classes

        nb_canaux_init = image_shape[0] if len(image_shape) == 3 else 1
        n_in_conv = nb_canaux_init * kernelsize * kernelsize
        self.liste_filtres = []
        nb_canaux_courant = nb_canaux_init
        for _ in range(nb_convolution):
            f = np.random.randn(nb_filtres, nb_canaux_courant, kernelsize, kernelsize) * np.sqrt(2.0 / (nb_canaux_courant * kernelsize ** 2))
            self.liste_filtres.append(f)
            nb_canaux_courant = nb_filtres # La sortie d'une couche devient l'entrée de la suivante

        # calcul auto de la taille après les convolutions pour le Flatten
        self.taille_flat = self._calculer_taille_aplatie(image_shape)

        # Création dynamique des couches Denses
        self.poids_dense = []  # On va stocker des tuples (W, b)

        # La première couche dense part de la taille "flat"
        taille_precedente = self.taille_flat

        # On boucle sur la liste fournie (ex: [128, 64])
        for nb_neurones in couches_denses:
            W = np.random.randn(taille_precedente, nb_neurones) * np.sqrt(2.0 / taille_precedente) #poids des filtres
            b = np.zeros(nb_neurones)
            self.poids_dense.append((W, b))
            taille_precedente = nb_neurones

        # Dernière couche : on connecte au nombre de classes (ex: 10)
        self.W_final = np.random.randn(taille_precedente, nb_classes) * np.sqrt(2.0 / taille_precedente)
        self.b_final = np.zeros(nb_classes)

    def _calculer_taille_aplatie(self, shape):
        """ Calcule la dimension du vecteur après toutes les étapes de convolution et de pooling.
            Cette valeur est indispensable pour définir la taille de la première couche Dense (poids W1).

            Params: shape (tuple): Format de l'image d'entrée (Canaux, Hauteur, Largeur).

            Returns: int: Le nombre total d'éléments (neurones) une fois le volume aplati.
                        Calculé par : Nb_Filtres * Hauteur_finale * Largeur_finale.
            """
        c, h, l = shape
        for _ in range(self.nb_couches_convolution):
            # Simule Conv
            h = (h + 2 * self.type_padding - self.taille_noyau) // self.pas + 1
            l = (l + 2 * self.type_padding - self.taille_noyau) // self.pas + 1
            # Simule Pool
            h = (h - self.pool_size) // self.pas + 1
            l = (l - self.pool_size) // self.pas + 1
        return self.nb_filtres * h * l

    def convolution(self, padding, stride, nb_filtres, kernelsize, image, biais, filtre=None):
        nb_canal, image_h_brute, image_l_brute = image.shape

        filtres_a_utiliser = filtre

        if padding > 0:
            image_ajout_pads = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            image_ajout_pads = image

        image_h, image_l = image_ajout_pads.shape[1], image_ajout_pads.shape[2]

        sortie_h = (image_h - kernelsize) // stride + 1
        sortie_l = (image_l - kernelsize) // stride + 1

        sortie = np.full((nb_filtres, sortie_h, sortie_l), biais)

        for f in range(nb_filtres):
            # ICI : On utilise filtres_a_utiliser au lieu de self.filtres
            filtre_actuel = filtres_a_utiliser[f]

            for y in range(sortie_h):
                for x in range(sortie_l):
                    y_start = y * stride
                    y_end = y_start + kernelsize
                    x_start = x * stride
                    x_end = x_start + kernelsize

                    fenetre = image_ajout_pads[:, y_start:y_end, x_start: x_end]
                    sortie[f, y, x] += np.sum(fenetre * filtre_actuel)

        return sortie



    def activation(self,image,type_fonction):
        """
        Applique une transformation sur l'image (nettoyer l'image en ne gardant que les signes positifs forts)
        :param image: l'image de sortie suite à la convolution
        :param type_fonction: ReLu ou Soft max
        :return: image de même taille avec les valeurs transformées
        """

        if type_fonction == "ReLu":
            return np.maximum(0, image) #ompare chq élémént d'image avec 0

        elif type_fonction == "Softmax":
            exp_img = np.exp(image - np.max(image))  #On calcule l'exp de chq point on soustr le max pour éviter des nombres trop grands
            return exp_img / np.sum(exp_img)

        return image



    def pooling(self,stride,pool_size,image,type_pooling):
        """
        Réduit la dimension de l'image
        :param stride: pas
        :param taille: taille de la fenêtre finale
        :param image: image retournée par la fonction d'activation
        :param type_pooling: Max ou Average pooling
        :param pool_size: taille du pooling
        :return: image de taille réduite
        """
        img = np.array(image)
        if img.ndim == 2:  # noir/blanc
            h_in, l_in = img.shape
            nb_canal = 1  #une info à recup : intensité de noir
            img = img[np.newaxis,:, :]
        else: #3d/couleur ou après convolution
            nb_canal, h_in, l_in= img.shape

        h_out = (h_in - pool_size) // stride + 1
        l_out = (l_in - pool_size) // stride + 1

        sortie = np.zeros((nb_canal, h_out, l_out))

        for canal in range(nb_canal):
            for i in range(h_out):
                for j in range(l_out):
                    y_0 = i * stride
                    x_0 = j * stride
                    fenetre = img[canal, y_0 : y_0 + pool_size, x_0 : x_0 + pool_size] #sliing
                    # y_0: y_0+taille: On prend toutes lignes entre point de départ et la hauteur de l'image
                    # x_0 : x_0+taille : idem pour les colonnes
                    # on reste sur le canal actuel

                    if type_pooling == "Max":
                        sortie[canal, i, j] = np.max(fenetre)
                    else:  #average
                        sortie[canal, i, j] = np.mean(fenetre)

        return sortie.squeeze() if nb_canal == 1 else sortie



    def forward(self, image_entree, biais):  #x est une var temporaire, représente l'état de doonés à chq etape de transformation
        self.cache = {} #dictionnaire pour stocker les étapes
        #boucle
        x = np.array(image_entree, dtype = float) #im brute
        # NORMALISATION : On passe de [0, 255] à [0, 1]
        x = x / 255.0
        if x.ndim == 2: x = x[np.newaxis, :, :] #newaxis crée une dim vide

        for i in range(self.nb_couches_convolution):
            self.cache[f'entree_conv_{i}'] = x  # On stocke l'entrée
            x = self.convolution(self.type_padding, self.pas, self.nb_filtres, self.taille_noyau, x, biais, filtre=self.liste_filtres[i])
            x = self.activation(x, type_fonction="ReLu")
            self.cache[f'avant_pool_{i}'] = x  # On stocke après activation
            x = self.pooling(self.pas, self.pool_size, x, type_pooling="Max")

        self.cache['avant_flatten'] = x
        #applatissement
        x_flat = x.flatten() #apres la boucle, notre image est ransformée en vecteur
        self.cache['activation_dense'] = [x_flat] #on stocke l'entrée du block dense
        x = x_flat
        self.cache['z_dense'] = [] # z= W*x + b
        for W, b in self.poids_dense:
            x = np.dot(x, W) + b #x devient un vecteur de "scores"
            self.cache['z_dense'].append(x) #on stocke le score avant activation
            x = self.activation(x, type_fonction="ReLu")  # Activation entre chaque couche
            self.cache['activation_dense'].append(x) #on stocke le score activé

        #couche de sortie
        score_final = np.dot(x, self.W_final) + self.b_final #x est un veteur de la taille de nos classes (ex si on a dix animaux), il represente maintenant une proba que l'image apartienne à une catégorie
        self.cache['z_final'] = score_final
        return self.activation(score_final, type_fonction="Softmax")




    def backward(self, erreur,lr):
        #Backward des couches densses
        entree_finale = self.cache['activation_dense'][-1]
        grad_W = np.outer(entree_finale,erreur) #np.outer = produit entre 2 vecteurs
        grad_b = erreur

        #on calcule l'erreur qu'on va renvoyer à la couche d'avant
        erreur_courante = np.dot(erreur,self.W_final.T)

        self.W_final -= lr*grad_W
        self.b_final -= lr*grad_b

        #on parcourt les couches en partant de la fin
        for i in reversed(range(len(self.poids_dense))):
            W, b = self.poids_dense[i]
            entree_couche = self.cache['activation_dense'][i] #ce qui est entré dans la couche
            sortie_couche = self.cache['activation_dense'][i + 1] #ce qui est sorti (après activation)
            erreur_courante[sortie_couche <= 0] = 0 #dérivée fonction ReLu

            #on calcule le gradient
            grad_W = np.outer(entree_couche, erreur_courante)
            grad_b = erreur_courante
            #calcul erreur pour la couche encore avant
            erreur_courante = np.dot(erreur_courante, W.T)

            #on met à jour les poids
            new_W = W - lr * grad_W
            new_b = b - lr * grad_b
            self.poids_dense[i] = (new_W, new_b)

        shape_originale = self.cache['avant_flatten'].shape
        gradient_image = erreur_courante.reshape(shape_originale)

        #Backward pooling+convolution
        for i in reversed(range(len(self.nb_couches_convolution))):
            grad_avant_pool = np.zeros_like(self.cache[f'avant_pool_{i}'])


       # return gradient_image

def main():
    animal_vers_indice = {nom: i for i, nom in enumerate(noms_classes)}
    Reseau = Convolution(nb_filtres=16, pas=1,padding=1,
                         nb_convolution=1,kernelsize=3, pool_size=2,
                         nb_pooling=1, image_shape=(3, 64,64),
                         nb_classes=10,couches_denses=[128])
    #x = images, y = labels
    x_train,y_train,x_test,y_test = melanger_images(images,labels)

    # parametres de l'entrainement
    nb_epoques = 20  # On repasse 20 fois sur les images
    lr = 0.01

    print(f"Début de l'entraînement sur {len(x_train)} images...")

    # Phase entrainement
    for epoque in range(nb_epoques):
        perte_totale = 0
        bonnes_reponses_train = 0

        # Mélange
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]

        for i in range(len(x_train)):
            img = x_train[i]
            vrai_idx = y_train[i]
            vecteur_vrai_label = np.zeros(10)
            vecteur_vrai_label[vrai_idx] = 1.0

            resultat = Reseau.forward(img, 0.01)
            erreur = resultat - vecteur_vrai_label
            perte_totale += np.sum(erreur ** 2)

            # On vérifie si la prédiction est correcte
            if np.argmax(resultat) == vrai_idx:
                bonnes_reponses_train += 1

            Reseau.backward(erreur, lr)

        perte_moyenne = perte_totale / len(x_train)
        precision_train = (bonnes_reponses_train / len(x_train)) * 100
        print(f"Époque {epoque + 1}/{nb_epoques} | Perte: {perte_moyenne:.4f} | Précision Train: {precision_train:.2f}%")
    print(f"Entraînement terminé sur {len(x_train)} images. ")

    #Phase de test
    bonnes_predictions = 0
    for i in range(len(x_test)):
        img = x_test[i]
        vrai_label = noms_classes[y_test[i]]
        resultat = Reseau.forward(img, 0.01)
        indice_resultat = np.argmax(resultat)
        prediction = noms_classes[indice_resultat]
        if prediction == vrai_label :
            bonnes_predictions+=1
    taux_reussite = (bonnes_predictions/len(x_test)) *100
    print(f"Le réseaux sait reconnaître les animaux à {taux_reussite}%")

if __name__ == "__main__":
    main()
