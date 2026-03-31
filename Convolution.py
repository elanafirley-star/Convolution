import numpy as np

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
        self.filtres = np.random.randn(nb_filtres, nb_canaux_init, kernelsize, kernelsize) * 0.1

        # calcul auto de la taille après les convolutions pour le Flatten
        self.taille_flat = self._calculer_taille_aplatie(image_shape)

        # Création dynamique des couches Denses
        self.poids_dense = []  # On va stocker des tuples (W, b)

        # La première couche dense part de la taille "flat"
        taille_precedente = self.taille_flat

        # On boucle sur la liste fournie (ex: [128, 64])
        for nb_neurones in couches_denses:
            W = np.random.randn(taille_precedente, nb_neurones) * 0.1
            b = np.zeros(nb_neurones)
            self.poids_dense.append((W, b))
            taille_precedente = nb_neurones  # La sortie devient l'entrée de la suivante

        # Dernière couche : on connecte au nombre de classes (ex: 10)
        self.W_final = np.random.randn(taille_precedente, nb_classes) * 0.1
        self.b_final = np.zeros(nb_classes)




    def convolution(self, padding, stride, nb_filtres, kernelsize, image, biais):
        """
                Applique un filtre sur l'image
                :param padding: nb de pixels ajoutés sur le bord de l'image (soit 0 ou copie du pixel voisin)
                :param stride: de combien on décale
                :param nb_filtres:
                :param kernelsize: taille du noyau
                :return: un nouvelle image de taille réduite ou non
                """

        #géstion de l'image d'entree: 2d : (H, L) vs 3d : (C, H, L)
        if image.ndim == 3:
            image = image[0]  # prendre 1er canal si c'est du 3d

        # on ajoute les cases autour de l'image (padding)
        if padding > 0:
            image_ajout_pads = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        else:
            image_ajout_pads = image

        # on récupère les dimensions de l'images avec les cases en plus
        image_h, image_l = image_ajout_pads.shape

        # on calcule la taille de l'image de sortie
        sortie_h = (image_h - kernelsize) // stride + 1
        sortie_l = (image_l - kernelsize) // stride + 1

        # on initialise la matrice de sortie avec les biais
        # sortie = np.full((sortie_h, sortie_l), biais)
        #la boucle tourne sur nb_filtres donc la matrice de sortie doit avoir 3 dim
        sortie = np.full((nb_filtres, sortie_h, sortie_l), biais) #je me suis permis de corriger car j'en avais besoin pour la forward

        for f in range(nb_filtres):
            filtre_actuel = self.filtres[f]

            for y in range(sortie_h):
                for x in range(sortie_l):
                    #on délimite la zone qu'on va regarder
                    y_start = y * stride
                    y_end = y_start + kernelsize
                    x_start = x * stride
                    x_end = x_start + kernelsize

                    #on en fait une matrice à part entière
                    fenetre = image_ajout_pads[y_start:y_end, x_start : x_end]
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
        x = np.array(image_entree) #im brute
        if x.ndim == 2: x = x[np.newaxis, :, :]

        for i in range(self.nb_couches_convolution):
            x = self.convolution(self.type_padding, self.pas, self.nb_filtres, self.taille_noyau, x, biais) #x devient un volume de caractéristiques (Feature Map)
            x = self.activation(x, type_fonction="ReLu") #x est nettoyé (les négatifs deviennent 0)
            x = self.pooling(self.pas, self.pool_size, x, type_pooling="Max") #x est réduit

        self.cache['avant_flatten'] = x
        #applatissement
        x_flat = x.flatten() #apres la boucle, notre image est ransformée en vecteur
        self.cache['activation_dense'] = [x_flat] #on stocke l'entrée du block dense
        x = x.flat
        self.cache['z_dense'] = [] # z= W*x + b
        for W, b in self.poids_dense:
            x = np.dot(x, W) + b #x devient un vecteur de "scores"
            self.cache['z_dense'].append(x) #on stocke le score avant activation
            x = self.activation(x, type_fonction="ReLu")  # Activation entre chaque couche
            self.cache['activations_dense'].append(x) #on stocke le score activé

        #couche de sortie
        score_final = np.dot(x, self.W_final) + self.b_final #x est un veteur de la taille de nos classes (ex si on a dix animaux), il represente maintenant une proba que l'image apartienne à une catégorie
        self.cache['z_final'] = score_final
        return self.activation(score_final, type_fonction="Softmax")

    def backward_dense(self,erreur,flatten,poids,biais,lr):
        #calcul du gradient par rapport aux poids (np.outer : calcule le produit de 2 vecteurs)
        erreur_poids = np.outer(flatten,erreur)

        #calcul du gradient par rapport aux biais
        erreur_biais = erreur

        #calcul du gradient
        gradient = np.dot(erreur,poids.T)
        nv_poids = poids - lr*erreur_poids
        nv_biais = biais - lr*erreur_biais

        return gradient, nv_poids, nv_biais

    def backward(self, erreur,lr):
        #entreée de la dernière couche = dernière activation srtockée
        entree_finale = self.cache['activations_dense'][-1]
        grad_W = np.outer(entree_finale,erreur)
        grad_b = erreur

        #on calcule l'erreur qu'on va renvoyer à la couche d'avant
        erreur_courante = np.dot(erreur,self.W_final.T)

        self.W_final -= lr*grad_W
        self.b_final -= lr*grad_b
        #on parcourt les couches en partant de la fin
        for i in reversed(range(len(self.poids_dense))):
            W, b = self.poids_dense[i]
            entree_couche = self.cache['activations_dense'][i]
            sortie_couche = self.cache['activations_dense'][i + 1]
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

        return gradient_image

def main():
    """
    Définition des paramètres
    Rentrer les données d'entraînement
    Entraînement :
     - Forward
     - Calcul de l'errreur
     - Backward
     - Recalcul des poids
    Sauvegarde du modèle
    """

if __name__ == "__main__":
    main()

