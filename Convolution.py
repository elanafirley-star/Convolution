import numpy as np

class Convolution():
    def __init__(self,nb_filtres,pas,padding,nb_convolution,kernelsize,pool_size, nb_pooling,image):

        self.nb_filtres = nb_filtres
        self.pas = pas
        self.type_padding = padding
        self.nb_couches_convolution = nb_convolution
        self.taille_noyau = kernelsize
        self.nb_couches_pooling = nb_pooling
        self.image = image
        self.pool_size = pool_size

        self.filtres = np.random.randn(nb_filtres, kernelsize, kernelsize) #paquet de filtres aléatoires pour eviter que le réseau oublie ce qu'il apprend à chq fois
        pass

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

        # on ajoute les cases autour de l'image
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



    def forward(self, image_entree, biais):
        """ Réalise la passe en avant (Forward Pass). Image traverse  la convolution, l'activation ReLU
        et enfin le pooling pour extraire les caractéristiques reconues pas le CNN.

    params:
        image_entree (np.array): Matrice 2D (H, L) représentant l'image brute.
        biais (float): Valeur ajoutée après la convolution pour décaler l'activation.

    Returns:
        np.array: L'image / feature map transformée et réduite
    """
        # convolution
        conv_out = self.convolution(padding=self.type_padding, stride=self.pas, nb_filtres=self.nb_filtres,kernelsize=self.taille_noyau,image=image_entree,biais=biais)

        #activation (ReLU pour les couches caches)
        activation_out = self.activation(conv_out, type_fonction="ReLu")

        # pooling
        pooling_out = self.pooling(stride=self.pas,pool_size=self.pool_size,image=activation_out,type_pooling="Max")

        return pooling_out

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

