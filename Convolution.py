class Convolution():
    def __init__(self,nb_filtres,pas,padding,nb_convolution,kernelsize,nb_pooling,image):

        self.nb_filtres = nb_filtres
        self.pas = pas
        self.type_padding = padding
        self.nb_couches_convolution = nb_convolution
        self.taille_noyau = kernelsize
        self.nb_couches_pooling = nb_pooling
        self.image = image
        pass

    def convolution(self,padding,stride,nb_couches,nb_filtres,kernelsize,image):
        """
        Applique un filtre sur l'image
        :param padding: nb de pixels ajoutés sur le bord de l'image (soit 0 ou copie du pixel voisin)
        :param stride: de combien on décale
        :param nb_couches: nb couches de convolution
        :param nb_filtres:
        :param kernelsize: taille du noyau
        :return: un nouvelle image de taille réduite ou non
        """

        pass

    def activation(self,image,type_fonction = "ReLu"):
        """
        Applique une transformation sur l'image (nettoyer l'image en ne gardant que les signes positifs forts)
        :param type_fonction: ReLu ou Soft max
        :return: image de même taille avec les valeurs transformées
        """
        pass
    def pooling(self,stride,taille,image, type_pooling):
        """
        Réduit la dimension de l'image
        :param stride: pas
        :param taille: taille de la fenêtre finale
        :param image: image retournée par la fonction d'activation
        :param type_pooling: Max ou Average pooling
        :return: image de taille réduite
        """
        pass

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

