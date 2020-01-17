
# Generals
import numpy as np
import matplotlib.pyplot as plt

def addOne(X):
	(n,d) = X.shape
	Xnew = np.zeros([n, d+1])
	Xnew[:,1:]=X
	Xnew[:,0]=np.ones(n)
	return Xnew

def RegLin (X, Y):
	Z = addOne(X)
	w = np.dot ( np.dot(np.linalg.inv( np.dot(np.transpose(Z), Z)), np.transpose(Z)), Y )
	return w
	

def vente_produit ():
	X = np.array ([[5.5], [6.0], [6.5], [6.0], [5.0], [6.5], [4.5], [5]])
	print (X.shape)
	Y = np.array ([[420], [380], [350], [400], [440], [380], [450], [420]])
	w = RegLin (X, Y)
	print ("w\n", w)

def eucalyptus():
	fichier = "eucalyptus.txt"
	donnees = np.loadtxt (fichier)
	
	# Explication X
	X = donnees[:,1]
	Xnew = np.zeros([len(X), 1])
	for i in range(0, len(X)):
		Xnew[i][0] = X[i]
	print ("Xnew\n", Xnew)
	
	# Cible Y
	Y = donnees[:,0]
	print ("Y\n", Y)
	
	print (Xnew.shape)
	w = RegLin (Xnew, Y)
	print ("w\n", w)
	
	

# ~ vente_produit()
eucalyptus ()

# ~ X = np.zeros([5,3]) # tableau de 0 de dimension 5x3
# ~ print ("X\n", X)
# ~ Y = np.ones([3,2]) # tableau de 1 de dimension 5x3
# ~ print ("Y\n", Y)



# ~ v = np.ones(3) # vecteur contenant trois 1
# ~ print ("v\n", v)
# ~ X[1:4,:2] = Y # remplacement d’une partie du tableau X
# ~ print ("X remplacé en partie par Y\n", X)
# ~ print (X.shape) # dimensions de X
# ~ print ("produit matriciel\n", np.dot(X,Y)) # produit matriciel
# ~ print ("produit matrice X et vecteur v\n", np.dot(X,v)) # produit de la matrice X et du vecteur v
# ~ print ("transposee X\n", np.transpose(X)) # transposée de X
# ~ #print ("inverse de X\n", np.linalg.inv(X)) # inverse de Z
