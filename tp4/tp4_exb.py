
# Generals
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict


def simple_reg_lin(exo5):
	X = np.array ([[5.5], [6.0], [6.5], [6.0], [5.0], [6.5], [4.5], [5]])
	# ~ X = X.reshape (-1, 1)
	y = np.array ([[420], [380], [350], [400], [440], [380], [450], [420]])
	print ("X\n", X)
	print ("y\n", y)
	
	if (exo5):	
		reg = LinearRegression(fit_intercept=False).fit(X, y)
	else:    	
		reg = LinearRegression().fit(X, y)
	y_pred = reg.predict(X)
	print ("\ny_pred\n", y_pred)
	print ("mean_squared_error\n", mean_squared_error(y, y_pred))
	print ("score\n", reg.score(X, y))
	print ("taux d'erreur\n", 1 - reg.score(X, y))
	print ("Parametres alpha et beta:" +str(reg.coef_) + str(reg.intercept_))
	#~ print ("\ncoef\n", reg.coef_)	# -> coef x de la droite ax+b (regression) (alpha)
	#~ print ("intercept\n", reg.intercept_) # -> origine b de ax+b (regression lineaire)	(beta)
	
	x0, x1 = 4.0, 7.0 # Les valeurs de X sont comprises entre ces coordonnées.
	coef = reg.coef_[0][0];
	print ("coef : ", coef)
	if (exo5):
		intercept = reg.intercept_
	else:
		intercept = reg.intercept_[0];
	print ("origine : ", intercept)
	
	# y = ax + b
	y0_reg = y_pred[0][0]
	y1_reg = y_pred[7][0]
	y0 = coef*x0 + intercept
	y1 = coef*x1 + intercept
	
	plt.plot ([x0, x1], [y0, y1], c='r')
	plt.plot ([x0, x1], [y0_reg, y1_reg], c='g')
	#plt.plot (X, y_pred, c='g')
	plt.scatter (X, y, c='b')
	plt.show()

def euc_ex1(C, H):
	plt.xlabel ("circonference")
	plt.ylabel ("hauteur")
	plt.title("Donnees Eucalyptus")
	plt.scatter (C, H, c='b')
	plt.show()

def euc_ex3a (Cold, C, H, reg):
	# ~ x0, x1 = 10.0, 30.0 # Les valeurs de H sont comprises entre ces coordonnées.
	x0, x1 = Cold.min(), Cold.max() # Les valeurs de H sont comprises entre ces coordonnées.
	coef = reg.coef_[0];
	print ("coef : ", coef)
	intercept = reg.intercept_
	print ("origine : ", intercept)
	# y = ax + b
	y0 = coef*x0 + intercept
	y1 = coef*x1 + intercept
	plt.plot ([x0, x1], [y0, y1], c='r')
	plt.scatter (C, H, c='b')
	plt.xlabel ("circonference")
	plt.ylabel ("hauteur")
	plt.title("Donnees Eucalyptus")
	plt.show()

def euc_ex3b (C, H, reg, H_pred):
	# cross_val_predict returns an array of the same size as `C` where each entry
	# is a prediction obtained by cross validation:
	predicted = cross_val_predict(reg, C, H, cv=10)
	
	# ~ fig, ax = plt.subplots()
	# ~ ax.scatter(C, predicted, edgecolors=(0, 0, 0))
	# ~ ax.plot([C.min(), C.max()], [C.min(), C.max()], 'k--', lw=4)
	# ~ ax.set_xlabel('Measured')
	# ~ ax.set_ylabel('Predicted')
	
	plt.plot([H.min(), H.max()], [H.min(), H.max()], lw=2, c='r')
	plt.scatter(H, predicted, edgecolors=(0, 0, 0))
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.show()	

def euc_ex3(Cold, H):
	C = []
	for i in range(0, len(Cold)):
		C.append([Cold[i]])
	# ~ print ("C\n", C)
	
	reg = LinearRegression().fit(C, H)
	H_pred = reg.predict(C)
	print ("\nH_pred\n", H_pred)
	print ("mean_squared_error\n", mean_squared_error(H, H_pred))
	print ("score\n", reg.score(C, H))
	print ("taux d'erreur\n", 1 - reg.score(C, H))
	print ("Parametres alpha et beta: " +str(reg.coef_) + " " + str(reg.intercept_))
	#~ print ("\ncoef\n", reg.coef_)	# -> coef x de la droite ax+b (regression) (alpha)
	#~ print ("intercept\n", reg.intercept_) # -> origine b de ax+b (regression lineaire)	(beta)
	
	#print ("hauteur de eucaplyptus de circonference 22.8 : ", C_pred[np.argwhere(C_pred==22.8)])
	print ("hauteur de eucaplyptus de circonference 22.8 : ", reg.predict(22.8))
	
	# ~ euc_ex3a(Cold, C, H, reg)
	euc_ex3b(C, H, reg, H_pred)

def euc_ex4a (Cold, C, H, reg):
	x0, x1 = Cold.min(), Cold.max() # Les valeurs de H sont comprises entre ces coordonnées.
	print ("x0\n", x0)
	print ("x1\n", x1)
	coef1 = reg.coef_[0];
	print ("coef1 : ", coef1)
	coef2 = reg.coef_[1];
	print ("coef2 : ", coef2)
	intercept = reg.intercept_
	print ("origine : ", intercept)
	# y = ax + b
	y0 = coef1*x0 + intercept + coef2*sqrt(x0)
	y1 = coef1*x1 + intercept + coef2*sqrt(x1)
	plt.plot ([x0, x1], [y0, y1], c='r')
	plt.scatter (Cold, H, c='b')
	plt.xlabel ("circonference")
	plt.ylabel ("hauteur")
	plt.title("Donnees Eucalyptus")
	plt.show()


def euc_ex4 (Cold, H):
	C = []
	for i in range(0, len(Cold)):
		C.append( [Cold[i], sqrt(Cold[i])] )
		
	print ("C\n", C[:4])
	
	reg = LinearRegression().fit(C, H)
	H_pred = reg.predict(C)
	print ("\nH_pred\n", H_pred)
	print ("mean_squared_error\n", mean_squared_error(H, H_pred))
	print ("score\n", reg.score(C, H))
	print ("taux d'erreur\n", 1 - reg.score(C, H))
	print ("Parametres alpha et beta: " +str(reg.coef_) + " " + str(reg.intercept_))
	
	# ~ euc_ex4a(Cold, C, H, reg)
	# ~ euc_ex3b (C, H, reg, H_pred)
	
	from mpl_toolkits.mplot3d import Axes3D
	a = np.array([0,1,2,3,4,5,6])
	b= np.sqrt(a)
	c = np.log10(a+b+1)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(a,b,c)
	ax.plot(a,b,np.log10(c),'r')
	

def eucalyptus ():
	fichier = "eucalyptus.txt"
	donnees = np.loadtxt (fichier)
	# ~ print (donnees)
	# Explication c
	C = donnees[:,1]
	print ("C\n", C)
	# Cible h
	H = donnees[:,0]
	print ("H\n", H)
	
	# ~ euc_ex1(C, H)
	# ~ euc_ex3(C, H)
	euc_ex4(C, H)
	
	

# ~ exo5 = False;
# ~ simple_reg_lin(exo5)
eucalyptus()
