import time
import math
import cv2
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

directory = "/Users/khanfar/detection" #repertoire sauvegarde image

#webcam : 0 = computer, 1 = external
cap = cv2.VideoCapture(0)

print("Programme de detection des rayonnements ionisants (muons,gamma,...)")
pixX = cap.get(3)
pixY = cap.get(4)
canal = 0 #canal rgb image
print("Camera : (%d,%d) pixels" % (pixX,pixY))
print("Canal RGB : %d" % canal)

###############################################################
#### 1ere etape : caracterisation du bruit de fond (seuil) ####
###############################################################

calibrage=input("Entrer le temps de mesure du bruit de fond (recommandation : minimum 60 secondes) : ")
calibrage=float(calibrage)

acquisition_time = calibrage #seconds
canal = 1 #canal rgb image (0 1 2)
summax=0
bkg  = np.array([])
t0 = time.process_time()
threshold_lum = 0 #a ajuster en fonction de la correction d'auto-luminosite sur certaines cameras

while((time.process_time() - t0) < acquisition_time):
    
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    H = []
    H = np.array(frame)
    
    if (np.max(H[:,:,canal])>threshold_lum):
        #recherche du pixel max de l'image 
        pixmax = 0
        xmax=0
        ymax=0
        for i in range(10,int(pixY-10)): #retirer les pixels sur les bords (dependant de la taille de la zone autour du pixel max)
            for j in range (10,int(pixX-10)):
                if H[i,j,canal]>pixmax:
                    pixmax = H[i,j,canal]
                    xmax = i
                    ymax = j
                    
        #calcul de la somme du signal dans un cluster de 10x10 autour du pixel max
        if(pixmax>0):     
            imin=xmax-10
            imax=xmax+10
            jmin=ymax-10
            jmax=ymax+10
            H2 = H[imin:imax,jmin:jmax,canal]
            a=np.sum(H2)
            bkg=np.append(bkg,a)
            print('Valeur signal %d pour position (%d,%d)' % (a,xmax,ymax))
            if (int(a)>summax):
                summax=a
        
print("Moyenne du bruit de fond est : %f" % np.mean(bkg))
print("Ecart-type du bruit de fond est : %f" % np.std(bkg))

cut_bkg = math.ceil(np.mean(bkg)+5*np.std(bkg)) #seuil de rejection du bruit de fond (5x ecart-type = 99.99994%)
print('Le seuil de detection (moyenne+5xsigma) est : %d' % cut_bkg)

#Plot bkg distribution
plt.figure()
plt.hist(bkg)
plt.xlabel("Valeur bruit de fond")
plt.show()

if summax>100:
   print("Attention valeur du seuil trop elevee, verifier le masquage de la camera")

print('Fin de caracterisation du bruit de fond')

##############################################
#### 2eme etape : mesure des rayonnements ####
##############################################

print("Entrer le temps de mesure (en secondes) :")
temps=input()
temps=float(temps)

acquisition_time = temps #seconds
detection = 0 #tag fichier jpg

t0 = time.process_time()

while((time.process_time() - t0) < acquisition_time):
    
    ret, frame = cap.read()   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    H = []
    H = np.array(frame)
    
    if (np.max(H[:,:,canal])>threshold_lum):
        #recherche position pixel max
        pixmax = 0
        xmax=0
        ymax=0
        for i in range(10,int(pixY-10)): #retirer les pixels sur les bords (dependant de la taille de la zone autour du pixel max)
            for j in range (10,int(pixX-10)):
                if H[i,j,canal]>pixmax:
                    pixmax = H[i,j,canal]
                    xmax = i
                    ymax = j
                    
        if(pixmax>0):    
            imin=xmax-10
            imax=xmax+10
            jmin=ymax-10
            jmax=ymax+10
            H2 = H[imin:imax,jmin:jmax,canal]
            
            #Si signal > coupure de selection, on enregistre l'image
            if(np.sum(H2)>cut_bkg):
                print('Valeur signal %d pour position (%d,%d)' % (pixmax,xmax,ymax))
                plt.imshow(H2)
                plt.axis('off')
                detection+=1
                outputname = '%s/%s%s.jpg' % (directory, 'detectionsignal',detection)
                plt.savefig(outputname,bbox_inches='tight',pad_inches=0)
                plt.close()
                
                print("Evenements : %d" % detection)

cpm=(detection/temps)*60
print("%d evenements ont ete detectes en %d secondes. Soit %f coups par minute" % (detection,(time.process_time() - t0),cpm))
print('Fin de la mesure')
