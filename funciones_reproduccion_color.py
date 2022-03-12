# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:28:09 2021

@author: Johan Cuervo
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import itertools

#%% Funciones generales para variables. 

#Guardar imagen con cv2
def imwrite(titulo,imagen,InColorSpace='RGB'):
    
    if(InColorSpace=='RGB'):
        if(np.max(imagen)>1):
            imagen/=255
    
    elif(InColorSpace=='Lab'):
        imagen=Lab2sRGB(imagen)
        
    elif(InColorSpace=='XYZ'):
        imagen=XYZ2RGB(imagen)
      
        
    elif(len(np.shape(imagen))==2 and InColorSpace=='GRAY'):
        imagen1= np.zeros((np.shape(imagen)[0],np.shape(imagen)[1],3))
        imagen1[:,:,0]=imagen
        imagen1[:,:,1]=imagen
        imagen1[:,:,2]=imagen
        imagen=imagen1
    
    else:
        print("Espacio de color no existente.")
        return
    
    imagen = np.array(imagen*255,dtype='uint8')
    imagen2= np.copy(imagen)
    imagen[:,:,0]=imagen2[:,:,2]
    imagen[:,:,2]=imagen2[:,:,0]
    cv2.imwrite(titulo,imagen)
    
def Write_Variable(nombre,variable):
    fichero = open('Resultados\Variables/'+nombre+'.pickle','wb')
    pickle.dump(variable,fichero)
    fichero.close()

def Read_Variable(nombre):
    fichero = open(nombre,'rb') 
    # Cargamos los datos del fichero
    lista_fichero = pickle.load(fichero)
    
    fichero.close()
    
    return lista_fichero

# condicionales de valores limites de imagenes despues de una transformación 
def recorte(im):
    im[np.where(im > 1)] = 1
    im[np.where(im < 0)] = 0
    
    return im

# offset de imagenes para transformaciones logaritmicas fijando valor minimo a 1/255
def offset(im):
    im[np.where(im <= 1/255)] = 1/255
    
    return im
    
#%% Funciones para Graficas

# Se grafican las firmas espectrales  Y= Promedios de los parches X = longitud de onda

def grafica_firmas_espectrales(espectro,ecualizadoN,colorN):
    plt.figure(figsize=(15,8))
    for i in range(len(ecualizadoN)):
        plt.plot(espectro,ecualizadoN[i],color=colorN[i]) 
    
    plt.title('Firmas espectrales')
    plt.xlabel('\lambda[nm]')
    plt.ylabel('Reflactancia [%]')
    plt.ylim(0,1.02)
    plt.xlim(400, 720)
    plt.show()
    plt.grid()
    
# funcion para mostrar imagenes con matplotlib  especificando formato de entrada y luego cambiado a RGB para ser visualizado correctamente
def imshow(titulo, imagen,InColorSpace='RGB'):
    
    if(InColorSpace=='RGB'):
        if(np.max(imagen)>1):
            imagen/=255
    
    elif(InColorSpace=='Lab'):
        imagen=Lab2sRGB(imagen)
        
    elif(InColorSpace=='XYZ'):
        imagen=XYZ2RGB(imagen)
      
        
    elif(len(np.shape(imagen))==2 and InColorSpace=='GRAY'):
        imagen1= np.zeros((np.shape(imagen)[0],np.shape(imagen)[1],3))
        imagen1[:,:,0]=imagen
        imagen1[:,:,1]=imagen
        imagen1[:,:,2]=imagen
        imagen=imagen1
    
    else:
        print("Espacio de color no existente.")
        return
    
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis('off')
    plt.show()
    
    
#%% Funciones de conversion de espacios de color

# error de reproduccion distancia euclidea  pixel por pixel de cada parche
# y promedio de error por parche para multiples imagenes reconstruidas


def sRGB2Lab(imagen_sRGB):
    
    shape_imag=np.shape(imagen_sRGB)
    
    if(np.max(imagen_sRGB)>1):
        imagen_sRGB=np.reshape(imagen_sRGB,(-1,3))/255
    
    else:
        imagen_sRGB=np.reshape(imagen_sRGB,(-1,3))
        
        
    RGB2XYZ= np.array([[0.4124, 0.3576, 0.1805],
             [0.2126, 0.7152, 0.0722],
             [0.0193, 0.1192, 0.9505]])
    

    imagen_RGB = np.where(imagen_sRGB<=0.04045,imagen_sRGB/12.92,((imagen_sRGB+0.055)/(1.055))**2.4)
    XYZ = np.dot(RGB2XYZ,imagen_RGB.T)
    XYZn=np.array([0.9505 ,1 ,1.089])
    
    XYZ=np.divide(XYZ,np.reshape(XYZn,(3,1)))
    Fxyz=np.where(XYZ>0.008856,XYZ**(1/3),7.787*XYZ+16/116)

    
    Lab = np.zeros_like(Fxyz)
    Lab[0,:]=116*np.copy(Fxyz[1,:])-16
    Lab[1,:]=500*np.copy(Fxyz[0,:]-Fxyz[1,:])
    Lab[2,:]=200*np.copy(Fxyz[1,:]-Fxyz[2,:])
    
    im_Lab = np.reshape(Lab.T, shape_imag)
    
    return im_Lab


def Lab2sRGB(imagen_CIE):
    
    imagen_CIELAB=np.copy(imagen_CIE)
    shape_imag=np.shape(imagen_CIELAB)
    imagen_CIELAB=np.reshape(imagen_CIELAB,(-1,3))
    
    imagen_CIELAB[:,0]=(imagen_CIELAB[:,0]+16)/116
    imagen_CIELAB[:,1]=imagen_CIELAB[:,0]+imagen_CIELAB[:,1]/500
    imagen_CIELAB[:,2]=imagen_CIELAB[:,0]-imagen_CIELAB[:,2]/200
    
    imagen_CIELAB = np.where(imagen_CIELAB>0.206893,imagen_CIELAB**3,(imagen_CIELAB-16/116)/7.787)
    YXZn=np.array([1,0.9505  ,1.089])
    
    YXZ = imagen_CIELAB*YXZn
    
    YXZ2RGB= np.array([[ -1.5372,3.2406, -0.4986],
                       [ 1.8758,-0.9689, 0.0415],
                       [ -0.2040,0.0557, 1.0570]])
    
    RGB = recorte(np.dot(YXZ2RGB,YXZ.T))
    sRGB = np.where(RGB<=0.0031308,12.92*RGB,1.055*np.power(RGB,1/2.4)-0.055)
    
    im_sRGB = np.round(np.reshape(sRGB.T, shape_imag)*255)/255
    return im_sRGB

def XYZ2Lab(imagen_XYZ):
    
    shape_imag=np.shape(imagen_XYZ)

    XYZn=np.array([0.9505 ,1 ,1.089])
    
    XYZ=np.divide(np.reshape(imagen_XYZ,(-1,3)).T,np.reshape(XYZn,(3,-1)))
    Fxyz=np.where(XYZ>0.008856,XYZ**(1/3),7.787*XYZ+16/116)

    
    Lab = np.zeros_like(Fxyz)
    
    Lab[0,:]=116*np.copy(Fxyz[1,:])-16
    Lab[1,:]=500*np.copy(Fxyz[0,:]-Fxyz[1,:])
    Lab[2,:]=200*np.copy(Fxyz[1,:]-Fxyz[2,:])
    
    im_Lab = np.reshape(Lab.T, shape_imag)
    
    return im_Lab

def XYZ2RGB(imagen_XYZ):
    
    shape_imag=np.shape(imagen_XYZ)

    RGB = recorte(np.dot(XYZ2RGB,np.reshape(imagen_XYZ,(3,-1)).T))
    sRGB = np.where(RGB<=0.0031308,12.92*RGB,1.055*np.power(RGB,1/2.4)-0.055)
    im_RGB=np.reshape(sRGB,shape_imag)
    
    
    return im_RGB
   
#%%Lectura de mascaras y colocación en una lista(cant mascaras) 

#Funcion para leer y agrupar las mascaras de cada parche extraidas en una carpeta
def ext_mascaras(carpeta, lista):
    mascaras=[]
    for nombre in sorted(lista):
        a=cv2.cvtColor(cv2.imread(carpeta+"/"+nombre), cv2.COLOR_BGR2GRAY) #lectura de imagen, transformacion a escala de grises
        [T, mascara] = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #binarizar mascara con algotirmo de OTSU
        mascaras.append(np.array(mascara))  #Se agrega la mascara a una lista de arrays
    return mascaras


#Lectura de imagen multiespectrales dada la carpeta y la lista 
def Read_Multiespectral_imag(carpeta, lista):
    imagenespatron=[]
    
    for nombre in sorted(lista):
        imagen=cv2.cvtColor(cv2.imread(carpeta+"/"+nombre), cv2.COLOR_BGR2GRAY)#cargamos imagenes multiespectrales en escala de grises
        imagenespatron=np.concatenate((imagenespatron,np.squeeze(np.reshape(imagen,(1,-1)))),axis=0) #se convierte la imagen en una columna y se concatena con las demas del espectro
    shape_imag=np.shape(imagen)
    imagenespatron=imagenespatron.reshape(len(lista),-1)#se redimensiona a  Filas * N imagenes multiespectrales filas de pixeles de las imagenes espectrales
    return imagenespatron,shape_imag

#Lectura del espectro de la imagen dada en el nombre de la imagen.
def Read_espectros_Imag(lista):
    espectro=[]
    for nombre in sorted(lista):
        a=nombre.find('(')+1
        espectro.append(int(nombre[a:a+3]))
    return espectro

#Crea una lista de pixeles de color check, de acuerdo a la cantidad de pixeles de la mascara. 
def Ideal_Color_Patch_pixel(color_check,mascaras):
    color_ipmask=[[0, 0, 0]]
    for i,mascara in enumerate(mascaras): #se recorren las mascaras
        N= np.shape(np.where(mascara==255))[1]
        color=np.concatenate((color_check[i][0]*np.ones(N),color_check[i][1]*np.ones(N),color_check[i][2]*np.ones(N)),axis=0)#un vector columna con los valores RGB ideales de cada parche N pixeles de parche
        color=color.reshape(3,-1).T #redimensiona
        color_ipmask= np.concatenate((color_ipmask,color),axis=0)#concatena el color ideal de los 24 parches
    color_ipmask=color_ipmask[1:,:]# se borra la primer fila que son 0
    return color_ipmask


# Se calculan los pesos para aproximar la media del parche neutral escogido. Al valor del color check
def Pesos_ecualizacion(imagenes_patron, mascara, valor_ideal=243,shape_imag=(480,640)):
    promedios=[]
    for i in range(len(imagenes_patron)):
        im = np.reshape(imagenes_patron[i],shape_imag)
        parche = im[np.where(mascara==255)]
        prom = np.mean(parche)
        promedios.append(prom)
    
    Pesos_ecu = np.divide(valor_ideal*np.ones(len(promedios)),np.array(promedios))
    return Pesos_ecu

#Función que ecualiza los parches para obtener los pesos de ecualización de las imagenes
def ecualizacion(prom,parche,brillo_parche):
    pesos=np.divide(brillo_parche*np.ones((1,len(prom[0]))),prom[parche-1,:])#pesos para cada espectro utilizando de referencia un parche neutral y su color ideal
    ecualizado=prom*pesos #se ecualiza los promedios 
    ecualizado[np.where(ecualizado>255)]=255 #los valores saturados se igualan a 255
    ecualizadoN= ecualizado/255 #Normalizacion
    
    
    return [pesos, ecualizadoN]

#Extrae los valores de los parches de la carta de color de la imagenEsecncialmente en RGB, sirve para cualquier formato de imagen
def Patch_Ext_Pixels(Irecons_RGB,mascaras):
    parches_r=[]
    parches_g=[]
    parches_b=[]
    
    for i in range(len(mascaras)):
        R,G,B=Irecons_RGB[:,:,0],Irecons_RGB[:,:,1],Irecons_RGB[:,:,2]
        parte=R[np.where(mascaras[i]==255)]
        parches_r= np.concatenate((parches_r,parte))
        parte=G[np.where(mascaras[i]==255)]
        parches_g= np.concatenate((parches_g,parte))
        parte=B[np.where(mascaras[i]==255)]
        parches_b= np.concatenate((parches_b,parte))
        
    parches_rgb = np.zeros((len(parches_r),3))
    parches_rgb[:,0] = parches_r
    parches_rgb[:,1] = parches_g
    parches_rgb[:,2] = parches_b
    return parches_rgb


#Igual que la funcion Patch_ext_pixels peroesta vez calcula la media de todos los pixeles de cada parche, diferente a enviar todos los pixeles
def Patch_Ext_Pixels_Mean(Irecons_RGB,mascaras):
    parches_rgb=[]
    
    for i in range(len(mascaras)):
        R,G,B=Irecons_RGB[:,:,0],Irecons_RGB[:,:,1],Irecons_RGB[:,:,2]
        parte=R[np.where(mascaras[i]==255)]
        parches_rgb.append(np.mean(parte))
        parte=G[np.where(mascaras[i]==255)]
        parches_rgb.append(np.mean(parte))
        parte=B[np.where(mascaras[i]==255)]
        parches_rgb.append(np.mean(parte))
    
    return np.reshape(parches_rgb,(-1,3))


# parches pixeles Imagen de infrarrojo cercano tomada con la imagen numero 
def N_IN(Irecons,mascaras):
    parches=[]
    for i in range(len(mascaras)):
        parte=Irecons[np.where(mascaras[i]==255)]
        parches= np.concatenate((parches,parte))
        
    return np.reshape(parches,(1,-1))


## Esta función calcula la distancia euclidea de multiples imagenes, para cada parche. Se necesitan las imagenes en una lista,
## las mascaras de las imagenes y el color check de referencia

def Error_de_reproduccion(imagenes, mascaras, color_check):
    error= []
    for imagen in imagenes:
        imagen=imagen.reshape(-1,3)
        for i,mascara in enumerate(mascaras):
            indices=np.where(mascara.reshape(-1,)==255)
            dif = imagen[indices] -color_check[i]
            DistEucl= np.sqrt(np.sum(np.power(dif,2),axis=1))
            error.append(np.mean(DistEucl))
    return np.reshape(error,(-1,len(mascaras)))





#%% Funciones CCM para una imagen
# Color Correction matriz linear
def CCM_Linear(im_RGB,colorn,mascaras,shape_imag=(480,640,3)):
    entrada = Patch_Ext_Pixels(im_RGB, mascaras).T
    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= colorn.T
    
    PseudoINV= np.linalg.pinv(entrada)
    Ccm= np.dot(colorn,PseudoINV)
    
    rgb= np.reshape(im_RGB,(-1,3)).T
    rgb= np.concatenate((rgb,np.ones((1,np.shape(rgb)[1]))))
    rgb= np.dot(Ccm,rgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    #im_rgb = recorte(im_rgb)
    return im_rgb, Ccm

# Color Correction matriz Compound
def CCM_Compound(im_RGB,colorn,mascaras,shape_imag=(480,640,3)):
    entrada = Patch_Ext_Pixels(im_RGB, mascaras).T
    #entradaN = N_IN(N, mascaras)
    #entrada = np.concatenate((entrada,entradaN))
    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= np.log(colorn.T/255)
    
    PseudoINV= np.linalg.pinv(entrada)
    
    Ccm= np.dot(colorn,PseudoINV)
    rgb= np.reshape(im_RGB,(-1,3)).T
    #rgbn= np.concatenate((rgb,np.reshape(N,(1,-1))))
    rgb= np.concatenate((rgb,np.ones((1,np.shape(rgb)[1]))))
    
    lnrgb= np.dot(Ccm,rgb)
    rgb = np.exp(lnrgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, Ccm

# Color Correction matriz Logarithm
def CCM_Logarithm(im_RGB,colorn,mascaras,shape_imag=(480,640,3)):
    entrada = np.log(offset(Patch_Ext_Pixels(im_RGB, mascaras).T))
    #entradaN = np.log(offset(N_IN(N, mascaras)))
    #entrada = np.concatenate((entrada,entradaN))
    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= colorn.T/255
    
    PseudoINV= np.linalg.pinv(entrada)
    
    Ccm= np.dot(colorn,PseudoINV)
    lnrgb= np.log(offset(np.reshape(im_RGB,(-1,3)).T))
    #lnrgbn = np.concatenate((lnrgb,np.log(offset(np.reshape(N,(1,-1))))))
    lnrgb= np.concatenate((lnrgb,np.ones((1,np.shape(lnrgb)[1]))))
    
    rgb= np.dot(Ccm,lnrgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, Ccm

# Color Correction matriz Polynomial Con NIR
def CCM_Polynomial_N(im_RGB,N,colorn,mascaras,shape_imag=(480,640,3)):
    
    entrada = Patch_Ext_Pixels(im_RGB, mascaras).T
    entradaN = N_IN(N, mascaras)
    entrada = np.concatenate((entrada,entradaN))
    
    R2=entrada[0,:]**2
    G2=entrada[1,:]**2
    B2=entrada[2,:]**2
    N2=entradaN**2
    RG=entrada[0,:]*entrada[1,:]
    RB=entrada[0,:]*entrada[2,:]
    RN=entrada[0,:]*entradaN
    GB=entrada[1,:]*entrada[2,:]
    GN=entrada[1,:]*entradaN
    BN=entrada[2,:]*entradaN
    
    entrada = np.concatenate((entrada,[R2]))
    entrada = np.concatenate((entrada,[G2]))
    entrada = np.concatenate((entrada,[B2]))
    entrada = np.concatenate((entrada,N2))
    entrada = np.concatenate((entrada,[RG]))
    entrada = np.concatenate((entrada,[RB]))
    entrada = np.concatenate((entrada,RN))
    entrada = np.concatenate((entrada,[GB]))
    entrada = np.concatenate((entrada,GN))
    entrada = np.concatenate((entrada,BN))
    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= colorn.T
    
    PseudoINV= np.linalg.pinv(entrada)
    
    entradaN= np.reshape(N,(1,-1))
    
    Ccm= np.dot(colorn,PseudoINV)
    rgb= np.reshape(im_RGB,(-1,3)).T
    rgb= np.concatenate((rgb,entradaN))
    R2=rgb[0,:]**2
    G2=rgb[1,:]**2
    B2=rgb[2,:]**2
    N2=entradaN**2
    RG=rgb[0,:]*rgb[1,:]
    RB=rgb[0,:]*rgb[2,:]
    RN=rgb[0,:]*entradaN
    GB=rgb[1,:]*rgb[2,:]
    GN=rgb[1,:]*entradaN
    BN=rgb[2,:]*entradaN
    
    rgb = np.concatenate((rgb,[R2]))
    rgb = np.concatenate((rgb,[G2]))
    rgb = np.concatenate((rgb,[B2]))
    rgb = np.concatenate((rgb,N2))
    rgb = np.concatenate((rgb,[RG]))
    rgb = np.concatenate((rgb,[RB]))
    rgb = np.concatenate((rgb,RN))
    rgb = np.concatenate((rgb,[GB]))
    rgb = np.concatenate((rgb,GN))
    rgb = np.concatenate((rgb,BN))
    rgb = np.concatenate((rgb,np.ones((1,np.shape(rgb)[1]))))
    
    rgb= np.dot(Ccm,rgb)
    im_rgb= np.reshape(rgb.T,shape_imag=(480,640,3))
    return im_rgb, Ccm, R2

# Color Correction matriz Polynomial
def CCM_Polynomial(im_RGB,colorn,mascaras,shape_imag=(480,640,3)):
    
    entrada = Patch_Ext_Pixels(im_RGB, mascaras).T
    
    R2=entrada[0,:]**2
    G2=entrada[1,:]**2
    B2=entrada[2,:]**2
    RG=entrada[0,:]*entrada[1,:]
    RB=entrada[0,:]*entrada[2,:]
    GB=entrada[1,:]*entrada[2,:]
    
    entrada = np.concatenate((entrada,[R2]))
    entrada = np.concatenate((entrada,[G2]))
    entrada = np.concatenate((entrada,[B2]))
    entrada = np.concatenate((entrada,[RG]))
    entrada = np.concatenate((entrada,[RB]))
    entrada = np.concatenate((entrada,[GB]))

    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= colorn.T
    
    PseudoINV= np.linalg.pinv(entrada)
    
    
    Ccm= np.dot(colorn,PseudoINV)
    rgb= np.reshape(im_RGB,(-1,3)).T
    R2=rgb[0,:]**2
    G2=rgb[1,:]**2
    B2=rgb[2,:]**2
    RG=rgb[0,:]*rgb[1,:]
    RB=rgb[0,:]*rgb[2,:]
    GB=rgb[1,:]*rgb[2,:]

    
    rgb = np.concatenate((rgb,[R2]))
    rgb = np.concatenate((rgb,[G2]))
    rgb = np.concatenate((rgb,[B2]))
    rgb = np.concatenate((rgb,[RG]))
    rgb = np.concatenate((rgb,[RB]))
    rgb = np.concatenate((rgb,[GB]))
    rgb = np.concatenate((rgb,np.ones((1,np.shape(rgb)[1]))))
    
    rgb= np.dot(Ccm,rgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    return im_rgb, Ccm

#%% FUNCIONES CCM Para multiples imagenes
# Color Correction matriz linear
def CCM_Linear_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=',',names=range(1,7))
    datatrain= datatrain.to_numpy()
    entrada= datatrain[:,:3].T/255
    colorn= datatrain[:,3:]
    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= colorn.T/255
    
    PseudoINV= np.linalg.pinv(entrada)
    Ccm_Linear= np.dot(colorn,PseudoINV)
    
    return Ccm_Linear

# Color Correction matriz Compound
def CCM_Compound_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=',',names=range(1,7))
    datatrain= datatrain.to_numpy()
    entrada= datatrain[:,:3].T/255
    colorn= datatrain[:,3:]
    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= np.log(colorn.T/255)
    
    PseudoINV= np.linalg.pinv(entrada)
    
    Ccm_Compound= np.dot(colorn,PseudoINV)
    
    return Ccm_Compound

# Color Correction matriz Logarithm
def CCM_Logarithm_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=',',names=range(1,7))
    datatrain= datatrain.to_numpy()
    entrada= datatrain[:,:3].T/255
    colorn= datatrain[:,3:]
    entrada = np.log(offset(entrada))
    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= colorn.T/255
    
    PseudoINV= np.linalg.pinv(entrada)
    
    Ccm_Logatirhm= np.dot(colorn,PseudoINV)
    
    return Ccm_Logatirhm

# Color Correction matriz Polynomial
def CCM_Polynomial_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=',',names=range(1,7))
    datatrain= datatrain.to_numpy()
    entrada= datatrain[:,:3].T/255
    colorn= datatrain[:,3:]
    
    R2=entrada[0,:]**2
    G2=entrada[1,:]**2
    B2=entrada[2,:]**2
    RG=entrada[0,:]*entrada[1,:]
    RB=entrada[0,:]*entrada[2,:]
    GB=entrada[1,:]*entrada[2,:]
    
    entrada = np.concatenate((entrada,[R2]))
    entrada = np.concatenate((entrada,[G2]))
    entrada = np.concatenate((entrada,[B2]))
    entrada = np.concatenate((entrada,[RG]))
    entrada = np.concatenate((entrada,[RB]))
    entrada = np.concatenate((entrada,[GB]))

    entrada = np.concatenate((entrada,np.ones((1,np.shape(entrada)[1]))))
    colorn= colorn.T/255
    
    PseudoINV= np.linalg.pinv(entrada)
    
    
    Ccm_Polynomial= np.dot(colorn,PseudoINV)
    
    return Ccm_Polynomial

#%% CCM test 
# Color Correction matriz linear
def CCM_Linear_Test(im_RGB,Ccm):
    shape_imag=np.shape(im_RGB)
    rgb= np.reshape(im_RGB,(-1,3)).T
    rgb= np.concatenate((rgb,np.ones((1,np.shape(rgb)[1]))))
    rgb= np.dot(Ccm,rgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb

# Color Correction matriz Compound
def CCM_Compound_Test(im_RGB,Ccm):
    shape_imag=np.shape(im_RGB)
    rgb= np.reshape(im_RGB,(-1,3)).T
    rgb= np.concatenate((rgb,np.ones((1,np.shape(rgb)[1]))))
    
    lnrgb= np.dot(Ccm,rgb)
    rgb = np.exp(lnrgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb

# Color Correction matriz Logarithm
def CCM_Logarithm_Test(im_RGB,Ccm):
    shape_imag=np.shape(im_RGB)
    lnrgb= np.log(offset(np.reshape(im_RGB,(-1,3)).T))
    
    lnrgb= np.concatenate((lnrgb,np.ones((1,np.shape(lnrgb)[1]))))
    
    rgb= np.dot(Ccm,lnrgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


# Color Correction matriz Polynomial
def CCM_Polynomial_Test(im_RGB,Ccm):
    shape_imag=np.shape(im_RGB)
    rgb= np.reshape(im_RGB,(-1,3)).T
    R2=rgb[0,:]**2
    G2=rgb[1,:]**2
    B2=rgb[2,:]**2
    RG=rgb[0,:]*rgb[1,:]
    RB=rgb[0,:]*rgb[2,:]
    GB=rgb[1,:]*rgb[2,:]

    
    rgb = np.concatenate((rgb,[R2]))
    rgb = np.concatenate((rgb,[G2]))
    rgb = np.concatenate((rgb,[B2]))
    rgb = np.concatenate((rgb,[RG]))
    rgb = np.concatenate((rgb,[RB]))
    rgb = np.concatenate((rgb,[GB]))
    rgb = np.concatenate((rgb,np.ones((1,np.shape(rgb)[1]))))
    
    rgb= np.dot(Ccm,rgb)
    im_rgb= np.reshape(rgb.T,shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb
#%% Generacion de imagen con parches ideales y reproducidos.

 
def comparacion_color_check(nombre,im_RGB,color_check_RGB,mascaras,carpeta=''):
    
    Grosor=2
    
    for i in range(4):
        fila=np.zeros((60,Grosor,3))
        for j in range(6):
            parchei= np.ones((60,60,3))*color_check_RGB[6*i+j,:]
            fila = np.concatenate((fila,parchei),axis=1)
            fila = np.concatenate((fila,np.zeros((60,Grosor,3))),axis=1)
        
        if i==0:
            imagen=np.zeros((Grosor,np.shape(fila)[1],3))
            
        imagen=np.concatenate((imagen,fila),axis=0)
        
        fila=np.zeros((60,Grosor,3))
        for j in range(6):
            
            parchei=im_RGB[np.where(255==mascaras[6*i+j])]*255
            Tam = 3600 / len(np.where(255==mascaras[6*i+j])[0])
            if Tam>1:
                parchecopy=parchei.copy()
                for k in range(int(Tam)):
                    parchei = np.concatenate((parchei,parchecopy))

            if len(parchei)>3600:
                parchei = parchei[:3600,:]
            parchei=np.reshape(parchei,(60,60,3)).astype(int)
            
            fila = np.concatenate((fila,parchei),axis=1)
            fila = np.concatenate((fila,np.zeros((60,Grosor,3))),axis=1)
            
        imagen=np.concatenate((imagen,fila),axis=0)   
        imagen=np.concatenate((imagen,np.zeros((Grosor,np.shape(fila)[1],3))),axis=0)
        
    imshow('Comparación Color_Check - '+ nombre,imagen)
    imwrite(carpeta + '/Comparacion Color_Check - '+ nombre+'.png',imagen)


#%%
def ReproduccionCie1931(imagenes_patron,shape_imag=(480,640,3),selec_imagenes='All',OutColorSpace='RGB'):
     
    if (selec_imagenes=='All'):
        selec_imagenes=range(np.shape(imagenes_patron)[0])
        
    D65=  np.array([
          [410,	91.486000],
          [450,	117.008000],
          [470,	114.861000],
          [490,	108.811000],
          [505,	108.578000],
          [530,	107.689000],
          [560,	100.000000],
          [590,	88.685600],
          [600,	90.006200],
          [620,	87.698700],
          [630, 83.288600],
          [650,	80.026800],
          [720,	61.604000],
      
          ])
    
    CIE1931 =  np.array([
         
         [410,	0.043510,	0.001210,	0.207400],
         [450,	0.336200,	0.038000,	1.772110],
         [470,	0.195360,	0.090980,	1.287640],
         [490,	0.032010,	0.208020,	0.465180],
         [505,	0.002400,	0.407300,	0.212300],
         [530,	0.165500,	0.862000,	0.042160],
         [560,	0.594500,	0.995000,	0.003900],
         [590,	1.026300,	0.757000,	0.001100],
         [600,	1.062200,	0.631000,	0.000800],
         [620,	0.854450,	0.381000,	0.000190],
         [630,	0.642400,	0.265000,	0.000050],
         [650,	0.283500,	0.107000,	0.000000],
         [720,	0.002899,	0.001047,	0.000000],
         ])
    
    XYZ2RGB= np.array([[3.2406, -1.5372, -0.4986],
             [-0.9689, 1.8758, 0.0415],
             [0.0557, -0.2040, 1.0570],])


    #% Coeficientes
    Coef= (CIE1931[selec_imagenes,1:]*(np.ones((3,1))*D65[selec_imagenes,1].T).T).T
    N = np.sum(Coef,axis=1)
    #%  Reproduccion de color usando CIE
    
    xyz = np.dot(Coef,imagenes_patron[selec_imagenes,:]).T
    xyz = xyz/N[1]
    
    if(OutColorSpace=='RGB'):
        RGB = recorte(np.dot(XYZ2RGB,xyz.T).T)
        sRGB = np.where(RGB<=0.0031308,12.92*RGB,1.055*np.power(RGB,1/2.4)-0.055)
        image=np.reshape(sRGB,shape_imag)
    
    if(OutColorSpace=='Lab'):
        Lab = XYZ2Lab(xyz)
        image=np.reshape(Lab,shape_imag)
        
    if(OutColorSpace=='XYZ'):
        image=np.reshape(xyz,shape_imag)
    
    return image


def ReproduccionCie19312(imagenes_patron,Pesos_ecu,shape_imag=(480,640,3),selec_imagenes='All'):
     
    if (selec_imagenes=='All'):
        selec_imagenes=range(np.shape(imagenes_patron)[0])
        
    D65=  np.array([
          [410,	91.486000],
          [450,	117.008000],
          [470,	114.861000],
          [490,	108.811000],
          [505,	108.578000],
          [530,	107.689000],
          [560,	100.000000],
          [590,	88.685600],
          [600,	90.006200],
          [620,	87.698700],
          [630, 83.288600],
          [650,	80.026800],
          [720,	61.604000],
      
          ])
    
    CIE1931 =  np.array([
         
         [410,	0.043510,	0.001210,	0.207400],
         [450,	0.336200,	0.038000,	1.772110],
         [470,	0.195360,	0.090980,	1.287640],
         [490,	0.032010,	0.208020,	0.465180],
         [505,	0.002400,	0.407300,	0.212300],
         [530,	0.165500,	0.862000,	0.042160],
         [560,	0.594500,	0.995000,	0.003900],
         [590,	1.026300,	0.757000,	0.001100],
         [600,	1.062200,	0.631000,	0.000800],
         [620,	0.854450,	0.381000,	0.000190],
         [630,	0.642400,	0.265000,	0.000050],
         [650,	0.283500,	0.107000,	0.000000],
         [720,	0.002899,	0.001047,	0.000000],
         ])
    
    XYZ2RGB= np.array([[3.2406, -1.5372, -0.4986],
             [-0.9689, 1.8758, 0.0415],
             [0.0557, -0.2040, 1.0570],])


    #% Coeficientes
    Coef= (CIE1931[selec_imagenes,1:]*(np.ones((3,1))*D65[selec_imagenes,1].T).T).T
    N = np.sum(Coef,axis=1)
    #%  Reproduccion de color usando CIE
    
    xyz = np.dot(Coef,(imagenes_patron[selec_imagenes,:].T*Pesos_ecu).T).T
    #print(N)
    xyz = xyz/N[1]
    
    rgb = recorte(np.dot(XYZ2RGB,xyz.T).T)
    
    im_RGB=np.reshape(rgb,shape_imag)
    
    return im_RGB


def error_funtions(errores,type_error):
    type_error= type_error.lower()
    
    if(type_error=='mean'):
        error=np.mean(errores)
    
    if(type_error=='max'):
        error=np.max(errores)
        
    if(type_error=='variance'):
        error=np.var(errores)
        
    if(type_error=='mean_for_standard'):
        error=np.mean(errores)*np.sqrt(np.var(errores))  
    
    if(type_error=='rango'):
        error=np.max(errores)-np.min(errores)
    
    return error



def mejor_combinacion(imagenes_patron,mascaras,color_check,Cant_Image,type_error='mean',imagen_write='off',OutColorSpace='RGB'):
    stuff= range(np.shape(imagenes_patron)[0])
    subset = list(itertools.combinations(stuff,Cant_Image))
    
    min_error=1000000
    a=0
    for i,Comb in enumerate(subset):
        if(i/len(subset)*100>a):
            a+=10
            print('Cant imagenes'+str(int(Cant_Image))+' Avance:' + str("{0:.2f}".format(i/len(subset)*100))+str('%'))


        
        image_recons = ReproduccionCie1931(imagenes_patron,selec_imagenes=Comb,OutColorSpace=OutColorSpace)
    
        errores = Error_de_reproduccion([image_recons], mascaras, color_check)
        
        error= error_funtions(errores,type_error)

        if(error<min_error):
            min_error=error
            mejor_comb=Comb
   
    im_RGB= ReproduccionCie1931(imagenes_patron,selec_imagenes=mejor_comb)
    imshow('IR ERGB CIE 1931 im '+str(int(Cant_Image)),im_RGB)
    if (imagen_write=='on'):
        imwrite('Resultados/Imagenes\IR ERGB CIE 1931 im '+str(int(Cant_Image))+'.png',im_RGB)
        
    return mejor_comb,min_error

def combinaciones_lista_errores(imagenes_patron,mascaras,color_check,Cant_Image,type_error='mean',imagen_write='off',OutColorSpace='RGB'):
    stuff= range(np.shape(imagenes_patron)[0])
    subset = list(itertools.combinations(stuff,Cant_Image))
    
    lista = []
    a=0
    for i,Comb in enumerate(subset):
        if(i/len(subset)*100>a):
            a+=10
            print('Cant imagenes'+str(int(Cant_Image))+' Avance:' + str("{0:.2f}".format(i/len(subset)*100))+str('%'))

        
        imagen= ReproduccionCie1931(imagenes_patron,selec_imagenes=Comb,OutColorSpace=OutColorSpace)

        errores = Error_de_reproduccion([imagen], mascaras, color_check)
        error= error_funtions(errores,type_error)
        lista+= list(Comb)
        lista+= [error]

        
    lista = np.reshape(lista,(len(subset),Cant_Image+1))
    return lista

    