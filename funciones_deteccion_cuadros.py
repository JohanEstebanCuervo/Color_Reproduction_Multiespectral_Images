# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:06:29 2021

@author: Johan Cuervo
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import system
import os
from sklearn.linear_model import LinearRegression


import funciones_reproduccion_color as func

def modificar_imagen(imagen):
    # agregar = np.zeros((80,640))
    # imagen = np.concatenate((agregar,imagen),axis=0).astype(int)
    # imagen = np.concatenate((imagen,agregar),axis=0).astype(int)
    #imagen = rotar_imagen(imagen, 10*np.pi/180,(240,320)) #pruebas rotando la 
    #imagen = imagen.T
    #imagen=np.flip(imagen)
    #imagen=np.flip(imagen, axis=1)
    return imagen

def filtro_mediana(imagen_filmediana1):
    i=0
    suma=1
    while(suma>0 and i<400):
        imagen_filmediana2 = cv2.medianBlur(imagen_filmediana1, 3)
        suma=np.sum(np.abs(imagen_filmediana1-imagen_filmediana2))
        imagen_filmediana1=imagen_filmediana2
        i+=1
    return imagen_filmediana1


def tamaño_cuadros(cnts):
    lista=[]
    for c in cnts:
        epsilon = 0.01*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        _,_,w,h = cv2.boundingRect(approx)
        if(len(approx)==4 and w>10 and h>10):
            if(1.2>w/h>0.8):
                lista=np.append(lista,(w,h))
    promedio = np.mean(lista)
    return promedio*1.1,promedio*0.9
        
def Contornos(carpeta,lista,imshow='off'):
    contornos=[]
    
    for i,nombre in enumerate(lista): 
        imagen = cv2.cvtColor(cv2.imread(carpeta+"/"+nombre), cv2.COLOR_BGR2GRAY)
        imagen = modificar_imagen(imagen)
        _, imagen_bin = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        imagen_bin = cv2.erode(imagen_bin, None, iterations=2)
        if imshow=='on':
            func.imshow('Imagen binaria '+str(i+1),imagen_bin/255)
        
        cnts,_ = cv2.findContours(imagen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 
        for c in cnts:
            contornos.append(c)
    tamaño= np.shape(imagen)   
    return contornos, tamaño 

def Contornos_cuadrados(contornos):
    maximo,minimo= tamaño_cuadros(contornos)
    contornos_cua=[]
    aristas=[]
    for c in contornos:
        epsilon = 0.01*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        _,_,w,h = cv2.boundingRect(approx)
        if( maximo> w > minimo and maximo> h > minimo and len(approx)<10 ): #len(approx)==4 and
            contornos_cua.append(c)
            aristas.append(len(approx))
    
    return contornos_cua,aristas
    
def grupos_contornos(promedios):
    grupos=[]
    seleccionados=[]
    for i in range(len(promedios)):
        if(np.shape(np.where(seleccionados==i)[0])[0]==0):
            seleccionados=np.append(seleccionados,i)
            grupo=i
            for j in range(i+1,len(promedios)):
                if(np.shape(np.where(seleccionados==j)[0])[0]==0):
                    dist= np.sqrt(np.sum((promedios[i]-promedios[j])**2))
                    if(dist<10):
                        seleccionados=np.append(seleccionados,j)
                        grupo=np.append(grupo,j)
            
            grupos.append(grupo)
        
    return grupos


def filt_Contornos(contornos,aristas):
    aristas=np.array(aristas)
    promedios=np.zeros((1,2))
    contornos_fil= []
    centros=[]
    for c in contornos:
        maximos=np.max(c,axis=0)
        minimos=np.min(c,axis=0)
        prom=np.mean(np.concatenate((maximos,minimos)),axis=0).reshape((1,2))
        promedios=np.concatenate((promedios,prom))
    promedios=promedios[1:,:]
    
    grupos = grupos_contornos(promedios)

    for grupo in grupos:
        dif = aristas[grupo]-4
        if(np.shape(dif)==()):
            contornos_fil.append(contornos[grupo])
            centros= np.concatenate((centros,promedios[grupo,:]))
        else:
            index_min=np.where(dif==min(dif))[0]
            if(np.shape(index_min) != 0):
                index_min=index_min[0]
            
            contornos_fil.append(contornos[grupo[index_min]])
            centros= np.concatenate((centros,promedios[grupo[index_min],:]))
        
    return contornos_fil,np.reshape(centros,(-1,2))

def estimacion_anguloytamanio(contornos):
    ang=0
    tamanio=0
    for c in contornos:
        epsilon = 0.01*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        _,tam,ang2 = cv2.minAreaRect(approx)
        tamanio+=np.sum(tam)
        if ang2>80:
            ang2=90-ang2
        ang+=ang2
    
    ang/=len(contornos)
    tamanio/=(2*len(contornos))
    if(ang>45):
        ang-=90
    return ang*np.pi/180, tamanio
        
def regresion_puntos(centros,order_index):
    if(order_index==1):
        Y= centros[:6,1]
        X= centros[:6,0]
    if(order_index==0):
        Y= centros[:6,0]
        X= centros[:6,1]
        
    modelo = LinearRegression()
    modelo.fit(X=X.reshape(-1,1),y=Y.reshape(-1,1))
    pendiente = modelo.coef_
    angulo = np.arctan(pendiente) #*180/np.pi
    return modelo, angulo

def regresion_lineal(puntos):
    X= puntos[:,0]
    Y= puntos[:,1]
   
    modelo = LinearRegression()
    modelo.fit(X=X.reshape(-1,1),y=Y.reshape(-1,1))
    pendiente = modelo.coef_
    intercepto = modelo.intercept_
    return pendiente, intercepto
    
def rotar_imagen(imagen,angulo,eje):
    
    alpha=np.cos(angulo).reshape(1)
    beta= np.sin(angulo).reshape(1)
    
    M2= np.array([
        [alpha, beta, (1-alpha)*eje[1]- beta*eje[0]],
        [-beta, alpha, beta*eje[1]+(1-alpha)*eje[0]],
        ]).reshape((2,3))
    
    return cv2.warpAffine(imagen,M2,(0,480))

def completar_centros(centros,ang_est):
    
    M=np.array([
        [np.cos(ang_est), -np.sin(ang_est)],
        [np.sin(ang_est), np.cos(ang_est)],
        ])
    
    centros_tras=np.copy(centros)
    centros_tras= np.dot(M,centros_tras.T).T
    
    centros_tras= centros_tras[np.argsort(centros_tras[:,1]).T]
    maxi,mini= np.max(centros_tras,axis=0),np.min(centros_tras,axis=0)
    spam= maxi-mini
    if(spam[0]>spam[1]):
        spam = np.divide(spam,(5,3))
        order_index=1
    else:
        spam = np.divide(spam,(3,5))
        order_index=0

    centros_tras= np.divide(centros_tras-mini,spam)
    centros_int=np.round(centros_tras).astype(int)
    error= np.max(np.abs(centros_tras-centros_int))*np.max(spam)
    print('error estimado de posicion: '+str(error))
    faltantes=[]
    for i in range(int(np.max(centros_int[:,0])+1)):
        for j in range(int(np.max(centros_int[:,1])+1)):
            bandera=0
            c=np.array([i,j])
            for k in centros_int:
                if(np.array_equal(k,c)==True):
                    bandera=1
                    continue
            if(bandera==0):
                faltantes=np.append(faltantes,c)
    faltantes=np.reshape(faltantes,(-1,2))  

    
    centros_tras=np.concatenate((centros_tras,faltantes))
    
    centros_tras= centros_tras[np.argsort(centros_tras[:,order_index]).T]

    for i in range(4):
        parte=np.copy(centros_tras[i*6:i*6+ 6,:])
        parte= parte[np.argsort(parte[:,1-order_index]).T]
        centros_tras[i*6:i*6+ 6,:]=parte
        
    centros_tras= centros_tras*spam+mini
    centros = np.dot(np.linalg.inv(M),centros_tras.T).T

    return centros, error, order_index

def completar_centros2(centros,ang_est):
    error=0
    M=np.array([
        [np.cos(ang_est), -np.sin(ang_est)],
        [np.sin(ang_est), np.cos(ang_est)],
        ])
  
    centros_tras=np.copy(centros)
    centros_tras= np.dot(M,centros_tras.T).T
     
    centros_tras= centros_tras[np.argsort(centros_tras[:,1]).T]
    maxi,mini= np.max(centros_tras,axis=0),np.min(centros_tras,axis=0)
    spam= maxi-mini
    if(spam[0]>spam[1]):
        spam = np.divide(spam,(5,3))
        order_index=1
    else:
        spam = np.divide(spam,(3,5))
        order_index=0

    centros_tras= np.divide(centros_tras-mini,spam)
    centros_int=np.round(centros_tras).astype(int)
    faltantes=[]
    for i in range(int(np.max(centros_int[:,0])+1)):
        for j in range(int(np.max(centros_int[:,1])+1)):
            bandera=0
            c=np.array([i,j])
            for k in centros_int:
                if(np.array_equal(k,c)==True):
                    bandera=1
                    continue
            if(bandera==0):
                faltantes=np.append(faltantes,c)
    faltantes_int=np.reshape(faltantes,(-1,2))  
    try:
        faltantes_tras=[]
        for centro_fal in faltantes_int:
            #regresion en direccion 1
            centros=[]
            for i,centro_list in enumerate(centros_int):
                if centro_fal[0]==centro_list[0]:
                    centros = np.append(centros,centros_tras[i])
            centros=np.reshape(centros,(-1,2))
            
            m1,b1 = regresion_lineal(centros)
            
            #regresion en direccion 2
            centros=[]
            for i,centro_list in enumerate(centros_int):
                if centro_fal[1]==centro_list[1]:
                    centros = np.append(centros,centros_tras[i])
            centros=np.reshape(centros,(-1,2))
            
            m2,b2 = regresion_lineal(centros)
            
            X = (b2-b1)/(m1-m2)
            Y = m1*X+b1
            
            faltantes_tras = np.append(faltantes_tras,[X,Y])
        faltantes_tras = np.reshape(faltantes_tras,(-1,2))
        centros_tras=np.concatenate((centros_tras,faltantes_tras))
    except:
        error= np.max(np.abs(centros_tras-centros_int))*np.max(spam)
        print('error estimado de posicion: '+str(error))
        centros_tras=np.concatenate((centros_tras,faltantes_int))
    
    
    centros_tras= centros_tras[np.argsort(centros_tras[:,order_index]).T]
    
    for i in range(4):
        parte=np.copy(centros_tras[i*6:i*6+ 6,:])
        parte= parte[np.argsort(parte[:,1-order_index]).T]
        centros_tras[i*6:i*6+ 6,:]=parte
        
    centros_tras= centros_tras*spam+mini
    centros = np.dot(np.linalg.inv(M),centros_tras.T).T

    return centros,error, order_index

def Mascaras(centros,tamanio_im,angulo,tamanio_cuadro):
    
    mascaras=[]
    for i in range(len(centros)):
        mascara=np.zeros((tamanio_im),dtype='uint8')
        punto1=centros[i]-round(tamanio_cuadro*0.9/2)
        punto2=centros[i]+round(tamanio_cuadro*0.9/2)
        cv2.rectangle(mascara,punto1,punto2,(255,255,255),-1)
        mascara= rotar_imagen(mascara,angulo,np.flip(centros[i]))
        #cv2.putText(mascara, str(i+1), centros[i],cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0))
        _,mascara= cv2.threshold(mascara, 128, 255, cv2.THRESH_BINARY)
        mascaras.append(mascara)
    return mascaras

def organizar_centros(centros,tamanio_im,angulo,tamanio_cuadro,carpeta,lista):
    
    mascaras_iniciales=[0,5,18,23]
    suma = np.zeros(4)
    for i,pos_centro in enumerate(mascaras_iniciales):
        mascara=np.zeros((tamanio_im),dtype='uint8')
        punto1=centros[pos_centro]-round(tamanio_cuadro*0.9/2)
        punto2=centros[pos_centro]+round(tamanio_cuadro*0.9/2)
    
        cv2.rectangle(mascara,punto1,punto2,(255,255,255),-1)
        mascara= rotar_imagen(mascara,angulo,np.flip(centros[pos_centro]))
        _,mascara= cv2.threshold(mascara, 128, 255, cv2.THRESH_BINARY)
        a=len(np.where(mascara==255)[0])
        for nombre in lista: 
            imagen = cv2.cvtColor(cv2.imread(carpeta+"/"+nombre), cv2.COLOR_BGR2GRAY)
            imagen = modificar_imagen(imagen)
            suma[i] = suma[i] + np.sum(imagen[np.where(mascara==255)]) / a
    
    print('tamaño mascara '+str(a)+' pixeles')   
    argmax = np.argmax(suma)

    if(argmax==0):
        centros2=np.zeros((1,2),dtype='int')
        for i in np.flip(range(4)):
            parte = centros[6*i:6*i+6,:]
            centros2= np.concatenate((centros2,parte))
        centros=centros2[1:,:]
    elif(argmax==1):
        centros = np.flip(centros, axis=0)
    elif(argmax==3):
        for i in np.flip(range(4)):
            parte = np.flip(centros[6*i:6*i+6,:],axis=0)
            centros[6*i:6*i+6,:]= parte
            
    return centros

def Principal(carpeta, lista,mostrar_imagenes='off'):
    
    imagen1=cv2.imread(carpeta+"/"+lista[1])
    imagen1=imagen1[:,:,0]
    imagen1 = modificar_imagen(imagen1)
    func.imshow('imagen',imagen1/255)
    
    contornos,tamanioim = Contornos(carpeta, lista[:13],imshow=mostrar_imagenes)
    contornos, aristas = Contornos_cuadrados(contornos)
    
    imagen=np.zeros((tamanioim),dtype='uint8')
    contornos= np.array(contornos,dtype=object)
    cv2.drawContours(imagen, contornos,-1, (255,255,255),2)
    func.imshow('contornos', imagen/255)
    
    contornos_fil,centros= filt_Contornos(contornos, aristas)
      
    print("Cuardros detectados: "+str(len(centros)))
    
    angulo_est,tamanio_cuadro = estimacion_anguloytamanio(contornos_fil)
    centros_int, error,order_index= completar_centros2(centros,-angulo_est)
    
       
    
    modelo,angulo = regresion_puntos(centros_int,order_index)
    if(angulo*angulo_est<0):
        angulo=-angulo
        
    print('angulo: '+ str(angulo*180/np.pi))
    print('angulo_est: '+str(angulo*180/np.pi))
    centros_int=(centros_int).astype(int)
    centros_org= organizar_centros(centros_int,tamanioim,-angulo,tamanio_cuadro,carpeta,lista)

    mascaras = Mascaras(centros_org,tamanioim,-angulo,tamanio_cuadro)

    imagen=np.zeros((tamanioim),dtype='uint8')
    cv2.drawContours(imagen, contornos_fil,-1, (255,255,255),2)  
    lista= [centros_int[:,1],centros_int[:,0]]
    print(centros_int)
    imagen[tuple(lista)]=255
    func.imshow('contornos filtrados', imagen/255)
    
    
    imagen=np.zeros((tamanioim),dtype='uint8')
    for i in range(len(mascaras)):
        imagen+= mascaras[i]
    
    imagen_mascaras = imagen1.astype(int)+imagen.astype(int)
    func.imshow('mascaras', imagen/255)
    func.imshow('imagen con mascaras',func.recorte(imagen_mascaras/255))
    
    return mascaras, centros_int , centros_org
    