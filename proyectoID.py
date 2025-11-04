import cv2
import numpy as np
from sklearn.cluster import KMeans

# Primer paso: Reducir la pelta de colores de la imagen a una menejable.

imagen_original = cv2.imread("../imagenes/ramita.jpg")
#imagen_original = cv2.imread("../imagenes/caraCubista.jpg")
#imagen_original = cv2.imread("../imagenes/loro2.jpg")
#imagen_original = cv2.imread("../imagenes/caraCubista.jpg")
#imagen_original = cv2.imread("../imagenes/img_gato_ejemplo.jpg")


imagen_orignal_resized = cv2.resize(imagen_original, (500,500))

cv2.imshow("Imagen original", imagen_orignal_resized)

#-------------------------------------------------------------------------------------------------------
#---------------------------------------------- K-MEANS  -----------------------------------------------
#-------------------------------------------------------------------------------------------------------

datos_pixel = imagen_orignal_resized.reshape(-1,3)
datos_pixel = np.float32(datos_pixel) #KMEANS trabaja mejor con floats

# Definimos el número de colores (K) que tendrá la paleta
K = 9

#Inicializaremos el modelo k-means
# n_clusters = K (número de colores con los que queremos terminar)
# random_state = 42 (Semilla fija para resultados consistentes)
# n_inits = 10 (Este es el número de repeticiones del algoritmo, cuantas más repeticiones mayor precisión para encontrar una buana paleta pero más timepo de ejecución)
modelo_kmeans = KMeans(n_clusters = K, random_state=42, n_init=10)

#Entrenamos al algoritmo
modelo_kmeans.fit(datos_pixel)

colores_paleta = modelo_kmeans.cluster_centers_ # Array con los K colores más predominantes en la imagen  
colores_paleta = np.uint8(colores_paleta) 

etiquetas_pixeles = modelo_kmeans.labels_ # Indica a cual de los K colores pertenece cada pixel 

imagen_colores = colores_paleta[etiquetas_pixeles]
imagen_posterizada = imagen_colores.reshape(imagen_orignal_resized.shape)

cv2.imshow('Imagen resultante', imagen_posterizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------
#------------------------------------- FILTRADO DE IMAGEN ----------------------------------------------
#-------------------------------------------------------------------------------------------------------

def thresh_canny(val):

    threshold = val
    
    canny_output = cv2.Canny(imagen_gray, threshold, threshold + 50)

    _, canny_invertido = cv2.threshold(canny_output, 0, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('Canny invertido', canny_invertido)

    return canny_invertido


imagen_posterizada = cv2.blur(imagen_posterizada,(5,5))
imagen_gray = cv2.cvtColor(imagen_posterizada, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Imagen', imagen_gray)

max_thresh = 250
thresh = 50 

cv2.imshow('Imagen filtrada', imagen_gray)
cv2.createTrackbar('Canny Thresh', 'Imagen filtrada', thresh, max_thresh, thresh_canny)

cv2.waitKey(0)

umbral_utilizado = cv2.getTrackbarPos('Canny Thresh', 'Imagen filtrada')

cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------
#------------------------------------- DIBUJAR CONTORNOS -----------------------------------------------
#-------------------------------------------------------------------------------------------------------

img_canny_final = cv2.Canny(imagen_gray, umbral_utilizado, umbral_utilizado + 50)

_, plantilla_sin_numeros = cv2.threshold(img_canny_final, 0, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('Imagen final a colorear sin numero', plantilla_sin_numeros)

cv2.waitKey(0)
cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------
#-------------------------------------  PONER NÚMEROS --------------------------------------------------
#-------------------------------------------------------------------------------------------------------

imagen_con_numeros = plantilla_sin_numeros.copy() 

color_texto = (0, 0, 0) 
escala_fuente = 0.5
grosor_fuente = 1

for i in range(K): # Iterar a través de cada uno de los K colores/números
    mascara_clase = (etiquetas_pixeles == i).reshape(imagen_orignal_resized.shape[:2]).astype(np.uint8) * 255   
    contornos, _ = cv2.findContours(mascara_clase,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        if cv2.contourArea(contorno) < 100:
            continue
        
        #Bounding Rect
        x, y, w, h = cv2.boundingRect(contorno)

        centroide_x = x + w // 2 
        centroide_y = y + h // 2 

        #Momentos
        #M = cv2.moments(contorno)
        #if M["m00"] != 0:
        #    centroide_x = int(M["m10"] / M["m00"])
        #    centroide_y = int(M["m01"] / M["m00"])
        #else:
        #    centroide_x, centroide_y = 0, 0

        cv2.putText(imagen_con_numeros, str(i+1), (centroide_x, centroide_y),
                        cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, color_texto, grosor_fuente, cv2.LINE_AA)
  
imagen_con_numeros = cv2.cvtColor(imagen_con_numeros, cv2.COLOR_GRAY2BGR)
#-------------------------------------------------------------------------------------------------------
#------------------------------------ SEGUNDA VENTANA --------------------------------------------------
#-------------------------------------------------------------------------------------------------------

tam_boton = 50
margen = 10

(ancho, alto), linea_base = cv2.getTextSize('GUADAR', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

paleta = np.zeros((70, (K*70) + ancho, 3), np.uint8)
paleta[:] = (255, 255, 255)

for i in range(K):
    
    x1 = (tam_boton * i) + (margen * i)
    x2 = x1 + tam_boton

    color_np = colores_paleta[i]
    color_tupla = (int(color_np[0]), int(color_np[1]), int(color_np[2]))
    cv2.rectangle(paleta, (x1, margen), (x2, margen + tam_boton), color_tupla, -1)
    cv2.putText(paleta, str(i+1), (x1 + 15, tam_boton - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


cv2.rectangle(paleta, (K*60, margen), ((K*60) + ancho + 20, margen + tam_boton), (0,0,0), 2)
cv2.putText(paleta, 'GUARDAR', (K*60 + 5, tam_boton - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)


#-------------------------------------------------------------------------------------------------------
#----------------------------------- FUNCIONES PINTAR --------------------------------------------------
#-------------------------------------------------------------------------------------------------------

def paleta_callback(event,x,y,flags,param):
    global color
    
    if event == cv2.EVENT_LBUTTONDOWN:    
        k = x//60

        #print("HE seleccionado el color", k)     

        if k + 1 > K:
            cv2.imwrite("plantilla_numeros.jpg", imagen_con_numeros)
        else:
            color_np = colores_paleta[k]
            color = (int(color_np[0]), int(color_np[1]), int(color_np[2]))


def plantilla_callback(event,x,y,flags,param):
    global x_prev,y_prev

    if event == cv2.EVENT_LBUTTONDOWN:
        x_prev,y_prev = x,y

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.line(imagen_con_numeros,(x_prev,y_prev),(x,y),color, 8)
        x_prev,y_prev = x,y

    cv2.imshow('Plantilla',imagen_con_numeros)   


cv2.imshow("Plantilla", imagen_con_numeros)
cv2.imshow("Paleta", paleta) 
cv2.imshow("Imagen coloreada", imagen_posterizada)

cv2.setMouseCallback("Paleta", paleta_callback)
cv2.setMouseCallback("Plantilla", plantilla_callback)


while True:
    

    key = cv2.waitKey(100)
    if key == 27 or key == ord('q'): break

cv2.destroyAllWindows()
