# ProyectoID

🎨 Proyecto: Plantilla de Colorear por Números con OpenCV
Este proyecto toma una imagen de entrada y la convierte en una plantilla interactiva para colorear por números, utilizando Python, OpenCV, NumPy y Scikit-learn (para el clustering K-Means).

🚀 Instalación
Para ejecutar este proyecto, necesitas las siguientes librerías de Python. Puedes instalarlas todas con un solo comando usando pip:

```
pip install opencv-python numpy scikit-learn
```
opencv-python: Para todo el procesamiento de imágenes y la interfaz gráfica.

numpy: Para la manipulación eficiente de los arrays de imágenes.

scikit-learn: Para aplicar el algoritmo K-Means y cuantificar los colores.

🏃‍♂️ Cómo Ejecutar el Proyecto
Clona o descarga este repositorio.

Asegúrate de tener una imagen: Coloca la imagen que deseas convertir (ej. mi_foto.jpg) en la misma carpeta que el script.

Actualiza el script: Abre el archivo .py y asegúrate de que el nombre de la imagen en la línea cv2.imread('mi_foto.jpg') coincida con el nombre de tu archivo.
Ejecuta el script: Abre tu terminal, navega a la carpeta del proyecto y ejecuta:
```
python proyectoID.py
```

🎮 Instrucciones de Uso
Al ejecutar el script, se abrirán dos ventanas:

"Paleta": Muestra los colores principales (K) de la imagen.

Haz clic en un color en esta ventana para seleccionar tu "pincel".

"Plantilla": Muestra la hoja de colorear con bordes y números.

Haz clic en una región cuyo número coincida con el color de tu pincel para rellenarla (usando cv2.floodFill).

Presiona la tecla 'q' con las ventanas activas para cerrar la aplicación.
