import cv2
import dlib
import pandas as pd
from matplotlib.pyplot import axis, grid, text
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk, Image
from tkinter import filedialog
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from tkinter import *
import tkinter
import os
import collections

from sklearn.utils.extmath import row_norms

root = Tk()
root.geometry("400x600+150+75")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    x = openfn()
    analize(x)


def GrabarVideo():
    detector = dlib.get_frontal_face_detector()

    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # cap = cv2.imread("fr.mp4")
    # cap = cv2.VideoCapture("H:\\Reconocimiento\\new analisis\\agrado.mp4")
    cap = cv2.VideoCapture(0)

    data = pd.DataFrame(np.asarray([[0] for i in range(12)]).T)

    _, frame = cap.read()

    while frame is not None:
        
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        faces=detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(image=gray, box=face)
            arreglo = {}
            j=0
            # for n in range(1,16):
            #     x = landmarks.part(n).x
            #     y = landmarks.part(n).y
            #     arreglo[j] = x
            #     j = j + 1
            #     arreglo[j] = y
            #     j = j + 1
            #     cv2.circle(img=frame, center=(x,y), radius = 1, color=(0,0,0), thickness=-1)
            
            # for n in range(37,47):
            #     x = landmarks.part(n).x
            #     y = landmarks.part(n).y
            #     arreglo[j] = x
            #     j = j + 1
            #     arreglo[j] = y
            #     j = j + 1
            #     cv2.circle(img=frame, center=(x,y), radius = 1, color=(0,0,0), thickness=-1)
            i = 0
            for n in range(48,60):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                arreglo[i] = x
                i = i + 1
                arreglo[i] = y
                i = i + 1
                cv2.circle(img=frame, center=(x,y), radius = 1, color=(0,0,0), thickness=-1)
            data.loc[len(data.index)] = arreglo
            

        cv2.imshow(winname="Detector de rostros", mat=frame)
        _, frame = cap.read()
        if cv2.waitKey(delay=1)==27:
            cap.release()
            data.drop(0,inplace=True)
            new_cols = ['columna1', 'columna2','.....','columna n']
            data.columns = new_cols
            data.to_csv("amargo_1.csv")
            btn2 = tkinter.Button (root, text = "Analizar", padx=160, pady=12, background="Green", foreground="white", command=Analizar)
            btn2.grid(row=3, column=0) 
            cv2.destroyAllWindows()
            break

    cap.release()
    # df=data
    # df.head()
    data.drop(0,inplace=True)
    new_cols = ['m1x','m1y','m2x','m2y','m3x','m3y','m4x','m4y','m5x','m5y','m6x','m6y','m7x','m7y','m8x','m8y','m9x','m9y','m10x','m10y','m11x','m11y','m12x','m12y','m13x','m13y','m14x','m14y','m15x','m15y','m16x','m16y','m17x','m17y','ci1x','ci1y','ci2x','ci2y','ci3x','ci3y','ci4x','ci4y','ci5x','ci5y','cd1x','cd1y','cd2x','cd2y','cd3x','cd3y','cd4x','cd4y','cd5x','cd5y','n1x','n1y','n2x','n2y','n3x','n3y','n4x','n4y','n5x','n5y','n6x','n6y','n7x','n7y','n8x','n8y','n9x','n9y','oi1x','oi1y','oi2x','oi2y','oi3x','oi3y','oi4x','oi4y','oi5x','oi5y','oi6x','oi6y','od1x','od1y','od2x','od2y','od3x','od3y','od4x','od4y','od5x','od5y','od6x','od6y','b1x', 'b1y', 'b2x','b2y','b3x','b3y','b4x', 'b4y','b5x', 'b5y', 'b6x', 'b6y', 'b7x', 'b7y','b8x', 'b8y','b9x', 'b9y','b10x', 'b10y','b11x', 'b11y','b12x', 'b12y','b13x', 'b13y','b14x', 'b14y','b15x', 'b15y','b16x', 'b16y','b17x', 'b17y','b18x', 'b18y','b19x', 'b19y']
    data.columns = new_cols
    data.to_csv("amargo_1.csv")
    btn2 = tkinter.Button (root, text = "Analizar", padx=160, pady=12, background="green", foreground="white", command=Analizar)
    btn2.grid(row=3, column=0) 
    cv2.destroyAllWindows()


def analize(ruta):
    detector = dlib.get_frontal_face_detector()

    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # cap = cv2.imread("fr.mp4")
    # cap = cv2.VideoCapture("H:\\Reconocimiento\\new analisis\\agrado.mp4")
    cap = cv2.VideoCapture(ruta)

    data = pd.DataFrame(np.asarray([[0] for i in range(134)]).T)

    _, frame = cap.read()

    while frame is not None:
        
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        faces=detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(image=gray, box=face)
            arreglo = {}
            
            i = 0
            for n in range(1,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                arreglo[i] = x
                i = i + 1
                arreglo[i] = y
                i = i + 1
                cv2.circle(img=frame, center=(x,y), radius = 1, color=(0,0,0), thickness=-1)
            data.loc[len(data.index)] = arreglo
            

        cv2.imshow(winname="Face", mat=frame)
        _, frame = cap.read()
        if cv2.waitKey(delay=1)==27:
            cap.release()
            # df=data
            # df.head()
            data.drop(0,inplace=True)
            new_cols = ['m1x','m1y','m2x','m2y','m3x','m3y','m4x','m4y','m5x','m5y','m6x','m6y','m7x','m7y','m8x','m8y','m9x','m9y','m10x','m10y','m11x','m11y','m12x','m12y','m13x','m13y','m14x','m14y','m15x','m15y','m16x','m16y','m17x','m17y','ci1x','ci1y','ci2x','ci2y','ci3x','ci3y','ci4x','ci4y','ci5x','ci5y','cd1x','cd1y','cd2x','cd2y','cd3x','cd3y','cd4x','cd4y','cd5x','cd5y','n1x','n1y','n2x','n2y','n3x','n3y','n4x','n4y','n5x','n5y','n6x','n6y','n7x','n7y','n8x','n8y','n9x','n9y','oi1x','oi1y','oi2x','oi2y','oi3x','oi3y','oi4x','oi4y','oi5x','oi5y','oi6x','oi6y','od1x','od1y','od2x','od2y','od3x','od3y','od4x','od4y','od5x','od5y','od6x','od6y','b1x', 'b1y', 'b2x','b2y','b3x','b3y','b4x', 'b4y','b5x', 'b5y', 'b6x', 'b6y', 'b7x', 'b7y','b8x', 'b8y','b9x', 'b9y','b10x', 'b10y','b11x', 'b11y','b12x', 'b12y','b13x', 'b13y','b14x', 'b14y','b15x', 'b15y','b16x', 'b16y','b17x', 'b17y','b18x', 'b18y','b19x', 'b19y']
            data.columns = new_cols
            data.to_csv("amargo_1.csv")
            break

    cap.release()
    # df=data
    # df.head()
    data.drop(0,inplace=True)
    new_cols = ['m1x','m1y','m2x','m2y','m3x','m3y','m4x','m4y','m5x','m5y','m6x','m6y','m7x','m7y','m8x','m8y','m9x','m9y','m10x','m10y','m11x','m11y','m12x','m12y','m13x','m13y','m14x','m14y','m15x','m15y','m16x','m16y','m17x','m17y','ci1x','ci1y','ci2x','ci2y','ci3x','ci3y','ci4x','ci4y','ci5x','ci5y','cd1x','cd1y','cd2x','cd2y','cd3x','cd3y','cd4x','cd4y','cd5x','cd5y','n1x','n1y','n2x','n2y','n3x','n3y','n4x','n4y','n5x','n5y','n6x','n6y','n7x','n7y','n8x','n8y','n9x','n9y','oi1x','oi1y','oi2x','oi2y','oi3x','oi3y','oi4x','oi4y','oi5x','oi5y','oi6x','oi6y','od1x','od1y','od2x','od2y','od3x','od3y','od4x','od4y','od5x','od5y','od6x','od6y','b1x', 'b1y', 'b2x','b2y','b3x','b3y','b4x', 'b4y','b5x', 'b5y', 'b6x', 'b6y', 'b7x', 'b7y','b8x', 'b8y','b9x', 'b9y','b10x', 'b10y','b11x', 'b11y','b12x', 'b12y','b13x', 'b13y','b14x', 'b14y','b15x', 'b15y','b16x', 'b16y','b17x', 'b17y','b18x', 'b18y','b19x', 'b19y']
    data.columns = new_cols
    data.to_csv("amargo_1.csv")
    btn2 = tkinter.Button (root, text = "Analizar", padx=155, pady=12, background="green", foreground="white", command=Analizar)
    btn2.grid(row=3, column=0) 
    cv2.destroyAllWindows()



def Analizar():
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    
    dataframe = pd.read_csv(r"amargo_1.csv")
    #print(dataframe.head()) #Muestra y pinta las etiquetas o cabeceras
    #clsprint(dataframe.describe()) #describe cada dato

    X = np.array(dataframe[['frame','m1x','m1y','m2x','m2y','m3x','m3y','m4x','m4y','m5x','m5y','m6x','m6y','m7x','m7y','m8x','m8y','m9x','m9y','m10x','m10y','m11x','m11y','m12x','m12y','m13x','m13y','m14x','m14y','m15x','m15y','m16x','m16y','m17x','m17y','ci1x','ci1y','ci2x','ci2y','ci3x','ci3y','ci4x','ci4y','ci5x','ci5y','cd1x','cd1y','cd2x','cd2y','cd3x','cd3y','cd4x','cd4y','cd5x','cd5y','n1x','n1y','n2x','n2y','n3x','n3y','n4x','n4y','n5x','n5y','n6x','n6y','n7x','n7y','n8x','n8y','n9x','n9y','oi1x','oi1y','oi2x','oi2y','oi3x','oi3y','oi4x','oi4y','oi5x','oi5y','oi6x','oi6y','od1x','od1y','od2x','od2y','od3x','od3y','od4x','od4y','od5x','od5y','od6x','od6y','b1x', 'b1y', 'b2x','b2y','b3x','b3y','b4x', 'b4y','b5x', 'b5y', 'b6x', 'b6y', 'b7x', 'b7y','b8x', 'b8y','b9x', 'b9y','b10x', 'b10y','b11x', 'b11y','b12x', 'b12y','b13x', 'b13y','b14x', 'b14y','b15x', 'b15y','b16x', 'b16y','b17x', 'b17y','b18x', 'b18y','b19x', 'b19y']])

    kmeans = KMeans(n_clusters=3).fit(X)

    labels = kmeans.predict(X)

    C = kmeans.cluster_centers_
    colores=['red','green', 'yellow']
    asignar=[]
    for row in labels:
        asignar.append(colores[row])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
    copy =  pd.DataFrame()
    copy['label'] = labels;
    cantidadGrupo =  pd.DataFrame()
    cantidadGrupo['color']=colores
    cantidadGrupo['cantidad']=copy.groupby('label').size()


    X_new = np.array(pd.read_csv(r"datos.csv")) #davidguetta

    arreglos2 =kmeans.predict(X_new)

    



    cont = np.count_nonzero(arreglos2 == 0)
    cont2 = np.count_nonzero(arreglos2==1)
    cont3 = np.count_nonzero(arreglos2==2)
    print("amargo: ",cont2)
    print("agrado: ",cont)
    print("neutral: ",cont3)
    resultado1 = (cont * 100) / len(X_new)
    resultado2 = (cont2 * 100) / len(X_new)
    resultado3 = (cont3 * 100) / len(X_new)

    lbl1 = tkinter.Label(root)
    lbl2 = tkinter.Label(root)
    lbl3 = tkinter.Label(root)
    lbl1.grid(row=5, column=0)
    lbl2.grid(row=6, column=0)
    lbl3.grid(row=7, column=0)
    if cont > cont2 and cont>cont3:
        
        lbl1.configure(text="Agrado: "+str(resultado1)+"%",background="green", foreground="white")
        lbl2.configure(text="Disgusto: "+str(resultado2)+"%", background="white")
        lbl3.configure(text="Neutral: "+str(resultado3)+"%", background="white")
    elif cont2>cont and cont2>cont2:
        
        lbl1.configure(text="Agrado: "+str(resultado1)+"%",background="white")
        lbl2.configure(text="Disgusto: "+str(resultado2)+"%", background="green", foreground="white")
        lbl3.configure(text="Neutral: "+str(resultado3)+"%", background="white")
    elif cont3>cont and cont3>cont2:
        
        lbl1.configure(text="Agrado: "+str(resultado1)+"%", background="white")
        lbl2.configure(text="Disgusto: "+str(resultado2)+"%", background="white")
        lbl3.configure(text="Neutral: "+str(resultado3)+"%", background="green", foreground="white")
    
label1 = tkinter.Label(root, text="Analizador de Videos", background="black", foreground="white", padx=150, pady=30)
button1 = tkinter.Button (root, text = "Abrir Video", padx=150, pady=12, background="orange", foreground="white", command=open_img) 
button2 = tkinter.Button (root, text = "Grabar Video", padx=145, pady=12, background="blue", foreground="white", command=GrabarVideo) 
label1.grid(row=0, column=0)
button1.grid (row=2, column=0) 
button2.grid (row=4, column=0)

root.title=("Reconocimiento de Emociones")
root.mainloop()