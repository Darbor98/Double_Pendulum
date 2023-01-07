import csv
import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile 
import Escalas as es
import random as rn

#Condiciones iniciales
#-------------------------------------------------------------------------------
m1, m2 = 1.0, 1.0                                 #masas
l1, l2 = 1.0, 1.0                                 #largo cuerdas
g = 9.8                                           #gravedad
theta_0, phi_0 = np.pi, np.pi/2                   #ángulos iniciales
#theta_0, phi_0 = np.pi, rn.uniform(0, np.pi)     #ángulos iniciales aleatorios
p1_0, p2_0 = 0.0, 0.0                             #velocidades iniciales
t_min = 0                                         #tiempo de ejecución
N = 2*500000                                      #tamaño de grilla, Iteraciones para RK4
print("Tiempo de simulación (float): ")
t_max = float(input())

h=(t_max - t_min)/N                 #paso temporal
#-------------------------------------------------------------------------------
def Hamilton(theta, phi, p1, p2):
#-------------------------------------------------------------------------------
#Constantes de simplificación
#-------------------------------------------------------------------------------
  c1 = (p1 * p2 * np.sin(theta - phi))/(l1 * l2 * (m1 + m2 * np.sin(theta - phi)**2))
  c2 = (m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(theta - phi))/(2 * l1**2 * l2**2 * (m1 + m2 * np.sin(theta - phi)**2)**2)
#-------------------------------------------------------------------------------
#ECUACIONES DE MOVIMIENTO DE HAMILTON
#-------------------------------------------------------------------------------
  w1 = (l1 * p1 - l1 * p2 * np.cos(theta - phi))/(l1**2 * l2 * (m1 + m2 * np.sin(theta - phi)**2))

  w2 = (-m2 * l2 * p1 * np.cos(theta - phi) + (m1 + m2) * l1 * p2)/(m2 * l1 * l2**2 * (m1 + m2 * np.sin(theta - phi)**2))
    
  p1d = -(m1 + m2) * g * l1 * np.sin(theta) - c1 + c2 * np.sin(2 * ( theta - phi))

  p2d = -m2 * g * l2 * np.sin(phi) + c1 - c2 * np.sin(2 * ( theta - phi))
#-------------------------------------------------------------------------------
#MOMENTO
#-------------------------------------------------------------------------------
  p1 = (m1 + m2) * l1**2 * w1 + m2 * l1 * l2 *w2 * np.cos(theta - phi)
  p2 = m2 * l2**2 * w2 + m2 * l1 * l2 * w1 * np.cos(theta - phi)
#-------------------------------------------------------------------------------
  return np.array([w1, w2, p1d, p2d])
#-------------------------------------------------------------------------------
#CORROBORANDO
H = Hamilton(theta_0, phi_0, p1_0, p2_0)
print(H)
#-------------------------------------------------------------------------------
#INTEGRANDO EL SISTEMA DE ECACIONES DEFERENCIALES MEDIEANTE EL MÉTODO DE RUNGE KUTTA 4
def RK4(H, h):
#VALORES INICIALES
#-------------------------------------------------------------------------------
  theta = theta_0
  phi   = phi_0
  p1    = p1_0
  p2    = p2_0
#-------------------------------------------------------------------------------
  y = np.array([theta, phi, p1, p2])
  coord = np.zeros([N,2])
  #print(shape(coord))
#-------------------------------------------------------------------------------
#RUNGE KUTTA 4
#-------------------------------------------------------------------------------
  for i in range(t_min, N, 1):  
    k1 = h*Hamilton(*y)
    k2 = h*Hamilton(*(y + k1/2))
    k3 = h*Hamilton(*(y + k2/2))
    k4 = h*Hamilton(*(y + k3))

    R = 1.0/6.0 *(k1 + 2.0*k2 + 2.0*k3 + k4)

    y[0]  += R[0]
    y[1]  += R[1]
    y[2]  += R[2]
    y[3]  += R[3]
#-------------------------------------------------------------------------------
#COORDENADAS CARTESIANAS
#-------------------------------------------------------------------------------
    coord[i][0] = l1*np.cos(y[0]) +l2* np.cos(y[1])
    coord[i][1] = -l1*np.sin(y[0])-l2* np.sin(y[1])
#-------------------------------------------------------------------------------    
    # print("itenarion: ", i)
    # print("Tiempo: ", i*h)
    # print(y1)
    # print("coords: ", coord)
#-------------------------------------------------------------------------------
  return coord
#-------------------------------------------------------------------------------
S1 = RK4(H,h)

#Velocidades
def RK4_2(H, h):
#VALORES INICIALES
#-------------------------------------------------------------------------------
  theta = theta_0
  phi   = phi_0
  p1    = p1_0
  p2    = p2_0
#-------------------------------------------------------------------------------
  y1 = np.array([theta, phi, p1, p2])
  vel = np.zeros([N,3])
  #print(shape(vel))
#-------------------------------------------------------------------------------
#RUNGE KUTTA 4
#-------------------------------------------------------------------------------
  for i in range(t_min, N, 1): 
    k1 = h*Hamilton(*y1)
    k2 = h*Hamilton(*(y1 + k1/2))
    k3 = h*Hamilton(*(y1 + k2/2))
    k4 = h*Hamilton(*(y1 + k3))

    R = 1.0/6.0 *(k1 + 2.0*k2 + 2.0*k3 + k4)

    y1[0]  += R[0]
    y1[1]  += R[1]
    y1[2]  += R[2]
    y1[3]  += R[3] 
#-------------------------------------------------------------------------------
#COORDENADAS CARTESIANAS
#-------------------------------------------------------------------------------
    vel[i][0] = Hamilton(*y1)[0]
    vel[i][1] = Hamilton(*y1)[1]
    vel[i][2] = i*h 
#-------------------------------------------------------------------------------    
    #print("itenarion: ", i)
    # print("Tiempo: ", i*h)
    # #print(y1)
    #print("velocidades: ", vel[i][0], vel[i][1])
#-------------------------------------------------------------------------------
  return vel
#-------------------------------------------------------------------------------
S2 = RK4_2(H,h)
#-------------------------------------------------------------------------------
#Pequeño error*** 
#Hay que restarle 1, porque la cuenta inicia en 0
print("Selección de escala (int): ")
print("1 - Jónica")
print("2 - Dórica")
print("3 - Frigia")
print("4 - Lidia")
print("5 - Mixolídia")
print("6 - Eólica")
print("7 - Dórica")
n_scale = int(input())

print("Selección de tónica (int):")

#Pequeño error*** 
#Hay que restarle 1, porque la cuenta inicia en 0
print("1 - C")
print("2 - C#")
print("3 - D")
print("4 - D#")
print("5 - E")
print("6 - F")
print("7 - F#")
print("8 - G")
print("9 - G#")
print("10 - A")
print("11 - A#")
print("12 - B")

n_root = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
ss = int(input())

nota = es.escalas(n_root[ss], n_scale)
L = l1 +l2
print(L)
#-------------------------------------------------------------------------------
track    = 0
channel  = 0
time     = 0   # beats
tempo    = 120  # BPM
MyMIDI = MIDIFile(1)                   
MyMIDI.addTempo(track,time, tempo)
#ALCANCE MAXIMO
#-------------------------------------------------------------------------------
y0 = L*np.cos(np.pi)
y1 = L*np.cos(0)
x0 = L*np.sin(np.pi/2)
m = 2*y1/7                #espacio entre puntos del grid X
n = y1/4                  #espacio entre puntos del grid Y
print(y0, y1, x0)
#-------------------------------------------------------------------------------
#GRID
#-------------------------------------------------------------------------------
X = np.array([y0, y0+m, y0+2*m, y0+3*m, y0+4*m, y0+5*m, y0+6*m, y0+7*m])
Y = np.array([0, n, 2*n, 3*n, 4*n, 5*n, 6*n, 7*n])
print(X)
print(Y)
#Normalización de las velocidades para el volumen
def norm(f):
  V = 50 + ( ( np.abs(f) - np.amin(np.abs(S2[:,1])) ) * (127-50) ) / (np.amax(np.abs(S2[:,1])) - np.amin(np.abs(S2[:,1])))

  return int(V)
print("normlaización")
print(norm(S2[5600][1]))
print(S2[5600][1])
#-------------------------------------------------------------------------------
#DURACIÓN DE LAS NOTAS RESPECTO LOS MODOS NORMALES
#-------------------------------------------------------------------------------
def dura(n,m):
  dur = 0.0
  if n<0 and 0<m:
    dur = 1.0
  elif 0<n and m<0:
    dur = 1.0
  elif 0<n and 0<m:
    dur = 2.0
  elif n<0 and m<0:
    dur = 2.0

  return dur
#-------------------------------------------------------------------------------
t = 0
print("duración notas: (float)")
fig = float(input())
for l in range(0, N, int(5000/3)): #(5000/3)*h es equivalente a 1/4 de nota para el rango de tiempo escogido

#Columna 6 GRADO 
    #-------------------------------------------------------------------------------
  if   X[0] < S1[l][0]<X[1] and Y[0]< S1[l][1] < Y[1]:
    
    MyMIDI.addNote(track,channel, nota[26], t ,fig, int(norm(S2[l][0])))
    print("E4      :",  nota[26], t, l,dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[0] < S1[l][0]<X[1] and Y[1]< S1[l][1] < Y[2]:
    
    MyMIDI.addNote(track,channel, nota[19],t,fig, int(norm(S2[l][0])))
    print("E3      :",  nota[19], t, l,dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[0] < S1[l][0]<X[1] and Y[2]< S1[l][1] < Y[3]:
    
    MyMIDI.addNote(track,channel, nota[12],t,fig, int(norm(S2[l][0])))
    print("E2      :",  nota[12], t, l,dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[0] < S1[l][0]<X[1] and Y[3]< S1[l][1] < Y[4]:
    
    MyMIDI.addNote(track,channel, nota[5],t,fig, int(norm(S2[l][0])))
    print("E1      :",  nota[5], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  #Columna 4 GRADO
  #-------------------------------------------------------------------------------
  elif X[1] < S1[l][0]<X[2] and Y[0]< S1[l][1] < Y[1]:
    
    MyMIDI.addNote(track,channel, nota[24],t,fig, int(norm(S2[l][0])))
    print("C#4     :",  nota[24], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[1] < S1[l][0]<X[2] and Y[1]< S1[l][1] < Y[2]:
    
    MyMIDI.addNote(track,channel, nota[17],t,fig, int(norm(S2[l][0])))
    print("C#3      :",  nota[17], t, l,dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[1] < S1[l][0]<X[2] and Y[2]< S1[l][1] < Y[3]:
    
    MyMIDI.addNote(track,channel, nota[10],t,fig, int(norm(S2[l][0])))
    print("C#2      :",  nota[10], t, l,dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[1] < S1[l][0]<X[2] and Y[3]< S1[l][1] < Y[4]:
    
    MyMIDI.addNote(track,channel, nota[3],t,fig, int(norm(S2[l][0])))
    print("C#1      :",  nota[3], t, l,dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 


  #Columna 2 GRADO
  #-------------------------------------------------------------------------------
  elif X[2] < S1[l][0]<X[3] and Y[0]< S1[l][1] < Y[1]:
    
    MyMIDI.addNote(track,channel, nota[22],t,fig, int(norm(S2[l][0])))
    print("A#4      :",  nota[22], t, l,dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[2] < S1[l][0]<X[3] and Y[1]< S1[l][1] < Y[2]:
    
    MyMIDI.addNote(track,channel, nota[15],t,fig, int(norm(S2[l][0])))
    print("A#3      :",  nota[15], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0])))

  elif X[2] < S1[l][0]<X[3] and Y[2]< S1[l][1] < Y[3]:
    
    MyMIDI.addNote(track,channel, nota[8],t,fig, int(norm(S2[l][0])))
    print("A#2      :",  nota[8], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[2] < S1[l][0]<X[3] and Y[3]< S1[l][1] < Y[4]:
    
    MyMIDI.addNote(track,channel, nota[1],t,fig, int(norm(S2[l][0])))
    print("A#1      :",  nota[1], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 


  #Columna RAIZ
  #-------------------------------------------------------------------------------
  elif X[3] < S1[l][0]<X[4] and Y[0]< S1[l][1] < Y[1]:
    
    MyMIDI.addNote(track,channel, nota[21],t,fig, int(norm(S2[l][0])))
    print("G#4      :",  nota[21], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[3] < S1[l][0]<X[4] and Y[1]< S1[l][1] < Y[2]:
    
    MyMIDI.addNote(track,channel, nota[14],t,fig, int(norm(S2[l][0])))
    print("G#3      :",  nota[14], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[3] < S1[l][0]<X[4] and Y[2]< S1[l][1] < Y[3]:
    
    MyMIDI.addNote(track,channel, nota[7],t,fig, int(norm(S2[l][0])))
    print("G#2      :",  nota[7], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[3] < S1[l][0]<X[4] and Y[3]< S1[l][1] < Y[4]:
    
    MyMIDI.addNote(track,channel, nota[0],t,fig, int(norm(S2[l][0])))
    print("G#1      :",  nota[0], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  #Columna 3 GRADO
  #-------------------------------------------------------------------------------
  elif X[4] < S1[l][0]<X[5] and Y[0]< S1[l][1] < Y[1]:
    
    MyMIDI.addNote(track,channel, nota[23],t,fig, int(norm(S2[l][0])))
    print("B4      :",  nota[23], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[4] < S1[l][0]<X[5] and Y[1]< S1[l][1] < Y[2]:
    
    MyMIDI.addNote(track,channel, nota[16],t,fig, int(norm(S2[l][0])))
    print("B3      :",  nota[16], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[4] < S1[l][0]<X[5] and Y[2]< S1[l][1] < Y[3]:
    
    MyMIDI.addNote(track,channel, nota[9],t,fig, int(norm(S2[l][0])))
    print("B2      :",  nota[9], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[4] < S1[l][0]<X[5] and Y[3]< S1[l][1] < Y[4]:
    
    MyMIDI.addNote(track,channel, nota[2],t,fig, int(norm(S2[l][0])))
    print("B1      :",  nota[2], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 


  #Columna 5 GRADO
  #-------------------------------------------------------------------------------
  elif X[5] < S1[l][0]<X[6] and Y[0]< S1[l][1] < Y[1]:
    
    MyMIDI.addNote(track,channel, nota[25],t,fig, int(norm(S2[l][0])))
    print("D#4      :",  nota[25], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[5] < S1[l][0]<X[6] and Y[1]< S1[l][1] < Y[2]:
    
    MyMIDI.addNote(track,channel, nota[18],t,fig, int(norm(S2[l][0])))
    print("D#3      :",  nota[18], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[5] < S1[l][0]<X[6] and Y[2]< S1[l][1] < Y[3]:
    
    MyMIDI.addNote(track,channel, nota[11],t,fig, int(norm(S2[l][0])))
    print("D#2      :",  nota[11], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[5] < S1[l][0]<X[6] and Y[3]< S1[l][1] < Y[4]:
    
    MyMIDI.addNote(track,channel, nota[4],t,fig, int(norm(S2[l][0])))
    print("D#1      :",  nota[4], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 


  #Columna 7 GRADO
  #-------------------------------------------------------------------------------
  elif X[6] < S1[l][0]<X[7] and Y[0]< S1[l][1] < Y[1]:
    
    MyMIDI.addNote(track,channel, nota[27],t,fig, int(norm(S2[l][0])))
    print("F#4      :",  nota[27], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[6] < S1[l][0]<X[7] and Y[1]< S1[l][1] < Y[2]:
    
    MyMIDI.addNote(track,channel, nota[20],t,fig, int(norm(S2[l][0])))
    print("F#3      :",  nota[20], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[6] < S1[l][0]<X[7] and Y[2]< S1[l][1] < Y[3]:
    
    MyMIDI.addNote(track,channel, nota[13],t,fig, int(norm(S2[l][0])))
    print("F#2      :",  nota[13], t,l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

  elif X[6] < S1[l][0]<X[7] and Y[3]< S1[l][1] < Y[4]:
    
    MyMIDI.addNote(track,channel, nota[6],t,fig, int(norm(S2[l][0])))
    print("F#1      :",   nota[6], t, l, dura(S2[l][0], S2[l][1]), int(norm(S2[l][0]))) 

#-------------------------------------------------------------------------------
  #t = int(l*h)
  t += dura(S2[l][0], S2[l][1])/2

#-------------------------------------------------------------------------------
#Guardado del archivo midi
#-------------------------------------------------------------------------------
print("Nombre del archivo: (string)")
nombre = input()

with open(nombre + ".mid", 'wb') as output_file:
  MyMIDI.writeFile(output_file)
#-------------------------------------------------------------------------------
#GRÁFICA ESTÁTICA DE LA POSICIÓN DE LA SEGUNDA MASA
#-------------------------------------------------------------------------------
ax = plt.axes() 
ax.set_facecolor("#121212")
plt.rcParams["figure.figsize"] = [15, 10]
plt.rcParams["figure.autolayout"] = True

x = S1[:,0]
y = S1[:,1]

plt.title("Line graph")
plt.plot(y, -x ,color="#FFB7B2")

plt.show()
#-------------------------------------------------------------------------------
#GRÁFICA ESTÁTICA DE LA VELOCIDAD DE LA SEGUNDA MASA
#-------------------------------------------------------------------------------
ax = plt.axes() 
ax.set_facecolor("#ffffff")
plt.rcParams["figure.figsize"] = [15, 10]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(t_min, t_max, N)
y = S2[:,1]

plt.title("Line graph")
plt.plot(x, y ,color="#000000")

plt.show()
#-------------------------------------------------------------------------------
#Guarda un CSV con todas las posiciones por las que pasó la masa 2.
cab = ["x", "y"]
with open('coordenadas2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(cab)
    writer.writerows(S1)
#-------------------------------------------------------------------------------
#FIN DEL PROGRAMA
#-------------------------------------------------------------------------------