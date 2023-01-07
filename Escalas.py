def escalas(root ,modo):
    if  modo == 0:
        #jónico
        escala = [root, root + 2, root + 4, root + 5, root + 7, root + 9, root + 11, root + 12, root + 2 + 12, root + 4 + 12, root + 5 + 12, root + 7 + 12, root + 9 + 12, root + 11 + 12, root + 12 + 12, root + 2 + 12 + 12, root + 4 + 12 + 12, root + 5 + 12 + 12, root + 7 + 12 + 12, root + 9 + 12 + 12, root + 11 + 12 + 12, root + 12 + 12 + 12, root + 2 + 12 + 12 + 12, root + 4 + 12 + 12 + 12, root + 5 + 12 + 12 + 12, root + 7 + 12 + 12 + 12, root + 9 + 12 + 12 + 12, root + 11 + 12 + 12 + 12]
    
    elif modo == 1:
        #dórico
        escala = [root, root + 2, root + 3, root + 5, root + 7, root + 9, root + 10, root + 12, root + 2 + 12, root + 3 + 12, root + 5 + 12, root + 7 + 12, root + 9 + 12, root + 10 + 12, root + 12 + 12, root + 2 + 12 + 12, root + 3 + 12 + 12, root + 5 + 12 + 12, root + 7 + 12 + 12, root + 9 + 12 + 12, root + 10 + 12 + 12, root + 12 + 12 + 12, root + 2 + 12 + 12 + 12, root + 3 + 12 + 12 + 12, root + 5 + 12 + 12 + 12, root + 7 + 12 + 12 + 12, root + 9 + 12 + 12 + 12, root + 10 + 12 + 12 + 12]

    elif modo == 2:
        #frigio
        escala = [root, root + 1, root + 3, root + 5, root + 7, root + 8, root + 10, root + 12, root + 1 + 12, root + 3 + 12, root + 5 + 12, root + 7 + 12, root + 8 + 12, root + 10 + 12, root + 12 + 12, root + 1 + 12 + 12, root + 3 + 12 + 12, root + 5 + 12 + 12, root + 7 + 12 + 12, root + 8 + 12 + 12, root + 10 + 12 + 12, root + 12 + 12 + 12, root + 1 + 12 + 12 + 12, root + 3 + 12 + 12 + 12, root + 5 + 12 + 12 + 12, root + 7 + 12 + 12 + 12, root + 8 + 12 + 12 + 12, root + 10 + 12 + 12 + 12]

    elif modo == 3:
        #lidio
        escala = [root, root + 2, root + 4, root + 6, root + 7, root + 9, root + 11, root + 12, root + 2 + 12, root + 4 + 12, root + 6 + 12, root + 7 + 12, root + 9 + 12, root + 11 + 12, root + 12 + 12, root + 2 + 12 + 12, root + 4 + 12 + 12, root + 6 + 12 + 12, root + 7 + 12 + 12, root + 9 + 12 + 12, root + 11 + 12 + 12, root + 12 + 12 + 12, root + 2 + 12 + 12 + 12, root + 4 + 12 + 12 + 12, root + 6 + 12 + 12 + 12, root + 7 + 12 + 12 + 12, root + 9 + 12 + 12 + 12, root + 11 + 12 + 12 + 12]
    
    elif modo == 4:
        #mixolidio
        #escala = [root, root + 2, root + 4, root + 5, root + 7, root + 9, root + 10, root + 12, root + 2 + 12, root + 4 + 12, root + 5 + 12, root + 7 + 12, root + 9 + 12, root + 10 + 12, root + 12 + 12, root + 2 + 12 + 12, root + 4 + 12 + 12, root + 5 + 12 + 12, root + 7 + 12 + 12, root + 9 + 12 + 12, root + 10 + 12 + 12, root + 12 + 12 + 12, root + 2 + 12 + 12 + 12, root + 4 + 12 + 12 + 12, root + 5 + 12 + 12 + 12, root + 7 + 12 + 12 + 12, root + 9 + 12 + 12 + 12, root + 10 + 12 + 12 + 12]
        #The next is a experiment for better harmony in the melody#*******************
        escala = [root, root + 2, root + 5, root + 7, root + 10, root + 12, root + 2 + 12, root + 4 + 12, root + 7 + 12, root + 9 + 12, root + 12 + 12, root + 2 + 12 + 12, root + 5 + 12 + 12, root + 7 + 12 + 12, root + 10 + 12 + 12, root + 12 + 12 + 12, root + 2 + 12 + 12 + 12, root + 4 + 12 + 12 + 12, root + 7 + 12 + 12 + 12, root + 9 + 12 + 12 + 12, root + 12 + 12 + 12 + 12, root + 12 + 12 + 12 + 12 + 2, root + 12 + 12 + 12 + 12 + 5, root + 12 + 12 + 12 + 12 + 7, root + 12 + 12 + 12 + 12 + 10, root + 12 + 12 + 12 + 12 +12, root + 12 + 12 + 12 + 12 +12 +2, root + 12 + 12 + 12 + 12 + 12 + 4 ]

    elif modo == 5:
        #eólico
        escala = [root, root + 2, root + 3, root + 5, root + 7, root + 8, root + 10, root + 12, root + 2 + 12, root + 3 + 12, root + 5 + 12, root + 7 + 12, root + 8 + 12, root + 10 + 12, root + 12 + 12, root + 2 + 12 + 12, root + 3 + 12 + 12, root + 5 + 12 + 12, root + 7 + 12 + 12, root + 8 + 12 + 12, root + 10 + 12 + 12, root + 12 + 12 + 12, root + 2 + 12 + 12 + 12, root + 3 + 12 + 12 + 12, root + 5 + 12 + 12 + 12, root + 7 + 12 + 12 + 12, root + 8 + 12 + 12 + 12, root + 10 + 12 + 12 + 12]

    elif modo == 6:
        #lócrio
        escala = [root, root + 1, root + 3, root + 5, root + 6, root + 8, root + 10, root + 12, root + 1 + 12, root + 3 + 12, root + 5 + 12, root + 6 + 12, root + 8 + 12, root + 10 + 12, root + 12 + 12, root + 1 + 12 + 12, root + 3 + 12 + 12, root + 5 + 12 + 12, root + 6 + 12 + 12, root + 8 + 12 + 12, root + 10 + 12 + 12, root + 12 + 12 + 12, root + 1 + 12 + 12 + 12, root + 3 + 12 + 12 + 12, root + 5 + 12 + 12 + 12, root + 6 + 12 + 12 + 12, root + 8 + 12 + 12 + 12, root + 10 + 12 + 12 + 12]

    return escala 
