import csv
import numpy as np

filename = '/home/kasia/Pulpit/macierz.csv'



#csv.register_dialect('csv', quoting=csv.QUOTE_NONE)
class ReadAdjacencyMatrix:
    matrix = []
    def __init__(self):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i != 0:
                    row = [e.replace('','0') for row in self.matrix for e in row]
                    ReadAdjacencyMatrix.matrix.append(row)
                    print(row)





