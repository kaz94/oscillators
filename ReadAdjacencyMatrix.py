import csv

filename = '/home/kasia/Pulpit/macierz.csv'
file = '/home/kasia/Pulpit/inzynierka/macierz_do_f.txt'

# csv.register_dialect('csv', quoting=csv.QUOTE_NONE)


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


def read_file(filename = file):
    with open(filename, 'r') as f:
        read_data = f.readlines()
        read_data = [line.split(',') for line in read_data]
        read_data = read_data[1:]
        for l, line in enumerate(read_data):
            for i, j in enumerate(line):
                read_data[l][i] = float(j.strip())

        # print(read_data)
    return read_data



