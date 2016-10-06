
class Oscillator:

    _id = 0

    def __init__(self, alfa, mi, d, e):
        self.id = Oscillator._id
        self.alfa = alfa
        self.mi = mi
        self.d = d
        self.e = e
        self.neighbours = []
        Oscillator._id += 1

    def params(self):
        self.params = [self.alfa, self.mi, self.d, self.e]

    def add(self, k):
        self.neighbours.append(k)


class sample:
    zmienna = 0
    _z = 0
    def __init__(self):
        self.zmienna += 1
        sample._z += 1
print(sample.zmienna)
s1 = sample()
print(sample.zmienna)
print(s1.zmienna)
print(sample._z)
print(s1._z)