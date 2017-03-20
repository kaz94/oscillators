
class Oscillator:

    _id = 0

    def __init__(self, p):
        x, y, alfa, mi, d, e, f = p
        self.x = x
        self.y = y
        self.id = Oscillator._id
        self.alfa = alfa
        self.mi = mi
        self.d = d
        self.e = e
        self.f = f
        self.couplings = []
        self.id = Oscillator._id
        Oscillator._id += 1

    #def params(self):
        self.params = [self.alfa, self.mi, self.d, self.e]

    def addcoupl(self, c, k):
        self.couplings.append([c, k])


'''
def load_params():
    adj_list = read_file()
    oscillators = []
    for i, osc in enumerate(adj_list):
        # Oscillator(x, y, alfa, mi, d, e)
        o = Oscillator(osc[0:6])
        osclist = list(zip(osc[6::2], osc[7::2]))
        for i in osclist:
            o.addcoupl(i[0], i[1])
        oscillators.append(o)
    print(oscillators)
    return oscillators
'''




'''
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
'''
