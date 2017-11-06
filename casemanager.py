import random

class MNIST_holder:


    def __init__(self,cases=1000):
        from tensorflow.examples.tutorials.mnist import input_data
        self.cases = cases
        self.mnist = input_data.read_data_sets("datasets/MNIST/", one_hot=True)
        self.features, self.labels = self.mnist.train.next_batch(self.cases)

    def train_full_batch(self):
        return self.features, self.labels


class CaseManager:
    def __init__(self, dataset):
        if dataset == "mnist":
            # Todo
            return
        else:
            with open("datasets/"+dataset+".txt","r") as f:
                _, self.no_of_cities = next(f).split(": ")
                #cast no_of_cities to int
                self.no_of_cities = int(self.no_of_cities)
                next(f)
                self.cities = []
                for line in f:
                    if "EOF" in line: break
                    _,x,y = line.split(" ")
                    x,y = float(x),float(y)
                    self.cities.append((x,y))

    def next(self):
        return self.cities[random.randint(0,self.no_of_cities)]

    def get_all_cases(self):
        return self.cities

    def get_xy_separate(self):
        x = []
        y = []
        for city in self.cities:
            x.append(city[0])
            y.append(city[1])
        return x,y


