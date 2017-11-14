import random

class MNIST_holder:


    def __init__(self, train_cases=100,test_cases=100):
        from tensorflow.examples.tutorials.mnist import input_data
        self.train_cases = train_cases
        self.mnist = input_data.read_data_sets("datasets/MNIST/", one_hot=True)
        self.train_features, self.train_labels = self.mnist.train.next_batch(self.train_cases)
        self.test_features, self.test_labels = self.mnist.test.next_batch(self.test_cases)

    def train_full_batch(self):
        return self.train_features, self.train_labels

    def test_full_batch(self):
        return self.test_features, self.test_labels


class CaseManager:
    def __init__(self, dataset):
        if dataset == "mnist":
            cases = 1000
            self.mnist = True
            self.train_features, self.train_labels = MNIST_holder(train_cases=cases).train_full_batch()
            self.test_features, self.test_labels = MNIST_holder(train_cases=cases).test_full_batch()
            self.no_of_cities = cases
        else:
            self.mnist=False
            with open("datasets/"+dataset+".txt","r") as f:
                _, self.no_of_cities = next(f).split(": ")
                #cast no_of_cities to int
                self.no_of_cities = int(self.no_of_cities)
                next(f)
                self.train_features = []
                for line in f:
                    if "EOF" in line: break
                    _,x,y = line.strip().split(" ")
                    x,y = float(x),float(y)
                    self.train_features.append((x, y))
            # Generate separat x,y coordinates, only used by graph_maker
            self.x = []
            self.y = []
            for city in self.train_features:
                self.x.append(city[0])
                self.y.append(city[1])

    def next(self):
        r = random.randint(0, self.no_of_cities - 1)
        if self.mnist:
            return self.train_features[r], self.train_labels[r]
        return self.train_features[r]

    def get_all_cases(self):
        if self.mnist:
            return self.train_features, self.train_labels
        return self.train_features

    def get_all_test_cases(self):
        return self.test_features,self.test_labels

    def get_xy_separate(self):
        return self.x,self.y

    def center(self):
        return sum(self.x)/len(self.x),sum(self.y)/len(self.y)

    def minmax(self):
        return min(self.x),max(self.x),min(self.y),max(self.y)


