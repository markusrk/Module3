import numpy as np
import matplotlib.pyplot as plt
import os


# Keeps track of the different available decay functions
class DecayFunctions:

    def __init__(self):
        return

    def exp(self,time):
        return np.exp(-time)

    def lin(self,time):
        return 1-time/10000

    def __getitem__(self, item):
        return getattr(self,item)


# Handles graph creation. Currently only for TSP
class GraphMaker:

    def __init__(self,output_dir):
        self.output_dir = output_dir
        self.plot_no = len(os.listdir(output_dir))

    def show_plot(self,cities_x,cities_y,som_x,som_y):
        plt.scatter(cities_x,cities_y)
        plt.scatter(som_x,som_y,color="RED")
        plt.plot(som_x,som_y,color="GREEN")
        plt.show()
        return

    def save_plot(self,cities_x,cities_y,som_x,som_y):
        plt.scatter(cities_x,cities_y)
        plt.scatter(som_x,som_y,color="RED")
        plt.plot(som_x,som_y,color="GREEN")
        plt.savefig(self.output_dir+"/plot"+str(self.plot_no))
        self.plot_no += self.plot_no

        return


# Man class that represents the self organising map
class SOM:

    # Sets a directory to use for outputting files
    def get_output_dir(self):
        dirno = len(os.listdir("output/"))
        os.makedirs("output/"+str(dirno),exist_ok=False)
        return "output/"+str(dirno)


    def __init__(self,
                lr,
                decay_rate,
                input_size,
                output_size,
                decay_func,
                caseman,
                weight_init_range=None,
                draw_interval = 10,
                graph_int = 10
                ):
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_func = DecayFunctions()[decay_func]
        self.input_size = input_size
        self.output_size = output_size
        self.weights_init_range = weight_init_range
        self.draw_interval = draw_interval
        self.cman = caseman
        self.output_dir = self.get_output_dir()
        self.graph_maker = GraphMaker(self.output_dir)
        self.n_factor = 5 # Defines neighbourhood size. This equals approx 3 neighbours on each side
        self.graph_int = graph_int # Defines how often graphs are to be saved

        self.inlayer = np.ndarray(input_size)
        self.outlayer = np.ndarray(output_size)
        self.weights = np.random.rand(output_size,input_size)
        self.updated_lr = self.lr

    # Returns the neighbours of the node
    def neighbours(self, node_index,cutoff_lim=0.05):
        neighbours = []
        i = 1
        while True:
            distance_factor = np.exp((-i**2)/(self.n_factor**2))
            if distance_factor <= cutoff_lim: break
            neighbours.append((node_index+1, distance_factor))
            neighbours.append((node_index-1, distance_factor))


        return neighbours

    # Adjusts the weights of the input node, depending on the learning rate and distance from winning node.
    def adjust(self, input, node_index, distance):
        diff = input-self.weights[node_index]
        self.weights[node_index] = self.weights[node_index] + diff*distance*self.updated_lr

    # Trains the neural network on one input event
    def train(self, input):
        result = np.matmul(self.weights,input)
        winner = result.argmax()

        # Adjusts weights of winner node itself
        self.adjust(input,winner,1)

        # adjust weights of neighbours
        for (node,distance) in self.neighbours(winner):
            self.adjust(input,node,distance)

    # Executes a training session with <iteration> cases
    def run(self, iterations):
        for i in range(iterations):
            # Run one training iteration
            input = self.cman.next()
            self.train(input)

            # Updates learning rate before next iteration
            self.updated_lr = self.lr*self.decay_func()

            # Generate charts
            if i == self.graph_int:
                self.graph_maker.save_graph(self.cman.get_all_cases(),self.weights)

