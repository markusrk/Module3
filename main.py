import numpy as np
import matplotlib.pyplot as plt
import os


class DecayFunctions:

    def __init__(self):
        return

    def exp(self,time):
        return np.exp(-time)

    def lin(self,time):
        return 1-time/10000

    def __getitem__(self, item):
        return getattr(self,item)

class GraphMaker:

    def __init__(self,output_dir):
        self.output_dir = output_dir
        self.plot_no = len(next(os.walk(self.output_dir))[2])

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


class SOM:

    def get_output_dir(self):
        dirf = open("output/vars.txt", "rw")
        dir = dirf.readline()
        dirno = int(dir)

        import os.makedirs
        os.makedirs(str(dirno))

        dir.replace(str(dirno), str(dirno + 1))
        dirf.close()
        return str(dirno)

    def __init__(self,
                lr,
                decay_rate,
                input_size,
                output_size,
                decay_func,
                caseman,
                weight_init_range=None,
                draw_interval = 10
                ):
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_func = DecayFunctions[decay_func]
        self.input_size = input_size
        self.output_size = output_size
        self.weights_init_range = weight_init_range
        self.draw_interval = draw_interval
        self.cman = caseman
        self.output_dir = self.get_output_dir()
        self.grap_maker = GraphMaker(self.output_dir)

        self.inlayer = np.ndarray(input_size)
        self.outlayer = np.ndarray(output_size)
        self.weights = np.random.rand(input_size, output_size)
        self.updated_lr = self.lr

        # Returs the neighbours of the node
        def neighbours(node):
            neighbours = []
            return neighbours, distance

        # Adjusts the weights of the input node, depending on the learning rate and distance from winning node.
        def adjust(input, node, distance):
            diff = input-node
            node = node + diff*distance*self.updated_lr

        # trains the neural network on one input event
        def train(input):
            result = input*self.weights
            winner = result.amax()

            # Adjusts weights of winner node itself
            adjust(input,winner,1)

            # adjust weights of neighbours
            for (node,distance) in neighbours(winner):
                adjust(input,node,distance)

        # Executes a training session with <iteration> cases
        def run(iterations):
            for i in range(iterations):
                # Run one training iteration
                input = self.cman.next()
                train(input)

                # Updates learning rate before next iteration
                self.updated_lr = self.lr*self.decay_func()

                # Generate charts
                if i == self.graph_int:
                    self.graph_maker.save_graph(self.cman.get_all_cases(),self.weights)

