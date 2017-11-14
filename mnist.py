import os
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keeps track of the different available decay functions
class DecayFunctions:

    def __init__(self, factor):
        self.factor = factor
        return

    def exp(self,time):
        return np.exp(-time/self.factor)

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
        plt.scatter(cities_x,cities_y,color="BLUE")
        plt.scatter(som_x,som_y,color="RED")
        plt.plot(som_x,som_y,color="GREEN")
        plt.savefig(self.output_dir+"/plot"+str(self.plot_no))
        plt.close()
        self.plot_no += 1
        return

    def show_2d_plot(self,matrix):
        plt.matshow(matrix,cmap='Accent')
        plt.show()
        return

    def save_2d_plot(self,matrix):
        plt.matshow(matrix,cmap='tab20')
        plt.colorbar()
        plt.savefig(self.output_dir + "/plot" + str(self.plot_no))
        plt.close()
        self.plot_no += 1
        return

# Main class that represents the self organising map
class SOM:

    # Sets a directory to use for outputting files
    def get_output_dir(self,output_dir=None):
        if output_dir:
            os.makedirs("output/" + output_dir, exist_ok=False)
            return  "output/"+output_dir
        else:
            dirno = len(os.listdir("output/"))
            os.makedirs("output/"+str(dirno),exist_ok=False)
        return "output/"+str(dirno)


    def rand_init(self):
        weights = np.random.random_sample((self.output_size,self.input_size))
        weights = weights*0.2-0.1
        return weights


    def center_init(self):
        x_cen,y_cen = self.cman.center()
        weights = np.full((self.output_size,self.input_size),0)
        weights[:,0] = weights[:,0]+x_cen
        weights[:, 1] = weights[:, 1] +y_cen
        return weights

    # Returns a list representing which nodes are actually on top of a city and which that are not
    def active_nodes(self):
        cities = self.cman.get_all_cases()
        active = np.full(self.output_size,0)
        for city in cities:
            results = np.square(self.weights - city)
            summarized = results.sum(1)
            winner_index = summarized.argmin()
            active[winner_index] = 1
        return active

    # Calculates the pathlength of the current SOM position, after removing nodes not assigned to any city
    def path_length(self, return_nodes=False):
        total_length = 0.
        active_nodes = self.active_nodes()
        weights = deepcopy(self.weights)
        for i in range(len(active_nodes)-1,-1,-1):
            if not active_nodes[i]: weights = np.delete(weights,i,0)
        for i in range(1,len(weights)):
            x = np.abs(weights[i-1][0]-weights[i][0])
            y = np.abs(weights[i - 1][1] - weights[i][1])
            dist = np.sqrt(x**2+y**2)
            total_length += dist
        if return_nodes:
            return total_length,weights
        return total_length

    def __init__(self,
                 lr,
                 input_size,
                 output_size,
                 decay_func,
                 decay_half_life,
                 caseman,
                 n_factor,
                 n_halftime,
                 graph_int,
                 video = False,
                 output_dir = None,
                 save = True,
                 nodes_per_row = None

                 ):
        self.lr = lr
        self.decay_half_life = DecayFunctions(decay_half_life)[decay_func]
        self.input_size = input_size
        self.output_size = output_size
        self.cman = caseman
        self.output_dir = self.get_output_dir(output_dir)
        self.graph_maker = GraphMaker(self.output_dir)
        self.n_factor = n_factor # Defines neighbourhood size. This equals approx 3 neighbours on each side
        self.updated_n_factor = self.n_factor
        self.graph_int = graph_int # Defines how often graphs are to be saved
        self.n_halftime = n_halftime
        self.video = video
        self.nodes_per_row = nodes_per_row
        if not self.nodes_per_row:
            self.nodes_per_row = self.output_size
        self.classification_batch_size = 1000

        self.weights = self.rand_init()
        self.updated_lr = self.lr
        self.save = save

    # Returns the neighbours of the node
    def neighbours(self, node_index,cutoff_lim=0.05):
        neighbours = []
        i = 1
        while True:
            distance_factor = np.exp((-i**2/(self.updated_n_factor**2)))
            if distance_factor <= cutoff_lim: break
            neighbours.append(((node_index-i)%len(self.weights), distance_factor))
            neighbours.append(((node_index+i)%len(self.weights), distance_factor))
            if not self.nodes_per_row == self.output_size:
                neighbours.append(((node_index - i*self.nodes_per_row) % len(self.weights), distance_factor))
                neighbours.append(((node_index + i*self.nodes_per_row) % len(self.weights), distance_factor))
            i += 1
        return neighbours

    # Adjusts the weights of the input node, depending on the learning rate and distance from winning node.
    def adjust(self, input, node_index, distance):
        diff = input-self.weights[node_index]
        self.weights[node_index] = self.weights[node_index] + diff*distance*self.updated_lr

    # Trains the neural network on one input event
    def train(self, input):
        results = np.square(self.weights-input)
        summarized = results.sum(1)
        winner = summarized.argmin()

        # Adjusts weights of winner node itself
        self.adjust(input,winner,1)

        # adjust weights of neighbours
        for (node,distance) in self.neighbours(winner):
            self.adjust(input,node,distance)
        return winner

    def winner(self, input):
        results = np.square(self.weights-input)
        summarized = results.sum(1)
        winner = summarized.argmin()
        return winner

    def most_common(self,lst):
        try:
            return max(set(lst), key=lst.count)
        except ValueError:
            return -1

    def node_classes(self):
        node_classes = []
        correct = 0
        winner_lists = [[] for i in range(self.output_size)]
        features, labels = self.cman.get_all_cases()
        for i in range(len(features)):
            f,l = features[i],labels[i]
            results = np.square(self.weights - f)
            summarized = results.sum(1)
            winner = summarized.argmin()
            winner_lists[winner].append(np.argmax(l))
        for wins in winner_lists:
            most_common = self.most_common(wins)
            node_classes.append(most_common)
            correct += wins.count(most_common)
        return node_classes, correct/len(features)

    def test_accuracy(self):
        correct = 0
        node_classes = self.node_classes()
        features, labels = self.cman.get_all_test_cases()
        for i in range(len(features)):
            f,l = features[i],labels[i]
            results = np.square(self.weights - f)
            summarized = results.sum(1)
            winner = summarized.argmin()
            if l == node_classes(winner):
                correct += 1
        return correct/len(features)


    # Executes a training session with <iteration> cases
    def run(self, iterations):
        for i in range(iterations):
            # Run one training iteration
            feature,label = self.cman.next()
            winner = self.train(feature)

            # Updates learning rate and neighbourhood rates before next iteration
            self.updated_lr = self.lr*self.decay_half_life(i)
            self.updated_n_factor = self.n_factor*np.exp(-i/self.n_halftime)

            # Generate charts
            if i%self.graph_int == 0:
                if self.nodes_per_row == self.output_size:
                    self.graph_maker.save_plot(self.cman.x,self.cman.y,self.weights[:,0],self.weights[:,1])
                else:
                    node_class,win_rate = self.node_classes()
                    matrix = np.reshape(node_class, (-1, self.nodes_per_row))
                    self.graph_maker.save_2d_plot(matrix)
            # Print step number and accuracy
            if i%10 == 0:
                print("Currently on step: " + str(i))
                print("win rate: "+ str(win_rate))
            # Print test set accuracy
            if i%100 == 0:
                print('Test accuracy: '+str(self.test_accuracy()))



        # Finishing comments
        if self.nodes_per_row == self.output_size:
            pl,used_nodes = self.path_length(return_nodes=True)
            print("Path length= " + str(pl))
            self.graph_maker.save_plot(self.cman.x, self.cman.y, used_nodes[:, 0], used_nodes[:, 1])

        # Makes a video of all the image files
        if self.video:
            cwd = os.getcwd()
            outputdir = os.path.join(cwd, self.output_dir)
            os.chdir(outputdir)
            os.system("ffmpeg -r 30 -f image2 -s 640x480 -i plot%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p _video.mp4 > /dev/null")
            os.chdir(cwd)

        # Saves the SOM to a file
        if self.save:
            cwd = os.getcwd()
            outputdir = os.path.join(cwd, self.output_dir)
            os.chdir(outputdir)
            with open("SOM.pkl", "wb") as f:
                pickle.dump(self, f)
            os.chdir(cwd)
            df = pd.DataFrame()
            #df = pd.read_csv("running_res.csv")
            params = deepcopy(vars(self))
            if self.output_size == self.nodes_per_row:
                params["path_length"] = self.path_length()
            # Todo make this
            else:
                params["train_accuracy"] = "something"
                params["test_accuracy"] = "something"
            params["iterations"] = iterations
            df = df.append(params, ignore_index=True)
            df.to_csv("running_res.csv")

        if self.output_size != self.nodes_per_row:
            return "mnist", self.output_dir
        return self.path_length(), self.output_dir



