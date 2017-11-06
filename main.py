import numpy as np


class SOM:

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
        self.decay_func = decay_func
        self.input_size = input_size
        self.output_size = output_size
        self.weights_init_range = weight_init_range
        self.draw_interval = draw_interval
        self.caseman = caseman

        self.inlayer = np.ndarray(input_size)
        self.outlayer = np.ndarray(output_size)
        self.weights = np.random.rand(input_size, output_size)
        self.updated_lr = self.lr

        # Returs the neighbours of the node
        def neighbours(node):
            neighbours = []
            return neighbours, distance

        # Adjusts the weights of the input node, depending on the learning rate and distance from winning node.
        def adjust(input,node,distance):
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


         def run(iterations,):
            for i in range(iterations):
                # Run one training iteration
                input = self.caseman.next()
                train(input)

                # Updates learning rate before next iteration
                self.updated_lr = self.updated_lr*decay_func()
