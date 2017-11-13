import mnist as m
from casemanager import CaseManager
import matplotlib.pyplot as plt
import pickle
import pandas as pd
cm = CaseManager("mnist")


param = {"lr": 0.5,
            "input_size": 784,
            "output_size": 196,
            "decay_func": "exp",
            "caseman": cm,
            "decay_half_life": 1000,
            "n_factor": 5,
            "n_halftime": 500,
            "graph_int": 10,
            "video": False,
            "output_dir": None,
            "nodes_per_row": 14
}

som = m.SOM(**param)
som.run(100)




def test_all(iterations_per_problem,out_dir=None):
    path_lengths = []
    output_dirs = []
    problem_no = []
    results = pd.DataFrame()
    #results = pd.read_csv("results.csv")
    for i in range(9,10):
        print("Starting on problem number "+str(i))
        cm = CaseManager(str(i))
        if out_dir: out_dir = out_dir + str(i)
        param = {"lr": 0.5,
                 "input_size": 2,
                 "output_size": 200,
                 "decay_func": "exp",
                 "caseman": cm,
                 "decay_half_life": 1000,
                 "n_factor": 5,
                 "n_halftime": 500,
                 "graph_int": 10,
                 "video": True,
                 "output_dir": out_dir
                 }
        som = m.SOM(**param)
        path_length,output_dir = som.run(iterations_per_problem)
        param["path_length"] = path_length
        param["output_dir"] = output_dir
        param["problem_no"] = i
        results = results.append(param,ignore_index=True)
    results.to_csv("results.csv")


def test_load_function():
    som = None
    with open("output/testttt9/SOM.pkl",'rb') as f:
        som = pickle.load(f)
        som.video = False
        som.save = True
        som.run(100)


#test_all(100,out_dir="csv_test2")



#test_all(4000)
#test_load_function()
#decay = m.DecayFunctions()
#print(decay["exp"](0.1))



#gm = m.GraphMaker("output")
#gm.save_plot(*cm.get_xy_separate(),[1,200],[1,200])

