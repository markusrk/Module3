import main as m
from casemanager import CaseManager
import matplotlib.pyplot as plt
import pickle
import pandas as pd
cm = CaseManager("1")


param = {"lr": 0.5,
            "input_size": 2,
            "output_size": 260,
            "decay_func": "exp",
            "caseman": cm,
            "decay_half_life": 1000,
            "n_factor": 20,
            "n_halftime": 500,
            "graph_int": 10000,
            "video": False,
            "output_dir": None,
            "print_interval" : 10000
}

som = m.SOM(**param)
print(som.run(4000))




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
                 "n_factor": 40,
                 "n_halftime": 1000,
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

