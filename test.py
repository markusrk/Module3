import main as m
from casemanager import CaseManager
import pickle
cm = CaseManager("4")

param = {"lr": 0.5,
            "input_size": 2,
            "output_size": 200,
            "decay_func": "exp",
            "caseman": cm,
            "decay_half_life": 1000,
            "n_factor": 3,
            "n_halftime": 500,
            "graph_int": 10,
            "video": True,
            "output_dir": None
}


def test_all(iterations_per_problem):
    for i in range(1,10):
        print("Starting on problem number "+str(i))
        cm = CaseManager(str(i))
        out_dir = "first_real_test" + str(i)
        param = {"lr": 0.5,
                 "input_size": 2,
                 "output_size": 200,
                 "decay_func": "exp",
                 "caseman": cm,
                 "decay_half_life": 1000,
                 "n_factor": 3,
                 "n_halftime": 500,
                 "graph_int": 10,
                 "video": True,
                 "output_dir": out_dir
                 }
        som = m.SOM(**param)
        som.run(iterations_per_problem)


def test_load_function():
    som = None
    with open("output/testttt9/SOM.pkl",'rb') as f:
        som = pickle.load(f)
        som.video = False
        som.save = True
        som.run(100)


test_all(4000)
#test_load_function()
#decay = m.DecayFunctions()
#print(decay["exp"](0.1))



#gm = m.GraphMaker("output")
#gm.save_plot(*cm.get_xy_separate(),[1,200],[1,200])

