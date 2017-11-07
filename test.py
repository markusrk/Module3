import main as m
from casemanager import CaseManager
import subprocess

cm = CaseManager("1")

param = {"lr": 0.5,
            "input_size": 2,
            "output_size": 200,
            "decay_func": "exp",
            "caseman": cm,
            "decay_half_life": 1000,
            "n_factor": 3,
            "n_halftime": 500,
            "graph_int": 10,
            "video": True
}

som = m.SOM(**param)
som.run(4000)


#decay = m.DecayFunctions()
#print(decay["exp"](0.1))



#gm = m.GraphMaker("output")
#gm.save_plot(*cm.get_xy_separate(),[1,200],[1,200])