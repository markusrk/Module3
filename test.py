import main as m
from casemanager import CaseManager

cm = CaseManager("8")

param = {"lr": 0.1,
            "decay_rate": 0.1,
            "input_size": 2,
            "output_size": 100,
            "decay_func": "exp",
            "caseman": cm
}

som = m.SOM(**param)
som.run(100)
print(som.weights)
#decay = m.DecayFunctions()
#print(decay["exp"](0.1))



#gm = m.GraphMaker("output")
#gm.save_plot(*cm.get_xy_separate(),[1,200],[1,200])