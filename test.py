import main as m

param = {"lr": 0.1,
            "decay_rate": 0.1,
            "input_size": 2,
            "output_size": 100,
            "decay_func": "exp"
}

som = m.SOM(**param)

print(som.outlayer)

