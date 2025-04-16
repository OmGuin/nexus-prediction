import optuna.visualization as vis
import pickle

with open("study.pkl", "rb") as file:
    study = pickle.load(file)

vis.plot_optimization_history(study).show()


vis.plot_parallel_coordinate(study).show()


vis.plot_contour(study).show()


vis.plot_slice(study).show()


vis.plot_param_importances(study).show()