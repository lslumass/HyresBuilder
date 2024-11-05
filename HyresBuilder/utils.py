import importlib.resources as pkg_res
from pathlib import Path

def load_ff(model='protein'):
    if model == 'protein':
        path1 = pkg_res.path("HyresBuilder", "top_hyres_GPU.inp")
        path2 = pkg_res.path("HyresBuilder", "param_hyres_GPU.inp")
    elif model == 'RNA':
        path1 = pkg_res.path("HyresBuilder", "top_RNA.inp")
        path2 = pkg_res.path("HyresBuilder", "param_RNA.inp")
    elif model == 'ATP':
        path1 = pkg_res.path("HyresBuilder", "top_ATP.inp")
        path2 = pkg_res.path("HyresBuilder", "param_ATP.inp")
    else:
        print("Error: The model type {} is not supported, choose from protein, RNA, ATP.".format(model))
        exit(1)
    
    top_inp, param_inp = Path(path1), Path(path2)
    return top_inp, param_inp
