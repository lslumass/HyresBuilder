import pkg_resources as pkg_res

def load_ff(model='protein'):
    for m in model:
        if m == 'protein':
            path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_GPU.inp")
            path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_GPU.inp")
        elif m == 'RNA':
            path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA.inp")
            path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA.inp")
        elif m == 'ATP':
            path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_ATP.inp")
            path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_ATP.inp")
        else:
            print("Error: The model type {} is not supported, choose from protein, RNA, ATP.".format(model))
            exit(1)
    top_inp = str(path1)
    param_inp = str(path2)

    return top_inp, param_inp
