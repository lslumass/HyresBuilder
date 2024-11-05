import pkg_resources as pkg_res

def load_ff(model='protein'):
    if model == 'protein':
        path1 = pkg_res.resource_filename("HyresBuilder", "top_hyres_GPU.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "param_hyres_GPU.inp")
    elif model == 'RNA':
        path1 = pkg_res.resource_filename("HyresBuilder", "top_RNA.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "param_RNA.inp")
    elif model == 'ATP':
        path1 = pkg_res.resource_filename("HyresBuilder", "top_ATP.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "param_ATP.inp")
    else:
        print("Error: The model type {} is not supported, choose from protein, RNA, ATP.".format(model))
        exit(1)
    
    top_inp, param_inp = str(path1), str(path2)
    return top_inp, param_inp
