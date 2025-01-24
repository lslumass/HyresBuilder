import pkg_resources as pkg_res

def load_ff(model='protein'):
    if model == 'protein':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_GPU.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_GPU.inp")
    elif model == 'protein_mix':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_hyres_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_hyres_mix.inp")
    elif model == 'RNA':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA.inp")
    elif model == 'RNA2':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA2.inp")
    elif model == 'RNA_mix':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_RNA_mix.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_RNA_mix.inp")
    elif model == 'ATP':
        path1 = pkg_res.resource_filename("HyresBuilder", "forcefield/top_ATP.inp")
        path2 = pkg_res.resource_filename("HyresBuilder", "forcefield/param_ATP.inp")
    else:
        print("Error: The model type {} is not supported, choose from protein, protein_mix, RNA, RNA2, RNA_mix, ATP.".format(model))
        exit(1)
    top_inp = str(path1)
    param_inp = str(path2)

    return top_inp, param_inp
