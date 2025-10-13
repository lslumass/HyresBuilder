import MDAnalysis as mda
from psfgen import PsfGen
import os
from .utils import load_ff


def set_terminus(gen, segid, charge_status):
    # re-set the charge status of terminus
    if segid.startswith("P"):
        nter, cter = gen.get_resids(segid)[0], gen.get_resids(segid)[-1]
        if charge_status == 'charged':
            gen.set_charge(segid, nter, "N", 1.00)
            gen.set_charge(segid, cter, "O", -1.00)
        elif charge_status == 'NT':
            gen.set_charge(segid, nter, "N", 1.00)
        elif charge_status == 'CT':
            gen.set_charge(segid, cter, "O", -1.00)
        elif charge_status == 'positive':
            gen.set_charge(segid, nter, "N", -1.00)
            gen.set_charge(segid, cter, "O", -1.00)
        else:
            print("Error: Only 'neutral', 'charged', 'NT', and 'CT' charge status are supported.")
            exit(1)
            
def at2hyres(pdb_in, pdb_out):
    '''
    at2hyres: convert all-atom protein to hyres cg pdb
    pdb_in: input all-atom pdb file
    pdb_out: output hyres cg pdb file
    '''
    # read input pdb file
    data, tmp = {}, {}
    natom, nres, pre, iatom = 0, 1, 1, 0
    with open(pdb_in, 'r') as f1:
        for l in f1:
            if l.startswith("REMARK"):
               pass
            elif l.startswith("ATOM"):
                 natom=natom+1
                 if l[22:26].strip() == str(pre):
                    iatom=iatom+1
                    tmp[iatom]=[l[:4].strip(), l[4:11].strip(), l[11:16].strip(), l[17:20].strip(), l[20:22].strip(), l[22:26].strip(), l[30:38].strip(), l[38:46].strip(), l[46:54].strip(), l[54:60].strip(), l[60:66].strip(), l[66:77].strip()]
                 elif l[22:26].strip() != str(pre):
                    data[nres]=tmp
                    nres=nres+1
                    pre=l[22:26].strip()
                    iatom=1
                    tmp={}
                    tmp[iatom]=[l[:4].strip(), l[4:11].strip(), l[11:16].strip(), l[17:20].strip(), l[20:22].strip(), l[22:26].strip(), l[30:38].strip(), l[38:46].strip(), l[46:54].strip(), l[54:60].strip(), l[60:66].strip(), l[66:77].strip()]

            else:
               pass
            data[nres]=tmp
    print("There are",natom,"atoms /",nres,"residues")

    # rename HSD, HSE, HSP as HIS (HYRES ONLY RECOGNIZES NEUTRAL HIS)
    for i in range(1,nres+1):
       if data[i][1][3] in ['HSD', 'HSE', 'HSP']:
          for j in range(1, len(data[i])+1):
             data[i][j][3] = 'HIS'

    # mapping rules
    reslist=['amn','cbx','gly', 'ala', 'val', 'leu', 'ile', 'met', 'asn', 'asp', 'gln', 'glu', 'cys', 'ser', 'thr', 'pro', 'lys', 'arg', 'his', 'phe', 'tyr', 'trp']
    single=['ala', 'val', 'leu', 'ile', 'met', 'asn', 'asp', 'gln', 'glu', 'cys', 'ser', 'thr', 'pro']
    def maprule(resname):
        nsc, sc1, sc2, sc3, sc4, sc5, bb, nter, cter = 0, [], [], [], [], [], [], [], []
        if resname in ['amn', 'cbx']:
           nter=['CAY', 'HY1', 'HY2', 'HY3', 'CY', 'OY']
           cter=['NT', 'HNT', 'CAT', 'HT1', 'HT2', 'HT3']
        if resname in reslist and resname not in ['pro', 'gly', 'amn', 'cbx']:
           bb=['CA', 'HA', 'C', 'O', 'N', 'HN']
        elif resname == 'gly':
           bb=['CA', 'HA1', 'HA2', 'C', 'O', 'N', 'HN']
        elif resname == 'pro':
           bb=['CA', 'HA', 'C', 'O', 'N']
        if resname == 'lys':
           nsc=2
           sc1=['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2', 'CD', 'HD1', 'HD2']
           sc2=['CE', 'HE1', 'HE2', 'NZ', 'HZ1', 'HZ2', 'HZ3']
        elif resname == 'arg':
           nsc=2
           sc1=['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2', 'CD', 'HD1', 'HD2']
           sc2=['NE', 'HE', 'CZ', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22']
        elif resname == 'his':
           nsc=3
           sc1=['CB', 'HB1', 'HB2', 'CG']
           sc2=['CD2', 'HD2', 'NE2']
           sc3=['ND1', 'HD1', 'CE1', 'HE1']
        elif resname == 'phe':
           nsc=3
           sc1=['CB', 'HB1', 'HB2', 'CG', 'CD1', 'HD1']
           sc2=['CD2', 'HD2', 'CE2', 'HE2']
           sc3=['CE1', 'HE1', 'CZ', 'HZ']
        elif resname == 'tyr':
           nsc=3
           sc1=['CB', 'HB1', 'HB2', 'CG', 'CD1', 'HD1']
           sc2=['CD2', 'HD2', 'CE2', 'HE2']
           sc3=['CE1', 'HE1', 'CZ', 'OH', 'HH']
        elif resname == 'trp':
           nsc=5
           sc1=['CB', 'HB1', 'HB2', 'CG']
           sc2=['CD1', 'HD1', 'NE1', 'HE1']
           sc3=['CD2', 'CE2']
           sc4=['CZ2', 'HZ2', 'CH2', 'HH2']
           sc5=['CE3', 'HE3', 'CZ3', 'HZ3']
        elif resname in single:
           nsc=1
        elif resname in ['gly', 'amn', 'cbx']:
           nsc=0
        return nsc,bb,nter,cter,sc1,sc2,sc3,sc4,sc5

    # output in pdb-format
    with open(pdb_out, 'w') as f2:
        def printcg(atom):
            f2.write("%4s  %5d %2s   %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %4s\n" % (atom[0],int(atom[1]),atom[2],atom[3][:3],atom[4], int(atom[5]),float(atom[6]),float(atom[7]),float(atom[8]),float(atom[9]),float(atom[10]), atom[11]))

        # convert at to cg
        bbcg1=['CA', 'N', 'HN', 'HT1']        # should be modified further
        bbcg2=['C', 'O']
        ntercg=['CAY', 'CY', 'OY']
        ctercg=['NT', 'HNT', 'CAT']
        inx=0
        for ires in range(1,nres+1,1):   # loop over all residues
            iresname=data[ires][1][3].lower()  # residue name in lowercase
            iresatnum=len(data[ires].keys())   # number of atoms in the residue
            if iresname in reslist:
               # mapping rules
               (nsc,bb,nter,cter,sc1,sc2,sc3,sc4,sc5)=maprule(str(iresname))
               if iresname in single:
                  sc1=[item[2] for item in data[ires].values() if item[2] not in bb if item[2] not in nter if item[2] not in cter]
               # initializing
               coor={}
               num={}
               for i in range(1,6,1):
                   coor[i]=[0,0,0]
                   num[i]=0
               # compute com for each sc bead
               for j in range(1,iresatnum+1,1): # loop over all atoms in the residue
                   if data[ires][j][2] in ntercg:
                       inx += 1
                       data[ires][j][1]=inx
                       nt = data[ires][j]
                       if data[ires][j][2] == 'CAY':
                          nt[2]='CL'
                       elif data[ires][j][2] == 'CY':
                          nt[2]='C'
                       elif data[ires][j][2] == 'OY':
                          nt[2]='O'
                       printcg(nt)
                   elif data[ires][j][2] in ctercg:
                       inx += 1
                       data[ires][j][1]=inx
                       ct = data[ires][j]
                       if data[ires][j][2] == 'NT':
                          ct[2]='N'
                       elif data[ires][j][2] == 'HNT':
                          ct[2]='H'
                       elif data[ires][j][2] == 'CAT':
                          ct[2]='CA'
                       printcg(ct)
                   elif data[ires][j][2] in bbcg1:
                      inx=inx+1
                      if data[ires][j][2] in ['HN', 'HT1']:
                         data[ires][j][2]='H'
                      data[ires][j][1]=inx
                      data[ires][j][3]=data[ires][j][3]+'_'
                      printcg(data[ires][j])
                   elif data[ires][j][2] in sc1:
                      num[1]=num[1]+1
                      coor[1][0]=coor[1][0]+float(data[ires][j][6]) # x
                      coor[1][1]=coor[1][1]+float(data[ires][j][7]) # y
                      coor[1][2]=coor[1][2]+float(data[ires][j][8]) # z
                   elif data[ires][j][2] in sc2:
                      num[2]=num[2]+1
                      coor[2][0]=coor[2][0]+float(data[ires][j][6]) # x
                      coor[2][1]=coor[2][1]+float(data[ires][j][7]) # y
                      coor[2][2]=coor[2][2]+float(data[ires][j][8]) # z
                   elif data[ires][j][2] in sc3:
                      num[3]=num[3]+1
                      coor[3][0]=coor[3][0]+float(data[ires][j][6]) # x
                      coor[3][1]=coor[3][1]+float(data[ires][j][7]) # y
                      coor[3][2]=coor[3][2]+float(data[ires][j][8]) # z
                   elif data[ires][j][2] in sc4:
                      num[4]=num[4]+1
                      coor[4][0]=coor[4][0]+float(data[ires][j][6]) # x
                      coor[4][1]=coor[4][1]+float(data[ires][j][7]) # y
                      coor[4][2]=coor[4][2]+float(data[ires][j][8]) # z
                   elif data[ires][j][2] in sc5:
                      num[5]=num[5]+1
                      coor[5][0]=coor[5][0]+float(data[ires][j][6]) # x
                      coor[5][1]=coor[5][1]+float(data[ires][j][7]) # y
                      coor[5][2]=coor[5][2]+float(data[ires][j][8]) # z
               for i in range(1,nsc+1,1):
                   inx=inx+1
                   if i == 1:
                      name='CB'
                   elif i == 2:
                      name='CC'
                   elif i == 3:
                      name='CD'
                   elif i == 4:
                      name='CE'
                   elif i == 5:
                      name='CF'
                   coor[i][0]=coor[i][0]/num[i]
                   coor[i][1]=coor[i][1]/num[i]
                   coor[i][2]=coor[i][2]/num[i]
                   tmp=[data[ires][1][0],inx,name,data[ires][1][3], data[ires][1][4], data[ires][1][5],coor[i][0],coor[i][1],coor[i][2],data[ires][1][9],data[ires][1][10],data[ires][1][11],]
                   printcg(tmp)
               # this is to make sure cg atoms are in correct order
               for j in range(1,iresatnum+1,1): # loop over all atoms in the residue
                   if data[ires][j][3] not in ['AMN', 'CBX'] and data[ires][j][2] in bbcg2:
                      inx=inx+1
                      data[ires][j][1]=inx
                      data[ires][j][3]=data[ires][j][3]+'_'
                      printcg(data[ires][j])
            else:
               print(iresname,"is not recognized" )
               quit()

        f2.write("%3s\n" % ("END"))
    print("At2Hyres conversion done, output written to", pdb_out)

def at2icon(pdb_in, pdb_out):
   '''
    at2icon: convert all-atom RNA to iConRNA pdb
    pdb_in: input all-atom pdb file
    pdb_out: output iConRNA pdb file
   '''
   # output in pdb-format
   def printcg(atom, f):
       f.write("%4s  %5d %2s   %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %4s\n" % (atom[0],int(atom[1]),atom[2],atom[3][:3],atom[4], int(atom[5]),float(atom[6]),float(atom[7]),float(atom[8]),float(atom[9]),float(atom[10]), atom[11]))
   
   def aa2cg(sel, resid, segid, cg_bead):
       atom_grp = u.select_atoms(sel)
       com = atom_grp.center_of_mass()
       atom = ['ATOM', idx, cg_bead, res.resname, 'X', resid, com[0], com[1], com[2], 1.00, 0.00, segid]
       printcg(atom, f)
   
   u = mda.Universe(pdb_in)
   idx = 1
   with open(pdb_out, 'w') as f:
       for segment in u.segments:
           segid = segment.segid
           for res in segment.residues:
               resname = res.resname
               resid = res.resid
               sel = f"(name P O1P O2P O5' and resid {resid} and segid {segid}) or (name O3' and resid {resid-1} and segid {segid})"
               aa2cg(sel, resid, segid, 'P')
               idx += 1
               sel = f"name C4' and resid {resid} and segid {segid}"
               aa2cg(sel, resid, segid, 'C1')
               idx += 1
               sel = f"name C1' and resid {resid} and segid {segid}"
               aa2cg(sel, resid, segid, 'C2')
               idx += 1
               if resname == 'ADE':
                   sel = f"name N9 C4 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NA')
                   idx += 1
                   sel = f"name C8 H8 N7 C5 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NB')
                   idx += 1
                   sel = f"name C6 N1 N6 H61 H62 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NC')
                   idx += 1
                   sel = f"name C2 H2 N3 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'ND')
                   idx += 1
               elif resname == 'GUA':
                   sel = f"name N9 C4 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NA')
                   idx += 1
                   sel = f"name C8 H8 N7 C5 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NB')
                   idx += 1
                   sel = f"name C6 N1 N6 H1 O6 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NC')
                   idx += 1
                   sel = f"name C2 N2 H21 H22 N3 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'ND')
                   idx += 1
               elif resname == 'CYT':
                   sel = f"name N1 C5 H5 C6 H6 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NA')
                   idx += 1
                   sel = f"name C4 N4 H41 H42 N3 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NB')
                   idx += 1
                   sel = f"name C2 O2 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NC')
                   idx += 1
               elif resname == 'URA':
                   sel = f"name N1 C5 H5 C6 H6 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NA')
                   idx += 1
                   sel = f"name C4 O4 N3 H3 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NB')
                   idx += 1
                   sel = f"name C2 O2 and resid {resid} and segid {segid}"
                   aa2cg(sel, resid, segid, 'NC')
                   idx += 1
       print('END', file=f)
   print('At2iCon conversion done, output written to', pdb_out)

def at2cg(pdb_in, pdb_out, charge_status='neutral'):
   '''
    at2cg: convert all-atom pdb to cg pdb, either hyres for protein or iConRNA for RNA
    in the input pdb, protein segid should start with "P", RNA segid should start with "R"
    pdb_in: input all-atom pdb file
    pdb_out: output cg pdb file
   charge_status: charge status of protein terminus, only 'neutral' and 'charged' are supported, default is 'neutral'
   '''
   # set up psfgen
   # load topology files
   RNA_topology, _ = load_ff('RNA')
   protein_topology, _ = load_ff('Protein')
   gen = PsfGen()
   gen.read_topology(RNA_topology)
   gen.read_topology(protein_topology)

   # convert pdb
   u = mda.Universe(pdb_in)
   segids = u.residues.segments.segids
   segnum = len(segids)
   if segnum == 1:
         if segids[0].startswith("P"):
            at2hyres(pdb_in, pdb_out)
            gen.add_segment(segid=segid, pdbfile=pdb_out, auto_angles=False)
         elif segids[0].startswith("R"):
            at2icon(pdb_in, pdb_out)
            gen.add_segment(segid=segid, pdbfile=pdb_out, auto_angles=False, auto_dihedrals=False)
         else:
            print("Error: Only protein or RNA is supported.")
            exit(1)
   elif segnum > 1:
       for i, segid in enumerate(segids):
           sel = u.select_atoms(f"segid {segid}")
           tmp_pdb = f'tmp_{segid}.pdb'
           tmp_cg_pdb = f'tmp_cg_{segid}.pdb'
           sel.atoms.write(tmp_pdb)
           if segid.startswith("P"):
               at2hyres(tmp_pdb, tmp_cg_pdb)
               gen.add_segment(segid=segid, pdbfile=tmp_cg_pdb, auto_angles=False)
           elif segid.startswith("R"):
               at2icon(tmp_pdb, tmp_cg_pdb)
               gen.add_segment(segid=segid, pdbfile=tmp_cg_pdb, auto_angles=False, auto_dihedrals=False)
           else:
               print("Error: Only protein-protein or protein-RNA complex is supported.")
               exit(1)
       gen.write_pdb(pdb_out)
       print("Complex conversion done, output written to", pdb_out)
   else:
       print("Error: No segment found.")
       exit(1)
   
   #e-set the charge status of terminus
   for segid in gen.get_segids():
       if charge_status != "neutral":
           set_terminus(gen, segid, charge_status)    
   
   # write psf file
   gen.write_psf(filename=f'{pdb_out[:-4]}.psf')
   print("PSF file written to", f'{pdb_out[:-4]}.psf')
   # clean up temporary files
   for file in os.listdir():
       if file.startswith("tmp_") and file.endswith(".pdb"):
           os.remove(file)
