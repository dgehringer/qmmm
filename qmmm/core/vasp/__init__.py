
from pymatgen.io.vasp import PotcarSingle, Potcar
import re


def potcar_from_string(string):
    fdata = string

    potcar = Potcar()
    potcar_strings = re.compile(r"\n?(\s*.*?End of Dataset)",
                                re.S).findall(fdata)
    functionals = []
    for p in potcar_strings:
        single = PotcarSingle(p)
        potcar.append(single)
        functionals.append(single.functional)
    if len(set(functionals)) != 1:
        raise ValueError("File contains incompatible functionals!")
    else:
        potcar.functional = functionals[0]
    return potcar