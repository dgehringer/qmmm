
import numpy as np
from pymatgen.io.vasp import Vasprun as PymatgenVasprun
from xml import etree

# Taken and modified from phonopy.interface.vasp.Vasprun


class Vasprun(object):

    def __init__(self, filename):
        # At first read the forces on the atoms
        try:
            self._fileptr = open(filename, 'r')
        except IOError:
            raise
        self._forces = self._read_forces()
        self._fileptr.close()
        self._vasprun = PymatgenVasprun(filename)


    @property
    def final_structure(self):
        return self._vasprun.final_structure

    @property
    def final_energy(self):
        return self._vasprun.final_energy

    @property
    def incar(self):
        return self._vasprun.incar

    @property
    def kpoints(self):
        return self._vasprun.kpoints

    @property
    def structures(self):
        return self._vasprun.structures

    @property
    def forces(self):
        return self._forces

    def __getattr__(self, item):
        # Fallback
        return getattr(self._vasprun, item)


    def _read_forces(self):
        vasprun_etree = self._parse_etree_vasprun_xml(tag='varray')
        return self._get_forces(vasprun_etree)

    def _get_forces(self, vasprun_etree):
        forces = []
        for event, element in vasprun_etree:
            if element.attrib['name'] == 'forces':
                for v in element:
                    forces.append([float(x) for x in v.text.split()])
        return np.array(forces)


    def _parse_etree_vasprun_xml(self, tag=None):
        return self._parse_by_etree(self._fileptr, tag=tag)

    def _parse_by_etree(self, fileptr, tag=None):
        import xml.etree.cElementTree as etree
        for event, elem in etree.iterparse(fileptr):
            if tag is None or elem.tag == tag:
                yield event, elem

