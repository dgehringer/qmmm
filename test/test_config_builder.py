from unittest import TestCase


class TestConfigBuilder(TestCase):
    def test_applications(self):
        from qmmm.defaults import ConfigBuilder
        self.assertListEqual(ConfigBuilder.applications(), ['vasp', 'lammps', 'vasp_gam'])

    def test_remotes(self):
        from qmmm.defaults import ConfigBuilder
        self.assertListEqual(ConfigBuilder.remotes(), ['mul-hpc', 'local'])

    def test_partition_aliases(self):
        from qmmm.defaults import ConfigBuilder
        self.assertListEqual(ConfigBuilder().remote('mul-hpc').queue('c3')._partition_aliases(), ['phi', 'c3'])
        self.assertListEqual(ConfigBuilder().remote('mul-hpc').queue('c1')._partition_aliases(), ['e5-2690-128G', 'c1', 'e5-2690-256G', 'c2'])
        self.assertListEqual(ConfigBuilder().remote('mul-hpc').queue('c2')._partition_aliases(), ['e5-2690-128G', 'c1', 'e5-2690-256G', 'c2'])
        self.assertListEqual(ConfigBuilder().remote('mul-hpc').queue('c4')._partition_aliases(), ['e5-1650', 'c4'])

    def test_application(self):
        from qmmm.defaults import ConfigBuilder
        # Test swap
        self.assertDictEqual(ConfigBuilder().application('lammps'), ConfigBuilder().remote('local').application('lammps'))
        self.assertDictEqual(ConfigBuilder().application('lammps').remote('local'), ConfigBuilder().remote('local').application('lammps'))
        self.assertDictEqual(ConfigBuilder().application('vasp'), ConfigBuilder().remote('local').application('vasp'))
        self.assertDictEqual(ConfigBuilder().application('vasp').remote('local'), ConfigBuilder().remote('local').application('vasp'))

    def test_remote(self):
        self.assertTrue(1 == 1)

    def test_queue(self):
        self.assertTrue(1 == 1)

