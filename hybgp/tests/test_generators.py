import unittest
from ..generators import new_const, new_var


class TestGenerators(unittest.TestCase):

    def test_new_const(self):
        self.assertIn('c', new_const())
        self.assertNotEqual(new_const(), new_const())
        self.assertIsInstance(new_const(), str)

    def test_var_const(self):
        self.assertIn('x', new_var())
        self.assertNotEqual(new_var(), new_var())
        self.assertIsInstance(new_var(), str)

