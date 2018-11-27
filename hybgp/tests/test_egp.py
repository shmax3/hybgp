import unittest
import operator
from ..egp import PrimitiveSet


class TestPrimitiveTree(unittest.TestCase):

    def setUp(self):
        pass

    def test_str(self):
        pass


class TestPrimitiveSet(unittest.TestCase):

    def setUp(self):
        self.basic = PrimitiveSet('main')

    def test_addOperator(self):
        self.basic.addOperator(operator.add, 2, 'Add')
        self.assertEqual(len(self.basic.primitives), 1)

    def test_addFunction(self):
        self.basic.addFunction(operator.neg, '-')
        self.assertIn('-', self.basic.context)
        self.assertEqual(self.basic.context['-'], operator.neg)

    def test_new_var(self):
        var = self.basic.new_var()
        self.assertListEqual([var], self.basic.variables)

    def test_var_num(self):
        self.assertEqual(self.basic.var_num, 0)
        self.basic.new_var()
        self.assertEqual(self.basic.var_num, 1)


class TestCompile(unittest.TestCase):

    def setUp(self):
        pass

    def test_compile(self):
        pass
