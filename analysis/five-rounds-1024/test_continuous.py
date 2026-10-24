"""多BS参数和证据隔离的CPU合同，不冒充NPU验证。"""
import unittest
from benchmark import case_label,resolve_cases


class ContinuousCases(unittest.TestCase):
    def test_max_capacity_is_explicit(self):
        self.assertEqual(resolve_cases(None,[32,96,4096],None),([32,96,4096],4096))
        self.assertEqual(resolve_cases(32,None,None),([32],32))
        self.assertEqual(resolve_cases(32,None,4096),([32],4096))

    def test_invalid_cases_rejected(self):
        for values,capacity in (([],None),([32,32],None),([33],None),([32,4096],32),([32],4097)):
            with self.subTest(values=values,capacity=capacity),self.assertRaises(ValueError):
                resolve_cases(None,values,capacity)

    def test_case_paths_preserve_existing_schema(self):
        self.assertEqual(case_label(32,'control-a'),'bs32-control-a')
        self.assertEqual(case_label(96,'control-a'),'bs96-control-a')
        self.assertEqual(case_label(32,'bs32-control-a'),'bs32-control-a')
        for label in ('../escape','bs96-control-a','preflight'):
            with self.subTest(label=label),self.assertRaises(ValueError):case_label(32,label)


if __name__=='__main__':unittest.main()
