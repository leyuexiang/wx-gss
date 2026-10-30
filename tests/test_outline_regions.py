"""顶底诊断区域应分离且不把共同罐身误计为补回盖顶。"""

import unittest
import numpy as np
from scripts.evaluate_outline_regions import region_masks


class OutlineRegionTests(unittest.TestCase):
    def test_separate_added_cap_removed_base_and_common_body(self):
        old = np.zeros((100, 60), bool)
        new = old.copy()
        old[30:95, 10:50] = True
        new[10:90, 10:50] = True
        regions = region_masks(old, new)
        self.assertTrue(regions['added_cap'][15, 30])
        self.assertTrue(regions['removed_base'][92, 30])
        self.assertTrue(regions['common_body'][50, 30])
        self.assertFalse(regions['added_cap'][50, 30])
        self.assertFalse((regions['added_cap'] & regions['removed_base']).any())

    def test_empty_or_mismatched_mask_rejected(self):
        for old, new in ((np.zeros((10, 10), bool), np.zeros((10, 10), bool)),
                         (np.ones((10, 10), bool), np.ones((8, 8), bool))):
            with self.assertRaises(ValueError):
                region_masks(old, new)


if __name__ == '__main__':
    unittest.main()
