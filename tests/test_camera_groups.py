"""验证组内共享、跨组隔离、裁切主点保留和错误分组阻断。"""

import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.capture.camera_groups import validate_feature_groups, audit_feature_groups


class CameraGroupTests(unittest.TestCase):
    def fixture(self, root):
        images = root / "images"
        images.mkdir()
        for name in ("a.png", "b.png", "top.png", "bottom.png"):
            Image.new("RGB", (64, 48)).save(images / name)
        groups = []
        for i, names in enumerate((("a.png", "b.png"), ("top.png",), ("bottom.png",))):
            listing = root / f"group{i}.txt"
            listing.write_text("\n".join(names) + "\n", encoding="utf-8")
            # 主点在裁切图外也是有效标定，跨组参数相同仍必须保持相机独立。
            groups.append({"image_list": str(listing), "camera_params": [100, 80, 19, 0]})
        return images, groups

    def database(self, path, assignments):
        with closing(sqlite3.connect(path)) as c, c:
            c.execute("CREATE TABLE images(name TEXT, camera_id INTEGER)")
            c.execute("CREATE TABLE cameras(camera_id INTEGER, width INTEGER, height INTEGER, params BLOB)")
            c.executemany("INSERT INTO images VALUES (?,?)", assignments)
            for camera_id in {item[1] for item in assignments}:
                c.execute("INSERT INTO cameras VALUES (?,?,?,?)", (camera_id, 64, 48,
                    np.array([100, 80, 19, 0], dtype="<f8").tobytes()))

    def test_shared_group_and_independent_identical_intrinsics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, groups = self.fixture(root)
            checked = validate_feature_groups(groups, images, "SIMPLE_RADIAL")
            self.database(root / "database.db", [("a.png", 1), ("b.png", 1), ("top.png", 2), ("bottom.png", 3)])
            report = audit_feature_groups(root / "database.db", checked)
            self.assertEqual(report["group_count"], 3)
            self.assertEqual(report["groups"][0]["initial_camera_params"][1], 80)

    def test_reject_per_frame_camera_and_cross_group_sharing(self):
        for assignments in ([("a.png", 1), ("b.png", 2), ("top.png", 3), ("bottom.png", 4)],
                            [("a.png", 1), ("b.png", 1), ("top.png", 1), ("bottom.png", 3)]):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                images, groups = self.fixture(root)
                checked = validate_feature_groups(groups, images, "SIMPLE_RADIAL")
                self.database(root / "database.db", assignments)
                with self.assertRaisesRegex(ValueError, "未共享|跨组"):
                    audit_feature_groups(root / "database.db", checked)

    def test_reject_incomplete_duplicate_mismatched_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            images, groups = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "完整覆盖"):
                validate_feature_groups(groups[:1], images, "SIMPLE_RADIAL")
            with self.assertRaisesRegex(ValueError, "重复"):
                validate_feature_groups(groups + groups[:1], images, "SIMPLE_RADIAL")
            for value in (-1, float("nan"), True):
                groups[0]["camera_params"][0] = value
                with self.assertRaises(ValueError):
                    validate_feature_groups(groups, images, "SIMPLE_RADIAL")
            groups[0]["camera_params"][0] = 100
            Image.new("RGB", (65, 48)).save(images / "b.png")
            with self.assertRaisesRegex(ValueError, "尺寸必须一致"):
                validate_feature_groups(groups, images, "SIMPLE_RADIAL")


if __name__ == "__main__":
    unittest.main()
