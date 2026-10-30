import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
import cv2

HERE = Path(__file__).resolve().parent

def definitions(*names):
    """Load tested notebook definitions without training or downloading data."""
    notebook = json.loads((HERE / 'work.ipynb').read_text())
    namespace = dict(np=np, torch=torch, nn=nn, cv2=cv2, Path=Path,
                     SEED=42, LEARNING_RATE=0.03, EPOCHS=5000,
                     POINTS_PER_CLASS=100, NUM_CLASSES=3)
    found = set()
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(module, str(HERE / 'work.ipynb'), 'exec'), namespace)
                found.add(node.name)
    if found != set(names):
        raise AssertionError(f'Missing definitions: {set(names) - found}')
    return namespace


class WorkTests(unittest.TestCase):
    def test_notebook_is_readable(self):
        notebook = json.loads((HERE / 'work.ipynb').read_text())
        self.assertEqual(notebook['nbformat'], 4)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertFalse(any(o.get('output_type') == 'error' for o in cell.get('outputs', [])))

    def test_area_perimeter_and_filtering(self):
        analyze = definitions('analyze_objects')['analyze_objects']
        image = np.zeros((60, 60), dtype=np.uint8)
        cv2.rectangle(image, (5, 5), (15, 15), 255, -1)
        objects = analyze(image, 100)
        self.assertEqual(len(objects), 1)
        _, area, perimeter, compact = objects[0]
        self.assertAlmostEqual(area, 100)
        self.assertAlmostEqual(perimeter, 40)
        self.assertAlmostEqual(compact, np.pi / 4)
        self.assertEqual(analyze(image, 101), [])
        count, _, stats, _ = cv2.connectedComponentsWithStats(image)
        self.assertEqual(count, 2)
        self.assertEqual(stats[1, cv2.CC_STAT_AREA], 121)

    def test_photo_is_available(self):
        self.assertIsNotNone(cv2.imread(str(HERE / 'ant.jpg')))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
