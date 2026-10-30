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

    def test_threshold_boundary(self):
        threshold = definitions('demo_simple_threshold')['demo_simple_threshold']
        pixels = np.array([[0, 119, 120, 121, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(threshold(pixels, 120), [[0, 0, 0, 255, 255]])

    def test_demo_image_range(self):
        image = definitions('create_threshold_demo_image')['create_threshold_demo_image']()
        self.assertEqual(image.shape, (240, 360))
        self.assertEqual(image.dtype, np.uint8)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
