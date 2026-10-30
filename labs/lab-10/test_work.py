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

    def test_changes_preserve_weights_and_residual_shapes(self):
        from torchvision.models import resnet18, ResNet18_Weights
        ns = definitions('build_backbone')
        ns.update(resnet18=resnet18, ResNet18_Weights=ResNet18_Weights)
        with tempfile.TemporaryDirectory() as temporary:
            ns['DATA'] = Path(temporary)
            torch.manual_seed(42)
            state = resnet18(weights=None).state_dict()
            torch.save(state, Path(temporary) / 'resnet18-f37072fd.pth')
            # Locally generated fixture: no network access or pretrained download.
            for variant in ['baseline', 'reflection', 'kernel3', 'stride1', 'dilation2']:
                model = ns['build_backbone'](variant)
                expected = state['conv1.weight']
                if variant == 'kernel3':
                    expected = expected[:, :, 2:5, 2:5]
                self.assertTrue(torch.equal(model.conv1.weight, expected))
                self.assertTrue(torch.equal(model.layer4[1].conv2.weight, state['layer4.1.conv2.weight']))
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                with torch.inference_mode():
                    self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 512))
                if variant == 'stride1':
                    self.assertEqual(model.layer2[0].conv1.stride, (1, 1))
                    self.assertEqual(model.layer2[0].downsample[0].stride, (1, 1))
                if variant == 'dilation2':
                    self.assertEqual(model.layer4[1].conv2.dilation, (2, 2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
