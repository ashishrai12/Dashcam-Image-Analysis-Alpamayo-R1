from python import Python
from ..common import get_torch

struct FlowMatching:
    var py_model: PythonObject

    fn __init__(inout self, num_steps: Int = 10):
        let py_code = """
import torch
class FlowMatching:
    def __init__(self, num_steps):
        self.num_inference_steps = num_steps
    
    def sample(self, batch_size, step_fn, device):
        x = torch.randn(batch_size, 64, 2, device=device)
        return x
"""
        let context = Python.dict()
        Python.run(py_code, context)
        self.py_model = context["FlowMatching"](num_steps)

    fn sample(self, batch_size: Int, step_fn: PythonObject, device: PythonObject) -> PythonObject:
        return self.py_model.sample(batch_size, step_fn, device)
