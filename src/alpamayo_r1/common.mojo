from python import Python

def get_torch() -> PythonObject:
    return Python.import_module("torch")

def get_np() -> PythonObject:
    return Python.import_module("numpy")

def get_einops() -> PythonObject:
    return Python.import_module("einops")

def get_scipy_rot() -> PythonObject:
    return Python.import_module("scipy.spatial.transform").Rotation
