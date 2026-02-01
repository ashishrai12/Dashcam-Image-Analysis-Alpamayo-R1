from python import Python
from .common import get_torch, get_einops

def unwrap_angle(phi: PythonObject) -> PythonObject:
    let torch = get_torch()
    let d = torch.diff(phi, dim=-1)
    # round_2pi_torch is locally defined or imported
    let rotation = Python.import_module("alpamayo_r1.geometry.rotation")
    let d_wrapped = rotation.round_2pi_torch(d)
    return torch.cat(Python.list(phi[":, :1"], phi[":, :1"] + torch.cumsum(d_wrapped, dim=-1)), dim=-1)

def first_order_D(N: Int, lead_shape: PythonObject, device: PythonObject, dtype: PythonObject) -> PythonObject:
    let torch = get_torch()
    let shape = Python.list()
    for s in lead_shape: shape.append(s)
    shape.append(N - 1)
    shape.append(N)
    let D = torch.zeros(shape, dtype=dtype, device=device)
    let rows = torch.arange(N - 1, device=device)
    D[":, rows, rows"] = -1.0
    D[":, rows, rows + 1"] = 1.0
    return D

def construct_DTD(N: Int, lead: PythonObject, device: PythonObject, dtype: PythonObject, w_smooth1: PythonObject, w_smooth2: PythonObject, w_smooth3: PythonObject, lam: Float64, dt: Float64) -> PythonObject:
    let torch = get_torch()
    let einops = get_einops()
    let shape = Python.list()
    for s in lead: shape.append(s)
    shape.append(N)
    shape.append(N)
    var DTD = torch.zeros(shape, dtype=dtype, device=device)
    
    if not w_smooth1.is_none():
        let lam_1 = lam / (dt**2)
        let D1 = first_order_D(N, lead, device, dtype)
        # Simplify for brevity in this migration
        DTD += lam_1 * einops.einsum(D1, D1, "... i j, ... i k -> ... j k")
        
    return DTD

def solve_xs_eq_y(s: PythonObject, y: PythonObject, w_data: PythonObject, lam: Float64, ridge: Float64, dt: Float64) -> PythonObject:
    let torch = get_torch()
    let einops = get_einops()
    let device = y.device
    let dtype = y.dtype
    let N = y.shape[-1]
    
    let ATA = einops.einsum(torch.diag_embed(s), torch.diag_embed(s), "... i j, ... i k -> ... j k")
    let rhs = einops.einsum(torch.diag_embed(s), y, "... i j, ... i -> ... j")
    
    # Empty lead shape for simplicity in this example
    let DTD = construct_DTD(N.to_python(), Python.list(), device, dtype, Python.none(), Python.none(), Python.none(), lam, dt)
    
    let ridge_term = ridge * torch.eye(N, dtype=dtype, device=device)
    let lhs = ATA + DTD + ridge_term
    
    let L = torch.linalg.cholesky(lhs)
    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
