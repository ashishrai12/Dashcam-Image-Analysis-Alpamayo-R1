import math
from alpamayo_r1.geometry.rotation import angle_wrap, Vec2, transform_coords_2d
from alpamayo_r1.config import ModelConfig

fn test_angle_wrap():
    print("Testing angle_wrap...")
    let pi = math.pi
    let test_cases = VariadicList[Float64](
        0.0, 
        pi, 
        -pi, 
        2 * pi, 
        -2 * pi, 
        3 * pi, 
        -3 * pi
    )
    let expected = VariadicList[Float64](
        0.0, 
        -pi, # wrap(pi) -> -pi
        -pi, 
        0.0, 
        0.0, 
        -pi, 
        -pi
    )
    
    for i in range(len(test_cases)):
        let result = angle_wrap(test_cases[i])
        # Use epsilon for float comparison
        if math.abs(result - expected[i]) > 1e-6:
            print("❌ Angle Wrap Fail: input=" + str(test_cases[i]) + " got=" + str(result) + " expected=" + str(expected[i]))
        else:
            print("✅ Angle Wrap Pass: " + str(test_cases[i]))

fn test_vec2_rotation():
    print("\nTesting Vec2 rotation...")
    let v = Vec2(1.0, 0.0)
    let turned = v.rotate(math.pi / 2) # 90 degrees
    
    if math.abs(turned.x - 0.0) < 1e-6 and math.abs(turned.y - 1.0) < 1e-6:
        print("✅ Vec2 Rotation Pass: (1,0) -> (0,1)")
    else:
        print("❌ Vec2 Rotation Fail: got (" + str(turned.x) + ", " + str(turned.y) + ")")

fn test_config_initialization():
    print("\nTesting ModelConfig...")
    let cfg = ModelConfig(num_future_steps=100)
    if cfg.num_future_steps == 100:
        print("✅ Config Init Pass")
    else:
        print("❌ Config Init Fail")

from alpamayo_r1.common import get_torch
from alpamayo_r1.action_space import create_unicycle_action_space

fn test_torch_interop():
    print("\nTesting Torch Interop...")
    try:
        let torch = get_torch()
        let tensor = torch.randn(3, 3)
        print("✅ Torch Tensor created: " + str(tensor.shape))
    except e:
        print("❌ Torch Interop Fail: " + str(e))

fn test_action_space():
    print("\nTesting ActionSpace Factory...")
    try:
        let space = create_unicycle_action_space(64)
        print("✅ ActionSpace created successfully")
    except e:
        print("❌ ActionSpace Fail: " + str(e))

fn main():
    print("--- Alpamayo-R1 Mojo Unit Tests ---")
    test_angle_wrap()
    test_vec2_rotation()
    test_config_initialization()
    test_torch_interop()
    test_action_space()
    print("\n--- All Tests Passed ---")
