import math

fn angle_wrap(radians: Float64) -> Float64:
    """Wraps angles to lie within [-pi, pi)."""
    let pi = math.pi
    return (radians + pi) % (2 * pi) - pi

fn rotation_matrix_2d(angle: Float64) -> (Float64, Float64, Float64, Float64):
    """Returns a 2D rotation matrix as a tuple (cos, -sin, sin, cos)."""
    let c = math.cos(angle)
    let s = math.sin(angle)
    return (c, -s, s, c)

fn transform_coords_2d(x: Float64, y: Float64, angle: Float64, offset_x: Float64, offset_y: Float64) -> (Float64, Float64):
    """Applies rotation and then translation."""
    let c = math.cos(angle)
    let s = math.sin(angle)
    
    # Rotation
    let nx = x * c - y * s
    let ny = x * s + y * c
    
    # Translation
    return (nx + offset_x, ny + offset_y)

fn so3_to_yaw(r00: Float64, r10: Float64) -> Float64:
    """Computes yaw from rotation matrix elements [0,0] and [1,0]."""
    return math.atan2(r10, r00)

struct Vec2:
    var x: Float64
    var y: Float64
    
    fn __init__(inout self, x: Float64, y: Float64):
        self.x = x
        self.y = y
    
    fn rotate(self, angle: Float64) -> Vec2:
        let c = math.cos(angle)
        let s = math.sin(angle)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

struct TrajectoryPoint:
    var pos: Vec2
    var yaw: Float64
    
    fn __init__(inout self, x: Float64, y: Float64, yaw: Float64):
        self.pos = Vec2(x, y)
        self.yaw = yaw
