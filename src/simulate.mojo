from python import Python
import math
from alpamayo_r1.config import ModelConfig
from alpamayo_r1.geometry.rotation import Vec2, angle_wrap

def simulate_scenario(plt: PythonObject, name: String, steering_rate: Float64, color: String):
    let cfg = ModelConfig(num_future_steps=64, time_step=0.1)
    let traj_x = Python.list()
    let traj_y = Python.list()
    
    for i in range(cfg.num_future_steps):
        let t = i * cfg.time_step
        var pos = Vec2(t * 12.0, 0.0) # 12 m/s
        let angle = steering_rate * t
        pos = pos.rotate(angle)
        traj_x.append(pos.x)
        traj_y.append(pos.y)
        
    plt.plot(traj_x, traj_y, color=color, linewidth=2, label=name)

def main():
    try:
        plt = Python.import_module("matplotlib.pyplot")
        os = Python.import_module("os")
        
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 8))
        
        # Simulate different maneuvers
        simulate_scenario(plt, "Maintain Lane (Straight)", 0.0, "#00ffcc")
        simulate_scenario(plt, "Soft Left Turn", 0.05, "#3399ff")
        simulate_scenario(plt, "Sharp Right Turn", -0.12, "#ff3366")
        
        plt.xlabel("Longitudinal Distance (m)")
        plt.ylabel("Lateral Offset (m)")
        plt.title("Alpamayo-R1 Multi-Scenario Trajectory Simulation")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.axis("equal")
        
        let output_dir = "data/results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        let save_path = "data/results/scenario_simulation.png"
        plt.savefig(save_path, dpi=150)
        print("Simulation complete. Visual saved to: " + save_path)
        
    except e:
        print("Visualization Error: " + str(e))

if __name__ == "__main__":
    main()
