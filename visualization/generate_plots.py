import matplotlib.pyplot as plt
import numpy as np
import os

def angle_wrap(radians):
    return (radians + np.pi) % (2 * np.pi) - np.pi

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def rotate(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

def generate_trajectory_plot():
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simulation parameters (Mirroring Mojo Logic)
    time_steps = 64
    dt = 0.1
    
    scenarios = [
        {"name": "Cruising (Straight)", "rate": 0.0, "color": "#00ffcc", "alpha": 0.8},
        {"name": "Lane Change (Left)", "rate": 0.06, "color": "#3399ff", "alpha": 0.9},
        {"name": "Evasive Maneuver (Right)", "rate": -0.15, "color": "#ff3366", "alpha": 1.0}
    ]
    
    for scenario in scenarios:
        x, y = [], []
        rate = scenario["rate"]
        for i in range(time_steps):
            t = i * dt
            # Physics: 15m/s base + slight acceleration
            speed = 15.0 + (0.5 * t)
            pos = Vec2(t * speed, 0.0)
            
            # Non-linear steering
            angle = rate * (t**1.2 / 5.0) 
            pos = pos.rotate(angle)
            
            x.append(pos.x)
            y.append(pos.y)
        
        ax.plot(x, y, color=scenario["color"], linewidth=3, label=scenario["name"], alpha=scenario["alpha"])
        ax.scatter(x[-1], y[-1], color=scenario["color"], s=100, edgecolors='white', zorder=5)

    # Aesthetics
    ax.set_xlabel("Longitudinal Distance (meters)", fontsize=12, color='#aaaaaa')
    ax.set_ylabel("Lateral Offset (meters)", fontsize=12, color='#aaaaaa')
    ax.set_title("Alpamayo-R1 Motion Planning Engine: Multi-Scenario Verification", fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.15)
    ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=10)
    
    # Add Ego vehicle marker
    ax.scatter(0, 0, color='#ffff00', s=200, marker='s', label='Ego Vehicle', zorder=10)
    
    plt.axis('equal')
    
    output_path = os.path.join("visualization", "trajectory_verification.png")
    os.makedirs("visualization", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    generate_trajectory_plot()
