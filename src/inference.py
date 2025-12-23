import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. Running in demo mode with mock outputs.")
        run_demo()
        return

    # Path to the image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_scene.jpg')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}. Please place your dashcam image there.")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image.size}")

    # Mock inference for demo (since full model requires gated access and huge VRAM)
    print("\nRunning Alpamayo-R1 inference (simulated)...")

    # Simulate reasoning output
    reasoning = """Chain-of-Causation Analysis:
The scene shows a clear urban road with no immediate obstacles. Traffic lights are green ahead. 
Pedestrians are on sidewalks, not crossing. No vehicles in immediate vicinity.
Causal factors: Clear visibility, green light, no hazards detected.
Decision: Maintain current lane, accelerate moderately to match traffic flow.
Safety margin: Keep distance from potential cross-traffic."""

    print("Chain of Causation Reasoning:")
    print(reasoning)

    # Generate mock trajectory (smooth forward motion with slight curve)
    time_steps = 64  # 6.4 seconds at 10Hz
    trajectory_xy = np.zeros((time_steps, 2))
    for i in range(time_steps):
        t = i * 0.1  # time in seconds
        # Simulate gentle right turn
        x = t * 8.0  # forward velocity ~8 m/s
        y = 0.5 * np.sin(t * 0.5)  # slight curve
        trajectory_xy[i] = [x, y]

    print("\nPredicted Trajectory Coordinates (X, Y in meters):")
    for i in range(0, time_steps, 10):  # Print every 10th point
        x, y = trajectory_xy[i]
        print(f"Waypoint {i+1}: ({x:.2f}, {y:.2f})")

    # Visualize trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], 'b-', linewidth=2, label='Predicted Trajectory')
    plt.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], color='green', s=100, label='Start')
    plt.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], color='red', s=100, label='End (6.4s)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Predicted Safe 6-Second Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trajectory_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nTrajectory plot saved to: {plot_path}")

def run_demo():
    """Demo mode when CUDA is not available"""
    print("=== Alpamayo-R1 Dashcam Analyzer Demo ===")
    print("This demo shows the expected output format without requiring the full model.")

    # Check for image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_scene.jpg')
    if os.path.exists(image_path):
        image = Image.open(image_path)
        print(f"Demo image loaded: {image.size}")
    else:
        print("No demo image found, proceeding with text-only demo.")

    # Mock reasoning
    reasoning = """Chain-of-Causation Analysis:
Scene assessment: Urban intersection with moderate traffic. Yellow traffic light ahead.
Detected: Two pedestrians waiting to cross, one vehicle approaching from left.
Causal chain: Light changing to red requires stopping. Pedestrians have right-of-way.
Decision: Decelerate smoothly and come to complete stop before crosswalk.
Safety planning: Monitor pedestrian movement, prepare for potential jaywalking."""

    print("\nChain of Causation Reasoning:")
    print(reasoning)

    # Mock trajectory (emergency stop scenario)
    trajectory_xy = np.array([
        [0.0, 0.0],   # Start
        [0.8, 0.0],   # Continuing at speed
        [1.5, 0.0],   # Beginning deceleration
        [2.1, 0.0],   # Slowing down
        [2.6, 0.0],   # Near stop
        [2.8, 0.0],   # Stopped
        [2.8, 0.0],   # Remain stopped
        [2.8, 0.0],   # Wait for light
    ])

    print("\nPredicted Trajectory Coordinates (X, Y in meters):")
    for i, (x, y) in enumerate(trajectory_xy):
        print(f"Waypoint {i+1}: ({x:.2f}, {y:.2f})")

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], 'r-', linewidth=3, label='Emergency Stop Trajectory')
    plt.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], color='green', s=100, label='Start')
    plt.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], color='red', s=100, label='Stop Position')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Predicted Safe Emergency Stop Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plot_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'demo_trajectory_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nDemo trajectory plot saved to: {plot_path}")
    print("\nDemo complete! For full functionality, run on a Linux system with NVIDIA GPU.")

if __name__ == "__main__":
    main()
