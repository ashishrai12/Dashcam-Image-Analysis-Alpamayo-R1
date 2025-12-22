import torch
from PIL import Image
import numpy as np
from transformers import BitsAndBytesConfig
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
import matplotlib.pyplot as plt
import os

def main():
    # Check for CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This model requires a GPU with sufficient VRAM (at least 24GB recommended).")

    # Path to the image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_scene.jpg')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}. Please place your dashcam image there.")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    image_tensor = to_tensor(image).unsqueeze(0)  # [1, 3, H, W]

    # Create messages (modify the prompt as per user request)
    messages = helper.create_message(image_tensor)
    # Change the user text to the specified prompt
    messages[1]["content"][-1]["text"] = "Explain the current scene and plan a safe 6-second trajectory."

    # Load model with 4-bit quantization
    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            quantization_config=quant_config,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to 8-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            quantization_config=quant_config,
            device_map="auto"
        )

    # Get processor
    processor = helper.get_processor(model.tokenizer)

    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Prepare model inputs with dummy ego history (since we have a single image)
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": torch.zeros(1, 16, 3, dtype=torch.float32),  # [batch, 16, 3]
        "ego_history_rot": torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1),  # [batch, 16, 3, 3]
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    # Run inference
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        try:
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    # Extract reasoning
    reasoning = extra["cot"][0]
    print("Chain of Causation Reasoning:")
    print(reasoning)

    # Extract trajectory (64 waypoints, 6.4s at 10Hz, xy coordinates)
    trajectory_xy = pred_xyz[0, 0, 0, :, :2].cpu().numpy()  # [64, 2]
    print("\nPredicted Trajectory Coordinates (X, Y in meters):")
    for i, (x, y) in enumerate(trajectory_xy):
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

    # Optionally show plot (comment out if running headless)
    # plt.show()

if __name__ == "__main__":
    main()
