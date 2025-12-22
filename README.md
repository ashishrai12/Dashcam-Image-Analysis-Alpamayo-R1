# Alpamayo-R1 Dashcam Analyzer

This project demonstrates the use of NVIDIA's Alpamayo-R1 (AR1) model, a vision-language-action (VLA) model that integrates Chain-of-Causation (CoC) reasoning with trajectory planning for autonomous driving applications. The model excels at handling rare, long-tail driving scenarios by providing interpretable reasoning traces alongside precise vehicle control predictions.

## What is Alpamayo-R1?

Alpamayo-R1 is developed by NVIDIA and bridges reasoning and action prediction in autonomous driving. It combines:
- **Chain-of-Causation Reasoning**: Structured, causal explanations of driving decisions
- **Trajectory Planning**: Future trajectory predictions with position and rotation in ego-vehicle coordinates
- **Multi-Modal Input**: Processes camera images, text prompts, and egomotion history
- **Diffusion-Based Action Decoder**: Generates smooth, physically plausible trajectories

The model uses reasoning to solve complex driving scenarios that traditional models struggle with, such as:
- Complex intersections with multiple actors
- Pedestrian interactions
- Adverse weather conditions
- Cut-in maneuvers and other rare events

## Project Structure

```
project/
├── src/
│   └── inference.py          # Main inference script
├── data/
│   ├── test_scene.jpg        # Place your dashcam image here
│   └── trajectory_plot.png   # Generated trajectory visualization
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU with at least 24GB VRAM (RTX 3090 or equivalent)
- Linux environment (recommended, though Windows may work with modifications)

### Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate with Hugging Face (required for gated model access):
   ```bash
   huggingface-cli login
   ```
   Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

4. Request access to the model datasets:
   - [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
   - [Alpamayo-R1-10B Model](https://huggingface.co/nvidia/Alpamayo-R1-10B)

## Usage

1. Place your dashcam image in the `data/` directory as `test_scene.jpg`

2. Run the inference script:
   ```bash
   cd src
   python inference.py
   ```

The script will:
- Load the Alpamayo-R1 model with 4-bit quantization (falls back to 8-bit if needed)
- Process your image with the prompt: "Explain the current scene and plan a safe 6-second trajectory"
- Generate Chain-of-Causation reasoning
- Predict a 6.4-second future trajectory (64 waypoints at 10Hz)
- Display the reasoning text and trajectory coordinates
- Save a visualization plot of the predicted trajectory

## Output

The script provides:
- **Reasoning Trace**: Detailed explanation of the scene analysis and driving decisions
- **Trajectory Coordinates**: X,Y positions in meters relative to the ego vehicle
- **Visualization**: A plot showing the predicted path over 6 seconds

## Technical Details

- **Model Size**: 10.5B parameters (8.2B backbone + 2.3B action expert)
- **Quantization**: 4-bit NF4 quantization for VRAM efficiency
- **Input**: Single RGB image (adapted from multi-camera training)
- **Output**: Text reasoning + 64-point trajectory with position and rotation
- **Inference Time**: ~10-30 seconds depending on hardware

## Limitations

- Designed for multi-camera, multi-timestep inputs; single image usage is adapted
- Requires significant GPU memory
- Non-commercial license for research use only
- Outputs are stochastic; results may vary between runs

## License

This project uses NVIDIA's Alpamayo-R1 model under its non-commercial research license. See the model card for details.

## References

- [Alpamayo-R1 Paper](https://arxiv.org/abs/2511.00088)
- [Model Repository](https://github.com/NVlabs/alpamayo)
- [Hugging Face Model](https://huggingface.co/nvidia/Alpamayo-R1-10B)
