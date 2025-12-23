# Alpamayo-R1 Dashcam Analyzer

This project demonstrates the use of NVIDIA's Alpamayo-R1 (AR1) model, a vision-language-action (VLA) model that integrates Chain-of-Causation (CoC) reasoning with trajectory planning for autonomous driving applications. The model excels at handling rare, long-tail driving scenarios by providing interpretable reasoning traces alongside precise vehicle control predictions.

<img width="716" height="543" alt="{A9914665-DABC-42AA-B51D-5F1C49C4002C}" src="https://github.com/user-attachments/assets/f266470a-a486-48ac-8347-0523ea54faa4" />

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

Output Summary
The script executed in demo mode because CUDA (GPU acceleration) is not available on your Windows system. This is expected - the full Alpamayo-R1 model requires a Linux system with NVIDIA GPU for actual inference.

What's Happening
Environment Check: The script first checks if CUDA is available. Since it's not (common on Windows), it switches to demo mode with mock outputs.

Image Loading: It successfully loaded the test image from test_scene.jpg (1920x1080 resolution).

Chain-of-Causation Reasoning: The demo simulates AI reasoning about a driving scenario:

Urban intersection with moderate traffic
Yellow traffic light ahead
Pedestrians waiting to cross and a vehicle approaching
Decision: Emergency stop before the crosswalk
Trajectory Prediction: It generates mock trajectory waypoints showing the vehicle decelerating and stopping:

Starts at (0.00, 0.00)
Moves forward initially, then slows down
Comes to a complete stop at ~2.8 meters forward
Remains stopped for the remaining time
Visualization: Creates a plot of the predicted trajectory and saves it as demo_trajectory_plot.png

The script demonstrates how the Alpamayo-R1 model would analyze dashcam footage to predict safe driving trajectories using causal reasoning, but without the actual AI model running. For full functionality, you'd need to run this on a Linux system with proper GPU setup.

The chain-of-causation reasoning text in the output comes from hardcoded mock data in the run_demo() function of your script. Output.txt is not a generated by an actual AI model - Output.txt is a simulated example to demonstrate what the output would look like.
