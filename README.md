# Spawn-Aim

![spawnaim](https://github.com/spawn9859/Spawn-AIm/assets/41244175/53f5551b-f423-4e4e-b12c-167160ae0fa8)

## Overview

Spawn-Aim is an advanced aim assistance tool designed for gaming enthusiasts. It uses computer vision and machine learning techniques to detect targets and provide aim assistance in compatible games. This project is for educational purposes only and should be used responsibly and in accordance with game rules and policies.


**Note:** Use of this tool may violate terms of service for many online games. Use at your own risk.

## Features

- Real-time object detection using YOLO (You Only Look Once) models
- Customizable aim assistance settings
- Trigger bot functionality
- Recoil control
- Field of View (FOV) limitation
- Preview window for debugging
- Overlay for visual feedback
- Support for multiple YOLO versions (v5 and v8)
- Xbox controller integration

## Requirements

- Windows 10/11
- Python 3.11.6 or higher
- NVIDIA GPU with CUDA support (for optimal performance)
- Xbox controller (optional, for certain features)

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/your-username/Spawn-Aim.git
   cd Spawn-Aim
   ```

2. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support:**
   Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions for installing PyTorch with CUDA support for your system.

5. **Install CUDA Toolkit:**
   Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) compatible with your GPU and PyTorch version.

## Configuration

1. Navigate to the `configuration` folder.
2. Open `config.json` and adjust settings as needed. Key settings include:
   - `yolo_version`: Set to "v5" or "v8" depending on your preference
   - `yolo_model`: Specify the name of your trained YOLO model
   - `sensitivity`: Adjust aim sensitivity
   - `fov_size`: Set the Field of View size for target detection

## Usage

1. **Activate the virtual environment:**
   ```
   venv\Scripts\activate
   ```

2. **Run the main script:**
   ```
   python src/main.py
   ```

3. **Using the Interface:**
   - Use the checkboxes to enable/disable features like auto-aim, trigger bot, etc.
   - Adjust sliders for fine-tuning sensitivity, FOV size, and other parameters
   - The preview window shows detected targets and aim assistance visualization
   - Use the "Show FOV" checkbox to display the current FOV in the preview

4. **In-Game Usage:**
   - Press the configured activation key (default: Alt) to activate aim assistance
   - Use the left trigger on the Xbox controller to toggle auto-aim (if enabled)
   - Press the quit key (default: Q) to exit the program

## Customization

- To use your own YOLO model, place the `.pt` file in the `models` folder and update the `yolo_model` setting in `config.json`
- Adjust keybindings in the `key_mapping.json` file in the `configuration` folder

## Troubleshooting

- If you encounter CUDA-related errors, ensure that your NVIDIA drivers and CUDA Toolkit are up to date
- For "DLL not found" errors, try installing the latest [Visual C++ Redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

## Disclaimer

This project is for educational purposes only. Using aim assistance software in online games may result in bans or other penalties. The developers are not responsible for any consequences resulting from the use of this software.

## Contributing

Contributions to Spawn-Aim are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
