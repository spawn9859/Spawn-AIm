![484c9c93-68e8-475a-871e-2069ab9c6223~2](https://github.com/spawn9859/Spawn-AIm/assets/41244175/cb5fa9ec-9a88-4457-a9b3-c0675f6f052f)

# Spawn-Aim

## Has only been tested with Fortnite!

Spawn-Aim is an advanced aimbot tool designed to enhance your gaming experience by providing features such as auto-aim, trigger bot, recoil control, and more. This tool leverages YOLOv5 and YOLOv8 models for object detection and aims to provide a seamless and customizable experience.

## Features

- Auto-aim
- Trigger bot
- Recoil control
- Aim shake
- Overlay and preview
- Customizable settings
- Support for multiple YOLO versions and models
- Xbox controller integration

## Requirements

### **Make sure your machine meets the following requirements to run Spawn AIm smoothly:**

- Ensure you've got Python installed – grab version 3.11 [HERE](https://www.python.org/downloads/release/python-3116/).

- Operating System: Windows 10/11

- Processor: At least dual-core

- Memory: Minimum of 1GB RAM (2GB RAM recommended for optimal performance)

- Storage: At least 1GB of free disk space

- Internet Connection: Stable connection required for downloading and updates

- Nvidia CUDA Toolkit 11.8 [DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

## Installation

### Step 1: Clone the Repository
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/spawn9859/Spawn-Aim.git
cd Spawn-Aim
```
### To install PyTorch, select the appropriate command based on your GPU.

Nvidia
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
AMD or CPU
```
pip install torch torchvision torchaudio
```

### Step 2: Install Dependencies
Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up YOLOv5
Ensure the YOLOv5 repository is cloned and set up:
```bash
    git clone https://github.com/ultralytics/yolov5
    pip install -r yolov5/requirements.txt
```

## Usage

### Step 1: Run the Main Script
Execute the main script to start the application:
```bash
python main.py
```

### Step 2: Configure Settings
Use the GUI to toggle features such as auto-aim, trigger bot, recoil control, and more. Adjust sensitivity, headshot offset, trigger bot distance, confidence, and other parameters using the sliders. Select the YOLO version, model, inference mode, and device from the dropdown menus.

### Step 3: Keybindings
Configure activation and quit keys using the keybindings window. Use the Xbox controller's left trigger to toggle auto-aim if enabled.

## Configuration

The configuration files are located in the `configuration` directory:

- `config.json`: Contains the default settings for the tool.
- `key_mapping.json`: Maps key names to their corresponding key codes.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes with a descriptive commit message:
    ```bash
    git commit -m "Add new feature"
    ```
4. Push your changes to your fork:
    ```bash
    git push origin feature-name
    ```
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [Pygame](https://www.pygame.org/)

## Contact

For any inquiries or support, please contact [spawn9859](https://github.com/spawn9859).
