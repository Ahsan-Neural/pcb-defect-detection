# PCB Defect Detection — YOLOv8s

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pcb-defect-detection.streamlit.app)

Real-time detection of 6 PCB manufacturing defects using YOLOv8s, trained on the DeepPCB benchmark dataset achieving **95.3% mAP@50**.

🚀 **[Live Demo](https://pcb-defect-detection.streamlit.app)** | 📊 **[Model Card](#model-performance)** | 📖 **[Kaggle Notebook](https://www.kaggle.com/ahsanneural)**

---

## 🎯 Features

- ✅ **Real-time Detection** - Detect 6 types of PCB defects instantly
- ✅ **High Accuracy** - 95.3% mAP@50 on DeepPCB benchmark
- ✅ **Web Interface** - Easy-to-use Streamlit web application
- ✅ **Adjustable Thresholds** - Customize confidence and IoU thresholds
- ✅ **Detailed Reports** - Per-class statistics and visualizations
- ✅ **Download Results** - Export annotated images and CSV reports
- ✅ **Lightweight Model** - Only 22.5MB (YOLOv8s)

### Detectable Defect Classes

| Class | Description |
|-------|-------------|
| **open_circuit** | Broken connections in PCB traces |
| **short_circuit** | Unintended electrical connections |
| **mouse_bite** | Irregular notches/bites on PCB edges |
| **spur** | Thin unwanted connections |
| **copper_spill** | Excessive copper deposits |
| **pin_hole** | Small holes in copper layers |

---

## 📊 Model Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| mAP@50 | **95.30%** |
| mAP@50-95 | 56.81% |
| Precision | 94.16% |
| Recall | 90.58% |
| Training Time | 12 minutes (Tesla T4 x2) |
| Best Checkpoint | Epoch 26 / 100 |

### Per-Class Performance

| Class | mAP@50 | Miss Rate |
|-------|--------|-----------|
| open_circuit | 93.13% | 1.2% |
| short_circuit | 88.74% | 9.0% |
| mouse_bite | 96.99% | 3.6% |
| spur | 96.55% | 5.4% |
| copper_spill | 98.30% | 5.2% |
| pin_hole | 98.09% | 2.3% |

**Note:** Miss Rate indicates the percentage of defects not detected at confidence threshold 0.25.

---

## 📁 Project Structure

```
pcb-defect-detection/
├── app/
│   └── streamlit_app.py         # Web application interface
├── model/
│   └── best.pt                  # Trained YOLOv8s weights (22.5MB)
├── src/
│   └── inference.py             # Core inference logic
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version specification
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python:** 3.8 or higher
- **pip:** Latest version
- **System RAM:** Minimum 2GB (4GB+ recommended)
- **GPU (Optional):** CUDA-capable GPU for faster inference

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Ahsan-Neural/pcb-defect-detection.git
cd pcb-defect-detection
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app:**
```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 💻 Usage

### Web Interface (Recommended)

1. Upload a PCB image (JPG or PNG)
2. Adjust confidence and IoU thresholds in the sidebar
3. View real-time detections with annotations
4. Download results as:
   - Annotated PNG image
   - CSV report with defect details

### Python API

```python
from src.inference import predict, CLASS_LABELS

# Run inference
results = predict(
    image_path="path/to/pcb_image.jpg",
    model_path="model/best.pt",
    conf=0.25,          # Confidence threshold
    iou=0.45,           # IoU threshold for NMS
    device="cpu"        # "cpu" or "0" for GPU
)

# Access results
print(f"Total defects: {results['total']}")
print(f"Detections: {results['detections']}")
print(f"Per-class counts: {results['per_class']}")

# results['annotated'] contains the annotated image (PIL Image)
results['annotated'].show()
```

---

## 📚 Dataset

**DeepPCB Benchmark** — [GitHub Repository](https://github.com/tangsanli5201/DeepPCB)

- **Total Images:** 1,500 (640×640 pixels)
  - Training: 850 images
  - Validation: 150 images
  - Test: 500 images
- **Annotations:** 10,013 defect instances across 6 classes
- **Format:** YOLO format with bounding boxes

---

## 🔧 Training Details

- **Base Model:** YOLOv8s (pretrained on COCO)
- **Optimizer:** AdamW (learning rate: 0.001)
- **Batch Size:** 16
- **Image Size:** 640×640
- **Epochs:** 100 (stopped at 46 with early stopping)
- **Early Stopping:** 20 epoch patience
- **Augmentations:** Mosaic, mixup, rotation, scaling, horizontal flip

---

## 🐛 Troubleshooting

### Issue: "Model file not found at model/best.pt"

**Solution:** Ensure the model directory and best.pt file exist:
```bash
ls -la model/best.pt
```

If missing, the model file may need to be added to the repository or downloaded separately.

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:** Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Slow inference on CPU

**Solution:** If you have a CUDA-capable GPU, ensure CUDA and cuDNN are installed, then run:
```bash
streamlit run app/streamlit_app.py --logger.level=debug
```

And modify the device parameter in the app or code to use GPU.

### Issue: Streamlit app crashes on upload

**Solution:** Check file size and format:
- Maximum recommended file size: 10MB
- Supported formats: JPG, PNG
- Ensure image dimensions are at least 100×100 pixels

---

## 🔗 Links

- **Live Demo:** [Streamlit App](https://pcb-defect-detection.streamlit.app)
- **Kaggle Notebook:** [Full Training & Evaluation](https://www.kaggle.com/ahsanneural)
- **Kaggle Profile:** [ahsanneural](https://www.kaggle.com/ahsanneural)
- **DeepPCB Dataset:** [GitHub](https://github.com/tangsanli5201/DeepPCB)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**M. Ahsan**
- GitHub: [@Ahsan-Neural](https://github.com/Ahsan-Neural)
- Email: ahsanatwork24@gmail.com

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## ⭐ Acknowledgments

- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **DeepPCB Dataset** by Tang et al.
- **Streamlit** for the amazing web framework