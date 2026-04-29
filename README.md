# PCB Defect Detection — YOLOv8s

Real-time detection of 6 PCB manufacturing defects using YOLOv8s,
trained on the DeepPCB benchmark dataset achieving **95.3% mAP@50**.

---

## Results

| Metric | Score |
|--------|-------|
| mAP@50 | 95.30% |
| mAP@50-95 | 56.81% |
| Precision | 94.16% |
| Recall | 90.58% |
| Training Time | 12 minutes (Tesla T4 x2) |
| Early Stop | Epoch 46 / 100 |

---

## Per-Class Performance

| Class | mAP@50 | Miss Rate |
|-------|--------|-----------|
| open_circuit | 93.13% | 1.2% |
| short_circuit | 88.74% | 9.0% |
| mouse_bite | 96.99% | 3.6% |
| spur | 96.55% | 5.4% |
| copper_spill | 98.30% | 5.2% |
| pin_hole | 98.09% | 2.3% |

---

## Project Structure

```
pcb-defect-detection/
├── app/
│   └── streamlit_app.py    # Web application
├── models/
│   └── best.pt             # Trained YOLOv8s weights (21.5MB)
├── src/
│   └── inference.py        # Core predict() function
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Ahsan-Neural/pcb-defect-detection.git
cd pcb-defect-detection
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Dataset

**DeepPCB Benchmark** — https://github.com/tangsanli5201/DeepPCB

- 1,500 images — 850 train / 150 val / 500 test
- 10,013 defect annotations across 6 classes
- All images 640 x 640 pixels

---

## Training Details

- Base model: YOLOv8s pretrained on COCO
- Optimizer: AdamW (lr=0.001)
- Augmentation: mosaic, mixup, rotation, scale, horizontal flip
- Early stopping patience: 20 epochs
- Best checkpoint: epoch 26

---

## Links

- Kaggle Notebook (full training + evaluation): [View on Kaggle](https://www.kaggle.com/ahsanneural)
- Kaggle Profile: [ahsanneural](https://www.kaggle.com/ahsanneural)
- Live Demo: [Streamlit App](#) ← will update after deployment

