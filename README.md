# 🎨 Style Transfer Web App (Beginner-Friendly, 2‑Hour Demo)

A fun, hands-on Streamlit app that applies the **artistic style** of one image to another using **Neural Style Transfer (NST)**.  
Designed for **beginners** and ideal for a **2-hour guided workshop** using **Python + PyCharm**.

---

## 🧠 What You’ll Learn
- The intuition behind **content** vs **style** in images
- How to reuse a **pre-trained VGG19** network for feature extraction
- How **Gram matrices** capture texture/style
- Pixel-level optimization with **PyTorch**
- Building a simple, interactive **Streamlit** app

---

## 🏁 Quick Start

> Recommended Python: 3.9 – 3.11

```bash
# 1) Create & activate a virtual env (example for macOS/Linux)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps (CPU-friendly versions pinned)
pip install -r requirements.txt

# 3) Run the Streamlit app
streamlit run streamlit_app.py
```

Then open the local URL printed by Streamlit. Upload a **content** image and a **style** image, or use the samples.

---

## 🗂 Project Structure
```
style_transfer_web_app/
├── streamlit_app.py      # UI: upload images, sliders, progress, preview, download
├── nst.py                # Classic neural style transfer (beginner-friendly code)
├── utils.py              # Image/device helpers
├── nst_cli.py            # Optional CLI usage
├── requirements.txt
└── samples/
    ├── content.jpg       # Generated sample
    └── style.jpg         # Generated sample
```

---

## 🧪 How It Works (Plain-English)

1. **Load Pre-trained VGG19**  
   We don’t train a CNN from scratch. We take a VGG19 trained on ImageNet and **use it as a fixed feature extractor**.

2. **Extract Features**  
   - **Content features** from a deep layer (e.g., `conv4_2`) represent structure/layout.
   - **Style features** from multiple layers (`conv1_1`…`conv5_1`) capture textures & colors.

3. **Gram Matrix for Style**  
   For each style layer, we compute a **Gram matrix** (feature correlations). Matching these encourages similar textures.

4. **Losses**  
   - **Content Loss:** Keep output close to content features.  
   - **Style Loss:** Match Gram matrices to style image.  
   - **Total Variation (TV) Loss:** Make the output smooth/pleasant.

5. **Optimize Pixels**  
   Start from the content image and **optimize the output pixels** using Adam for N steps.

6. **Preview Progress**  
   The Streamlit app shows intermediate previews and a progress bar.

---

## 🧰 2‑Hour Teaching Plan (Beginner Friendly)

> Time estimates assume CPU. Encourage smaller image sizes (e.g., 256–384) and ~150 steps.

### 0–10 min: Setup
- Create virtualenv, install requirements
- Launch `streamlit run streamlit_app.py`
- Tour of the UI

### 10–25 min: The Idea
- Content vs Style: show sample images
- VGG19 as a feature extractor (no training!)
- Style via Gram matrices (intuitive explanation)

### 25–35 min: Code Walkthrough (Top Level)
- `streamlit_app.py`: file uploaders, sliders, progress callback
- Parameters: steps, style strength, content strength, TV (smoothness)

### 35–55 min: Code Walkthrough (Core)
- `nst.py`: 
  - Which VGG layers are used and why
  - Content/Style losses, TV loss
  - Optimization loop
- `utils.py`: preprocessing & device selection

### 55–75 min: Run Experiments
- Try small sizes (256/320) & 100–150 steps
- Change style strength & TV weight and observe effects
- Discuss CPU vs GPU/MPS (Apple Silicon) speeds

### 75–90 min: Mini‑Challenges
- Try different content/style images
- Increase steps to improve output
- Tweak which style layers are used

### 90–110 min: Additions & UX
- Explain the progress callback and live preview
- Add a “Download” button (already included)
- (Optional) Save intermediate frames to make a GIF later

### 110–120 min: Wrap‑up & Q&A
- Common pitfalls (too large images on CPU)
- Where this is used in industry (creative tools, filters)
- Extensions (fast style transfer models, video stylization)

---

## ⚙️ Parameters to Try

- **Max image size**: 256–512 (CPU: prefer ≤384)  
- **Steps**: 100–300 (more steps = better but slower)  
- **Style weight**: `1e5` to `1e7`  
- **TV weight**: `0` to `1e-3` (more = smoother)

---

## 🖥️ Running from CLI (Optional)

```bash
python nst_cli.py --content samples/content.jpg --style samples/style.jpg --out stylized.jpg --size 384 --steps 150
```

---

## 🧩 Troubleshooting

- **Slow on CPU?**  
  Reduce image size and steps. Close other apps.
- **`use_container_width` error in Streamlit**  
  Some older versions don’t support that flag. This app uses `use_column_width=True` for wider compatibility.
- **“CUDA not available”?**  
  The app auto-detects device. It runs on CPU, CUDA, or Apple MPS.

---

## 🧠 Where Is This Used?
- Creative photo apps & filters
- Design ideation & concept art
- Education/teaching neural representations
- Lightweight artistic effects in mobile/desktop tools

---

## 📚 Learn More (optional reading)
- **Gatys et al. (2015/2016)**: *A Neural Algorithm of Artistic Style*
- **Johnson et al. (2016)**: *Perceptual Losses for Real-Time Style Transfer* (fast models idea)

---

## ✅ License
MIT (for this demo code). Pre-trained VGG19 weights are from torchvision (under their respective license).
