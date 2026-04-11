---
title: "SaliencyMap: Modeling Human Visual Attention for Ad Creative Optimization"
description: "End-to-end saliency prediction pipeline from dataset preparation and VGG-16 encoder–decoder construction, through three-phase transfer learning training, to evaluation with KL-Divergence & AUC-Judd metrics."
author: "Fati Buulolo"
image:
  url: "../../assets/images-014/display_saliency.png"
  alt: "SaliencyMap: Modeling Human Visual Attention"
pubDate: 2026-04-09
tags:
  ["Python", "TensorFlow", "Keras", "VGG-16", "Transfer Learning", "Saliency Prediction", "Computer Vision", "Streamlit"]
---


## Overview

Every advertisement competes for a fraction of a second of human attention. Visual saliency, defined as the property of image regions that instinctively capture the eye, is a fundamental determinant of whether a creative succeeds or is ignored. Understanding where users look before they consciously decide to engage is arguably more valuable than any post-click conversion metric.

This project builds a deep learning saliency prediction model that, given any advertising image, outputs a spatially calibrated heatmap indicating the probability distribution of human eye fixations. The model is an **encoder–decoder network** built on a frozen **VGG-16** backbone trained on **ImageNet**, with a custom learned decoder trained against ground-truth fixation density maps from the **SALICON** dataset [[**CLick Here**]](https://www.salicon.net/).

The final application is deployed as an interactive **Streamlit** web app on **Hugging Face Spaces**, enabling marketing teams and creatives to upload an ad and immediately receive actionable attention maps.

---

## Problem Statement

Standard ad performance metrics (CTR, CPM, ROAS) are outcome signals, they just tell what happened. Two creatives with identical budgets can produce wildly different results because of pure visual design choices: where the headline sits, whether the CTA is in the visual periphery, or whether the product is obscured by a competing background element.

Saliency maps answer a prior question: before a user decides anything, where is their eye? This project operationalises saliency prediction as:

1. A **regression task** — given RGB pixels, predict a 2D probability map of eye fixation density.
2. A **transfer learning problem** — leverage general visual features from VGG-16 and adapt them to fixation prediction via a supervised decoder.

---

## Dataset — SALICON

<div class="my-8 overflow-hidden rounded-2xl border border-zinc-200 shadow-sm">
  <table class="w-full text-left text-sm border-collapse bg-white/50 backdrop-blur-sm">
    <thead class="bg-zinc-50 border-b border-zinc-200">
      <tr>
        <th class="px-6 py-4 font-bold text-zinc-400 uppercase tracking-widest text-[10px]">Property</th>
        <th class="px-6 py-4 font-bold text-zinc-400 uppercase tracking-widest text-[10px]">Value</th>
      </tr>
    </thead>
    <tbody class="divide-y divide-zinc-100">
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Dataset</td>
        <td class="px-6 py-4 text-zinc-600 italic">SALICON (SALIency in CONtext)</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Images</td>
        <td class="px-6 py-4 text-zinc-600">MS-COCO images (training + validation split)</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Ground Truth</td>
        <td class="px-6 py-4 text-zinc-600 text-xs">Eye-tracking fixation density maps (grayscale)</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Preprocessing</td>
        <td class="px-6 py-4 text-zinc-600">
          <span class="bg-zinc-100 px-2 py-1 rounded font-mono text-[11px]">224 × 224</span>, normalised to [0, 1]
        </td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Augmentation</td>
        <td class="px-6 py-4 text-zinc-600">Random rotation (±15°), horizontal flip</td>
      </tr>
    </tbody>
  </table>
</div>

The SALICON dataset provides paired `(image, fixation_map)` samples. Each fixation map is a continuous grayscale heatmap derived from aggregating multiple participants' eye-tracking fixation points, producing a smooth probability distribution over image regions.

---

## Model Architecture

The architecture follows an **encoder–decoder** design, a natural choice for dense prediction tasks (analogous to semantic segmentation).

### Encoder — VGG-16 (Frozen)

```
Input: (224, 224, 3)
  └─ VGG-16 convolutional backbone (weights='imagenet', trainable=False)
     Output feature map: (7, 7, 512)
```

The VGG-16 backbone is loaded with pretrained ImageNet weights and kept **frozen** during all training phases. The top (fully connected) layers are excluded.

### Decoder — Progressive Upsampling

The decoder reconstructs spatial resolution through five consecutive `Conv2D → UpSampling2D` blocks:

```
(7, 7, 512)
  → Conv2D(512, 3×3, relu, same) + UpSampling2D(2×2)  →  (14, 14, 512)
  → Conv2D(256, 3×3, relu, same) + UpSampling2D(2×2)  →  (28, 28, 256)
  → Conv2D(128, 3×3, relu, same) + UpSampling2D(2×2)  →  (56, 56, 128)
  → Conv2D(64, 3×3, relu, same)  + UpSampling2D(2×2)  →  (112, 112, 64)
  → Conv2D(32, 3×3, relu, same)  + UpSampling2D(2×2)  →  (224, 224, 32)
  → Conv2D(1, 1×1, sigmoid, same)                     →  (224, 224, 1)
```

The final `1×1` convolution with `sigmoid` activation collapses the channel dimension to a single-channel saliency probability map in the range `[0, 1]`, spatially aligned with the input image.

**Total architecture:**

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
outputs = layers.Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
```

---

## Training Pipeline

Training was conducted in **three sequential phases** on Google Colab, each resuming from the best checkpoint of the previous phase. This phased approach was adopted to handle Colab session limits while maintaining training continuity.

### Data Generator

A custom paired generator feeds synchronized `(image, fixation_map)` batches:

```python
def combine_generator(gen1, gen2):
    while True:
        yield (next(gen1), next(gen2))
```

Both generators share the same `seed=42` to guarantee that augmentation transforms (rotation, flip) are applied identically to the image and its corresponding map.

<div class="my-8 overflow-hidden rounded-2xl border border-zinc-200 shadow-sm">
  <table class="w-full text-left text-sm border-collapse bg-white/50 backdrop-blur-sm">
    <thead class="bg-zinc-50 border-b border-zinc-200">
      <tr>
        <th class="px-6 py-4 font-bold text-zinc-400 uppercase tracking-widest text-[10px]">Parameter</th>
        <th class="px-6 py-4 font-bold text-zinc-400 uppercase tracking-widest text-[10px]">Value</th>
      </tr>
    </thead>
    <tbody class="divide-y divide-zinc-100">
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Batch size</td>
        <td class="px-6 py-4 text-zinc-600 font-mono">16</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Input size</td>
        <td class="px-6 py-4 text-zinc-600">224 × 224</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Augmentation</td>
        <td class="px-6 py-4 text-zinc-600">Rotation ±15°, horizontal flip</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Rescaling</td>
        <td class="px-6 py-4 text-zinc-600">
          Divide by 255 <span class="text-zinc-400 mx-1">→</span> <span class="bg-zinc-100 px-2 py-0.5 rounded text-[11px] font-mono">[0, 1]</span>
        </td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Map color mode</td>
        <td class="px-6 py-4 text-zinc-600 italic">Grayscale</td>
      </tr>
    </tbody>
  </table>
</div>

---

### Phase 1 — Initial Training (Epochs 1–10)

```python
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='mse',
    metrics=['mae']
)

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='loss',
    mode='min',
    verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    callbacks=[checkpoint_callback]
)
```

**Loss:** Mean Squared Error (MSE) between predicted saliency map and ground-truth fixation density. MSE is appropriate here because both tensors are continuous probability distributions in `[0, 1]`.

**Optimizer:** Adam with a conservative learning rate of `1e-5` to protect the frozen VGG-16 feature representations from gradient interference while training only decoder weights.

**Checkpoint strategy:** `save_best_only=True` with `monitor='loss'` ensures only the epoch with the lowest training loss is persisted, preventing regression from noisy late-epoch updates.

---

### Phase 2 — Resumed Training (Epochs 5–10, `initial_epoch=4`)

The best checkpoint from Phase 1 was loaded and training resumed from epoch 5:

```python
model = tf.keras.models.load_model(checkpoint_path)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    initial_epoch=4,
    callbacks=[checkpoint_callback]
)
```

The `initial_epoch` argument ensures Keras correctly reports and tracks epoch numbers for logging continuity. The same checkpoint is overwritten if a better result is found.

---

### Phase 3 — Final Convergence (Epochs 9–10, `initial_epoch=8`)

```python
model = tf.keras.models.load_model(checkpoint_path)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    initial_epoch=8,
    callbacks=[checkpoint_callback]
)
```

This final phase allowed the model to converge fully from epoch 9 onward, squeezing the remaining gradient descent steps from the training schedule.

---

### Model Serialization

After Phase 3 completed, the model was saved in two formats for portability:

```python
model.save('/content/drive/MyDrive/SALICON/model_final_saliency.keras')  
model.save('/content/drive/MyDrive/SALICON/model_final_saliency.h5')     
```

The `.keras` format is preferred for deployment as it correctly serializes custom layers, optimizer state, and build configurations.

---

## Evaluation

Evaluation was performed on a held-out **validation split** from the SALICON dataset.

### Metrics

#### KL-Divergence

Measures the information-theoretic distance between the predicted distribution and the ground-truth fixation map. Lower is better.

```python
def calculate_kl_divergence(y_true, y_pred):
    y_true = y_true.flatten().astype(np.float32)
    y_pred = y_pred.flatten().astype(np.float32)
    eps = 1e-10
    y_true = y_true / (np.sum(y_true) + eps)
    y_pred = y_pred / (np.sum(y_pred) + eps)
    kl_div = np.sum(y_true * np.log((y_true + eps) / (y_pred + eps)))
    return kl_div
```

#### AUC-Judd

Area Under the ROC Curve computed with a per-image adaptive threshold (mean fixation value). Measures how well the predicted saliency map separates fixated from non-fixated pixels. Higher is better.

```python
def calculate_auc_judd(y_true, y_pred):
    y_true_norm = y_true.flatten().astype(np.float32) / 255.0
    y_pred_flat = y_pred.flatten().astype(np.float32)
    threshold = np.mean(y_true_norm)
    y_true_binary = (y_true_norm > threshold).astype(int)
    if len(np.unique(y_true_binary)) == 1:
        return np.nan
    return roc_auc_score(y_true_binary, y_pred_flat)
```

### Inference Loop

```python
for i in tqdm(range(len(images_files))):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_width, input_height))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    gt_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    gt_map_resized = cv2.resize(gt_map, (input_width, input_height))

    pred_map = model.predict(img_input, verbose=0)
    pred_map = np.squeeze(pred_map)
    pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-10)

    kl = calculate_kl_divergence(gt_map_resized, pred_map)
    auc = calculate_auc_judd(gt_map_resized, pred_map)
    kl_scores.append(kl)
    auc_scores.append(auc)
```

---

## Inference & Heatmap Generation

At inference time, the following pipeline converts a raw PIL image into a coloured saliency heatmap:

```
Input PIL image  (any resolution)
  → Resize to (224, 224)
  → Normalise to [0, 1]
  → Add batch dimension → (1, 224, 224, 3)
  → model.predict()  → (1, 224, 224, 1)
  → squeeze + normalise to [0, 255] uint8
  → cv2.resize() back to original resolution
  → cv2.applyColorMap(COLORMAP_JET)
  → Convert BGR → RGB
Output: heatmap RGB array (H, W, 3)
```

**JET colormap interpretation:**
- 🔴 Red / Warm → high predicted attention probability
- 🟡 Yellow / Green → moderate attention
- 🔵 Blue / Cool → low attention

---

## Business Application

The saliency heatmap output can be used by marketing and creative teams to:

1. **CTA Placement Audit** — Verify the call-to-action button falls inside a red/warm region of the heatmap. If it lands in a blue zone, the design is actively working against conversion intent.
2. **Headline Hierarchy Check** — Confirm the primary headline draws more eye focus than secondary copy.
3. **Logo Visibility** — Assess brand mark salience without relying on post-campaign recall surveys.
4. **A/B Creative Pre-screening** — Compare heatmaps of two creative variants before running spend, selecting the one with more strategically distributed attention.

---

## Tools and Libraries

<div class="my-8 overflow-hidden rounded-2xl border border-zinc-200 shadow-sm">
  <table class="w-full text-left text-sm border-collapse bg-white/50 backdrop-blur-sm">
    <thead class="bg-zinc-50 border-b border-zinc-200">
      <tr>
        <th class="px-6 py-4 font-bold text-zinc-400 uppercase tracking-widest text-[10px]">Category</th>
        <th class="px-6 py-4 font-bold text-zinc-400 uppercase tracking-widest text-[10px]">Library / Tool</th>
      </tr>
    </thead>
    <tbody class="divide-y divide-zinc-100">
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Deep Learning</td>
        <td class="px-6 py-4 text-zinc-600">TensorFlow / Keras</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Pretrained Model</td>
        <td class="px-6 py-4 text-zinc-600 font-mono text-xs">VGG-16 (ImageNet weights)</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Computer Vision</td>
        <td class="px-6 py-4 text-zinc-600">OpenCV, Pillow</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Numerical Computing</td>
        <td class="px-6 py-4 text-zinc-600 font-mono text-xs">NumPy</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Evaluation</td>
        <td class="px-6 py-4 text-zinc-600">scikit-learn (ROC-AUC)</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Visualization</td>
        <td class="px-6 py-4 text-zinc-600">Matplotlib</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Web Application</td>
        <td class="px-6 py-4 text-[#c9b99a] font-bold">Streamlit</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Deployment</td>
        <td class="px-6 py-4 text-zinc-600">Hugging Face Spaces</td>
      </tr>
      <tr class="hover:bg-white/80 transition-colors">
        <td class="px-6 py-4 font-semibold text-zinc-700">Training Environment</td>
        <td class="px-6 py-4 text-zinc-500 text-xs italic">Google Colab + Google Drive</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Source Code

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px;">
  <a href="https://github.com/FatiBuuloloo/SaliencyMap-Modeling_Human_Visual_Attention_for_Ad_Creative_Optimization-mini_project-014">View on GitHub</a>
</strong>

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">
  <a href="https://viewww-saliencymap-ads-optimization.hf.space/">Live Demo</a>
</strong>

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">
  <a href="https://huggingface.co/spaces/Viewww/SaliencyMap-Ads_Optimization/tree/main/">HuggingFace Space</a>
</strong>
