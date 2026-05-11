---
marp: true
theme: default
paginate: true
html: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* ── Base ── */
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 1.02rem;
    padding: 44px 64px;
    color: #1e293b;
    background: #ffffff;
  }
  h1 {
    color: #00325d;
    border-bottom: 3px solid #00325d;
    padding-bottom: 10px;
    margin-bottom: 20px;
    font-size: 1.75rem;
    font-weight: 700;
  }
  h2 { color: #455d82; margin-top: 6px; font-size: 1.2rem; font-weight: 500; }
  ul { margin-top: 10px; }
  li { margin-bottom: 6px; line-height: 1.55; }
  strong { color: #00325d; }
  code {
    background: #eff1f5;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.88rem;
  }
  blockquote {
    border-left: 4px solid #00325d;
    padding: 12px 24px;
    background: #eff1f5;
    margin: 18px 0;
    font-style: italic;
    color: #334155;
    border-radius: 0 8px 8px 0;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 0.9rem; }
  th { background: #00325d; color: white; padding: 10px 14px; text-align: left; }
  td { padding: 9px 14px; border-bottom: 1px solid #d1d7df; }
  tr:nth-child(even) td { background: #f5f6f9; }

  /* ── Cover ── */
  section.cover {
    background: linear-gradient(135deg, #00192f 0%, #00325d 40%, #455d82 100%);
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.cover h1 { color: white; border-bottom: 2px solid rgba(255,255,255,0.25); font-size: 2.4rem; }
  section.cover h2 { color: rgba(255,255,255,0.8); font-size: 1.3rem; font-weight: 400; }
  section.cover p  { color: rgba(255,255,255,0.55); }

  /* ── Section breaks ── */
  section.break {
    background: linear-gradient(135deg, #eff1f5 0%, #e8edf5 100%);
    display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
  }
  section.break h1 { font-size: 2.2rem; border-bottom: none; color: #00325d; }
  section.break h2 { font-size: 1.15rem; font-weight: 400; color: #5d799b; margin-top: 8px; }

  /* ── Cards ── */
  .card {
    background: #ffffff;
    border-left: 4px solid #00325d;
    padding: 14px 18px;
    margin: 8px 0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    line-height: 1.55;
  }
  .card.warn { border-left-color: #da3943; background: #fef2f2; }
  .card.tip  { border-left-color: #f6a800; background: #fffbeb; }
  .card.ok   { border-left-color: #28b223; background: #f0fdf4; }
  .card.purple { border-left-color: #5d799b; background: #f0f4fa; }

  /* ── Grids ── */
  .grid { display: grid; gap: 14px; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: 1fr 1fr 1fr; }

  /* ── Tags ── */
  .tag {
    display: inline-block; background: #00325d; color: white;
    padding: 2px 12px; border-radius: 12px; font-size: 0.76rem;
    margin-right: 6px; vertical-align: middle;
  }

  /* ── Flow arrows ── */
  .flow { display: flex; align-items: center; gap: 8px; justify-content: center; flex-wrap: wrap; margin: 20px 0; }
  .fbox {
    background: #00325d; color: white;
    padding: 10px 18px; border-radius: 10px;
    font-weight: 600; text-align: center; min-width: 88px; font-size: 0.88rem;
    box-shadow: 0 2px 6px rgba(0,50,93,0.2);
  }
  .arrow { color: #00325d; font-size: 1.6rem; font-weight: 700; }

  /* ── Fragment reveals ── */
  /* Fragments are visible by default so PPTX/PDF exports render correctly.
     The reveal script at the end hides them on load in HTML presentations. */
  .fragment { opacity: 1; transition: opacity 0.25s; }
  .fragment.f-hidden { opacity: 0; }
  .fragment.f-on { opacity: 1; }

  /* ── Illustrations ── */
  .illust {
    font-size: 4rem;
    opacity: 0.88;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.15));
  }

  /* ── Timeline ── */
  .timeline-item { display: flex; align-items: flex-start; gap: 14px; margin: 11px 0; }
  .timeline-dot {
    width: 12px; height: 12px; background: #00325d;
    border-radius: 50%; flex-shrink: 0; margin-top: 6px;
    box-shadow: 0 0 0 3px rgba(0,50,93,0.15);
  }
  .timeline-dot.dim { background: #abb8ca; box-shadow: none; }
---

<!-- _class: cover -->

<img src="images/uni-mannheim-logo.svg" style="position: absolute; top: 30px; right: 50px; width: 220px; filter: brightness(0) invert(1) opacity(0.9);">

# Facial Attractiveness Prediction from Geometric Beauty Markers

## Ethnic Bias and Cross-Dataset Generalization

<div style="margin-top: 40px; font-size: 0.92rem;">
Team 3 — IE500 Data Mining, University of Mannheim
</div>

<div style="margin-top: 10px; font-size: 0.8rem; color: rgba(255,255,255,0.55);">
Maxim Froehlich, David Siregar, Timon Kuhl, Lars Bullmahn, Jagadeesh Gunti, Simon Heinz
</div>

---

# Agenda

<div class="grid grid-2" style="margin-top: 20px;">

<div class="fragment">
<div class="card">
<strong>1. Problem & Motivation</strong><br>
Why facial attractiveness prediction matters and what can go wrong
</div>
</div>

<div class="fragment">
<div class="card">
<strong>2. Data & Features</strong><br>
3 datasets, 17,870 faces, 30 geometric beauty markers
</div>
</div>

<div class="fragment">
<div class="card">
<strong>3. Methods</strong><br>
7 models from baselines to stacking ensembles
</div>
</div>

<div class="fragment">
<div class="card">
<strong>4. Results & Fairness</strong><br>
Performance, cross-dataset generalization, and ethnic bias analysis
</div>
</div>

</div>

---

<!-- _class: break -->

# Part 1: Problem & Motivation

Can facial geometry alone predict attractiveness? And is it fair?

---

# The Problem

<div class="fragment">
<div class="card tip">
<strong>Task:</strong> Given a face image, predict the mean attractiveness rating assigned by human annotators (supervised regression)
</div>
</div>

<div class="fragment">
<div class="card warn">
<strong>Challenge:</strong> Beauty is subjective, culturally shaped, and varies across ethnic groups — models can encode demographic bias
</div>
</div>

<div class="fragment">
<div class="card purple">
<strong>Our Approach:</strong> Use interpretable geometric features (not black-box CNNs) to understand <em>which</em> proportions drive predictions and how they differ across demographics
</div>
</div>

---

# Research Questions

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>RQ1:</strong> Does a model trained on limited ethnicities (Asian/Caucasian) fail on diverse populations?</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>RQ2:</strong> Does training on more diverse data reduce racial disparities in prediction accuracy?</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>RQ3:</strong> Which geometric measurements predict attractiveness, and do patterns differ across ethnic groups?</div>
</div>
</div>

---

<!-- _class: break -->

# Part 2: Data & Features

3 datasets, 6 ethnicities, 30 beauty markers

---

# Datasets Overview

| Dataset | N | Scale | Ethnicities | Raters/Image |
|---------|---|-------|-------------|--------------|
| SCUT-FBP5500 | 5,500 | 1-5 | Asian, Caucasian | 60 |
| MEBeauty | 2,370 | 1-10 | 6 groups | ~300 |
| LiveBeauty | 10,000 | 1-5 | Asian | ~20 |

<div class="fragment">
<div class="card ok" style="margin-top: 16px;">
<strong>Combined: 17,870 faces</strong> — scores z-normalized per dataset to align different rating scales
</div>
</div>

---

# Ethnic Distribution

<div style="display: flex; gap: 20px; align-items: flex-start;">
<div style="flex: 1;">

| Ethnicity | N | % |
|-----------|---:|---:|
| Asian | 14,351 | 80.3% |
| Caucasian | 2,480 | 13.9% |
| Black | 296 | 1.7% |
| Hispanic | 296 | 1.7% |
| Mid-Eastern | 291 | 1.6% |
| Indian | 156 | 0.9% |

</div>
<div style="flex: 1;">

<div class="fragment">
<div class="card warn">
<strong>Heavy imbalance!</strong><br>
80% Asian faces due to LiveBeauty dataset. Minority groups have very few samples — this directly impacts fairness results.
</div>
</div>

</div>
</div>

---

# Feature Extraction Pipeline

<div class="flow">
<div class="fbox">Face Image</div>
<div class="arrow">&#8594;</div>
<div class="fbox">MediaPipe<br>478 Landmarks</div>
<div class="arrow">&#8594;</div>
<div class="fbox">30 Geometric<br>Ratios</div>
<div class="arrow">&#8594;</div>
<div class="fbox">Prediction</div>
</div>

<div class="fragment">
<div class="grid grid-2" style="margin-top: 14px;">
<div class="card">
<strong>Eyes (9 features)</strong><br>
canthal_tilt, eye_width_ratio, eye_area_ratio, eye_asymmetry, ...
</div>
<div class="card">
<strong>Nose (3 features)</strong><br>
nose_width_ratio, nose_length_ratio, nose_symmetry
</div>
<div class="card">
<strong>Mouth (7 features)</strong><br>
lip_fullness, mouth_chin_ratio, cupids_bow_ratio, ...
</div>
<div class="card">
<strong>Face & Proportions (11 features)</strong><br>
facial_symmetry, jaw_width_ratio, gonial_angle, phi_deviation, ...
</div>
</div>
</div>

---

# Feature Extraction: Visualization

<div style="display: flex; gap: 24px; align-items: flex-start;">
<div style="flex: 1.2;">
<img src="images/feature-extraction-overlay.jpg" style="width: 100%; max-height: 440px; object-fit: contain; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.15);">
</div>
<div style="flex: 0.8;">

<div class="fragment">
<div class="card" style="font-size: 0.85rem;">
<strong>Color coding:</strong><br>
<span style="color: #ffff00;">Yellow</span> = Canthal tilt<br>
<span style="color: #ff64ff;">Pink</span> = Eyebrow distance<br>
<span style="color: #ff00ff;">Magenta</span> = Eye spacing<br>
<span style="color: #00ff00;">Green</span> = Nose measurements<br>
<span style="color: #ff0000;">Red</span> = Lip/mouth features<br>
<span style="color: #0096ff;">Blue</span> = Jaw contour<br>
<span style="color: #ffffff; background: #333; padding: 0 4px;">White</span> = Face frame
</div>
</div>

<div class="fragment">
<div class="card tip" style="font-size: 0.85rem;">
<strong>All values are ratios</strong><br>
Normalized to face width/height for scale invariance
</div>
</div>

</div>
</div>

---

# Feature Design Principles

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Scale-invariant:</strong> All distances as ratios to face width/height — works regardless of resolution or distance</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Roll-corrected:</strong> Canthal tilt adjusted for head rotation using inter-eye baseline angle</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Symmetry-aware:</strong> 9 bilateral landmark pairs measure facial asymmetry relative to midline</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Golden ratio:</strong> phi_deviation measures departure from classical proportions (1.618)</div>
</div>
</div>

---

# PCA: Explained Variance

<div style="text-align: center;">
<img src="images/scree_plot.png" style="width: 88%; max-height: 380px; object-fit: contain; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);">
</div>

<div class="fragment">
<div class="card ok" style="margin-top: 12px;">
<strong>14 components explain 90% of variance</strong> — our 39 raw features (30 geometric + 9 expression) compress well. The first 4 PCs alone capture &gt;55%, indicating strong correlation structure among facial ratios.
</div>
</div>

---

# PCA: Which Components Predict Attractiveness?

<div style="text-align: center;">
<img src="images/component_target_correlation.png" style="width: 88%; max-height: 360px; object-fit: contain; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);">
</div>

<div class="fragment">
<div class="grid grid-2" style="margin-top: 10px;">
<div class="card">
<strong>PC2 dominates</strong> (r=0.305)<br>
The second principal component is most predictive of attractiveness — not the first (which captures overall face size variation).
</div>
<div class="card purple">
<strong>Distributed signal</strong><br>
Many PCs correlate weakly (r=0.13–0.19). Attractiveness is not captured by a single axis — it's a multi-dimensional combination of proportions.
</div>
</div>
</div>

---

<!-- _class: break -->

# Part 3: Methods

From simple baselines to stacking ensembles

---

# Model Overview

<span class="tag">Baseline</span> <span class="tag">Gradient Boosting</span> <span class="tag">Ensemble</span> <span class="tag">Neural Net</span>

<div class="fragment">

| Model | Family | Key Config |
|-------|--------|-----------|
| Global Mean | Baseline | Always predicts mean score |
| XGBoost | Boosting | 1000 trees, depth 6, lr=0.05 |
| LightGBM | Boosting | 1000 trees, leaf-wise, 31 leaves |
| CatBoost | Boosting | 1000 iters, ordered boosting |
| Stacking Ensemble | Meta-learning | XGB + RF + GBR + Ridge |
| MLP | Neural Network | 128-64-32 neurons, Adam |
| Quantile XGBoost | Boosting | Predicts median (robust) |

</div>

---

# Stacking Ensemble Architecture

<div style="display: flex; align-items: center; gap: 24px; justify-content: center; margin-top: 20px;">

<div class="fbox" style="padding: 16px 22px; font-size: 0.95rem;">30 Features</div>

<div style="display: flex; flex-direction: column; align-items: center; gap: 4px; color: #00325d; font-size: 1.4rem; font-weight: 700;">
<span>&#8594;</span>
<span>&#8594;</span>
<span>&#8594;</span>
<span>&#8594;</span>
</div>

<div style="display: flex; flex-direction: column; gap: 8px;">
<div class="fbox" style="padding: 8px 16px; font-size: 0.82rem;">XGBoost (300 trees)</div>
<div class="fbox" style="padding: 8px 16px; font-size: 0.82rem;">Random Forest (300 trees)</div>
<div class="fbox" style="padding: 8px 16px; font-size: 0.82rem;">Gradient Boosting</div>
<div class="fbox" style="padding: 8px 16px; font-size: 0.82rem;">Ridge Regression</div>
</div>

<div style="display: flex; flex-direction: column; align-items: center; gap: 4px; color: #00325d; font-size: 1.4rem; font-weight: 700;">
<span>&#8594;</span>
<span>&#8594;</span>
<span>&#8594;</span>
<span>&#8594;</span>
</div>

<div class="fbox" style="background: #00192f; padding: 16px 22px; font-size: 0.95rem;">Ridge<br>Meta-Learner</div>

</div>

<div class="fragment">
<div class="card ok" style="margin-top: 18px;">
<strong>4 base learners</strong> generate out-of-fold predictions via internal 5-fold CV. A Ridge meta-learner learns the optimal blend weights. Best overall: MAE=0.554, r=0.710.
</div>
</div>

---

# Hyperparameter Optimization

<div class="fragment">
<div class="card">
<strong>Optuna (TPE Sampler, 200 trials)</strong><br>
Searches over: max_depth [3-10], learning_rate [0.01-0.3], n_estimators [100-2000], subsample, colsample, regularization
</div>
</div>

<div class="fragment">
<div class="card tip">
<strong>Data Augmentation (4x)</strong><br>
Gaussian noise (sigma=0.02) added proportionally to each feature. Simulates landmark detection variance. Training set: ~14K &#8594; ~57K samples per fold.
</div>
</div>

<div class="fragment">
<div class="card purple">
<strong>Evaluation: 5-Fold CV</strong><br>
10% validation held out within each fold for early stopping. No information leakage between train/val/test.
</div>
</div>

---

<!-- _class: break -->

# Part 4: Results

Performance, generalization, and fairness

---

# Overall Model Performance

<span class="tag">Holdout Test</span> <span class="tag">3,574 samples</span> <span class="tag">Baseline MAE = 0.858</span>

| Model | MAE | Pearson r | vs. Baseline |
|-------|-----|-----------|--------------|
| Global Mean (baseline) | 0.858 | — | — |
| Ridge Regression | 0.600 | 0.660 | -30.0% |
| XGBoost | 0.554 | 0.708 | -35.4% |
| LightGBM | 0.556 | 0.705 | -35.2% |
| CatBoost | 0.556 | 0.709 | -35.3% |
| Stacking Ensemble | 0.554 | 0.710 | -35.4% |
| **MLP** | **0.544** | **0.714** | **-36.6%** |

<div class="fragment">
<div class="card ok">
<strong>Key finding:</strong> All models substantially beat the baseline. Gradient-boosted trees cluster at ~35% improvement; MLP achieves best holdout MAE. Even linear Ridge shows 30% signal — geometry carries real predictive power.
</div>
</div>

---

# Results vs. Baseline: Why It Matters

<div class="grid grid-2" style="margin-top: 14px;">

<div class="fragment">
<div class="card">
<strong>Baseline = Global Mean</strong><br>
Always predicts average z-score (MAE = 0.858). Any improvement must come from geometric signal.
</div>
</div>

<div class="fragment">
<div class="card ok">
<strong>36.6% Improvement (MLP)</strong><br>
MAE drops from 0.858 &#8594; 0.544. Proves 30 facial ratios carry genuine predictive information.
</div>
</div>

<div class="fragment">
<div class="card purple">
<strong>Linear vs. Nonlinear</strong><br>
Ridge: -30% | Trees: -35% | MLP: -36.6%<br>
Most signal is linear! Nonlinear models add only ~6% extra.
</div>
</div>

<div class="fragment">
<div class="card tip">
<strong>Tree Models Cluster</strong><br>
XGBoost, LightGBM, CatBoost, Ensemble all at 0.554-0.556. Architecture choice barely matters — data quality does.
</div>
</div>

</div>

---

# Cross-Dataset Generalization

> RQ1: Does training on limited ethnicities generalize poorly?

<div class="fragment">

| Experiment | MAE | Pearson r | Test N |
|-----------|-----|-----------|--------|
| SCUT &#8594; MEBeauty | 0.849 | **0.280** | 2,370 |
| SCUT &#8594; LiveBeauty | 0.764 | 0.477 | 10,000 |
| MEBeauty &#8594; SCUT | 0.770 | 0.361 | 5,500 |
| MEBeauty &#8594; LiveBeauty | 0.744 | 0.483 | 10,000 |
| LiveBeauty &#8594; MEBeauty | 0.857 | 0.418 | 2,370 |
| LiveBeauty &#8594; SCUT | 0.720 | 0.497 | 5,500 |
| **Combined (5-fold CV)** | **0.568** | **0.680** | **17,870** |

</div>

<div class="fragment">
<div class="card warn">
<strong>SCUT &#8594; MEBeauty: r = 0.28!</strong> Training on Asian/Caucasian data fails catastrophically on diverse populations. Combining all datasets substantially improves generalization.
</div>
</div>

---

# Comparison to CNN State-of-the-Art

<span class="tag">SCUT-FBP5500 Benchmark</span>

| Method | Input | Pearson r |
|--------|-------|-----------|
| ResNet-18 (Liang et al.) | Raw pixels | 0.89 |
| ArcFace + Ridge (Deng et al.) | CNN embeddings | 0.91 |
| **Our XGBoost** | **30 geometric features** | **0.71** |
| **Our MLP** | **30 geometric features** | **0.71** |

<div class="fragment">
<div class="card purple" style="margin-top: 14px;">
<strong>Gap Analysis:</strong> CNN r &#8776; 0.89-0.92 vs. Our r = 0.71. The 20% gap arises from fundamental differences in what each approach can learn.
</div>
</div>

<div class="fragment">
<div class="card tip">
<strong>Important caveat:</strong> CNN results are within-dataset (SCUT only, ethnically homogeneous). Our evaluation uses a harder multi-ethnic corpus where cross-cultural variation adds noise.
</div>
</div>

---

# Why Are There Differences to CNN Results?

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>1. Texture & Appearance Cues:</strong> CNNs capture skin quality, hair style, grooming, and color — features that geometry alone cannot encode. This likely accounts for the <em>majority</em> of the gap.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>2. Holistic Pattern Learning:</strong> Deep networks learn high-level "gestalt" representations (face shape as a whole) rather than relying on predefined ratios between landmark pairs.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>3. Evaluation Setting Differs:</strong> Published CNN results use within-dataset splits on ethnically homogeneous data. Our evaluation combines 3 datasets with 6 ethnic groups — a harder, noisier task.</div>
</div>
</div>

<div class="fragment">
<div class="card ok" style="margin-top: 14px;">
<strong>Our advantage:</strong> Full interpretability + bias auditability. SHAP reveals exactly which proportions drive predictions per group — something no CNN can offer.
</div>
</div>

---

# Why Do Our Models Differ From Each Other?

<div class="grid grid-2" style="margin-top: 14px;">

<div class="fragment">
<div class="card">
<strong>Linear (Ridge): -30%</strong><br>
Captures direct proportional relationships. Proves most attractiveness signal is linear in geometric ratios.
</div>
</div>

<div class="fragment">
<div class="card">
<strong>Tree Models: -35%</strong><br>
Capture feature interactions and nonlinearities (e.g., wide nose + narrow eyes together). +5% over linear.
</div>
</div>

<div class="fragment">
<div class="card ok">
<strong>MLP: -36.6%</strong><br>
Learns smooth continuous interaction surfaces. Best at modeling subtle nonlinear combinations of ratios.
</div>
</div>

<div class="fragment">
<div class="card purple">
<strong>Ensemble: -35.4%</strong><br>
Combining diverse base learners adds robustness but not accuracy — individual trees already saturate the geometric signal.
</div>
</div>

</div>

<div class="fragment">
<div class="card tip" style="margin-top: 10px;">
<strong>Key insight:</strong> The 30-feature ceiling means architecture matters less than feature quality. All models converge near MAE &#8776; 0.55 — the limit of geometry-only prediction.
</div>
</div>

---

<!-- _class: break -->

# Fairness & Bias Analysis

Multi-model comparison of ethnic disparities

---

# Fairness: Multi-Model Bias Comparison

> Do different model architectures produce different fairness outcomes?

<div class="fragment" style="font-size: 0.88rem;">

| Model | Asian | Cauc. | Black | Hisp. | Mid-E. | Indian | **Gap** |
|-------|-------|-------|-------|-------|--------|--------|---------|
| Ridge | 0.587 | 0.616 | 0.582 | 0.762 | 0.743 | 1.038 | **0.456** |
| Decision Tree | 0.648 | 0.648 | 0.689 | 0.840 | 0.716 | 1.007 | 0.359 |
| Random Forest | 0.605 | 0.635 | 0.607 | 0.780 | 0.686 | 1.020 | 0.415 |
| XGBoost | 0.552 | 0.601 | 0.616 | 0.717 | 0.718 | 0.876 | 0.324 |
| **MLP** | **0.531** | **0.575** | **0.593** | **0.698** | **0.703** | **0.851** | **0.320** |

</div>

<div class="fragment">
<div class="card warn" style="margin-top: 10px;">
<strong>Indian group consistently worst</strong> across ALL model families. This confirms bias is driven by <em>data scarcity</em> (N=156, vs. 14,351 Asian) — not model architecture.
</div>
</div>

---

# Absolute Fairness Gap

<div class="fragment">
<div class="card" style="text-align: center; font-size: 1.1rem; padding: 20px;">
<strong>Fairness Gap = max(group MAE) − min(group MAE)</strong>
</div>
</div>

<div class="grid grid-2" style="margin-top: 18px;">

<div class="fragment">
<div class="card">
<strong>Gap by Model Family</strong><br><br>
Ridge: <strong>0.456</strong> z-score<br>
Random Forest: <strong>0.415</strong> z-score<br>
Decision Tree: <strong>0.359</strong> z-score<br>
XGBoost: <strong>0.324</strong> z-score<br>
MLP: <strong>0.320</strong> z-score
</div>
</div>

<div class="fragment">
<div class="card purple">
<strong>Interpretation</strong><br><br>
&#8226; More expressive models <em>reduce</em> but don't eliminate the gap<br>
&#8226; Range: 0.32–0.46 z-score units<br>
&#8226; Best-served (Asian) vs. worst-served (Indian): gap persists regardless of architecture<br>
&#8226; 92× fewer samples for Indian vs. Asian
</div>
</div>

</div>

<div class="fragment">
<div class="card ok" style="margin-top: 14px;">
<strong>Conclusion:</strong> The fairness problem is fundamentally a <em>data problem</em>. No model architecture can compensate for having 92× fewer training examples for minority groups.
</div>
</div>

---

# Bias Deep-Dive: What Drives the Gap?

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Data Imbalance:</strong> 80.3% Asian, 0.9% Indian. Models learn Asian beauty patterns well but lack signal for underrepresented groups.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Annotation Bias:</strong> Raters are predominantly East Asian (LiveBeauty) — ground truth itself reflects cultural preferences of the majority group.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Feature Magnitude Differences:</strong> SHAP shows nose_width has 2.6× higher importance for Black faces (0.702) vs. Asian (0.270) — same features, different reliance patterns.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Gender Gap (small):</strong> Male MAE = 0.54, Female MAE = 0.61 (gap: 0.07). Much smaller than ethnic gap — gender is better balanced in training data.</div>
</div>
</div>

---

# SHAP Feature Importance

> RQ3: Which features matter, and do they differ by ethnicity?

<div class="grid grid-2" style="margin-top: 14px;">

<div class="fragment">
<div class="card">
<strong>Universal Top-3</strong><br>
1. Nose width ratio<br>
2. Eye width ratio<br>
3. Mouth-chin ratio<br>
<em>Consistent across all groups</em>
</div>
</div>

<div class="fragment">
<div class="card purple">
<strong>Key Difference</strong><br>
For Black faces, nose_width SHAP = <strong>0.702</strong><br>
For Asian faces, nose_width SHAP = <strong>0.270</strong><br>
<strong>2.6× difference</strong> — same feature, vastly different reliance
</div>
</div>

</div>

<div class="fragment">
<div class="card warn" style="margin-top: 12px;">
<strong>Canthal tilt</strong> — despite popularity in online beauty discourse — ranks near the <em>bottom</em> of feature importance across all groups. Symmetry features also show r &lt; 0.04 (likely a MediaPipe artifact).
</div>
</div>

---

# Per-Ethnicity SHAP Breakdown

<div style="font-size: 0.92rem;">

| Ethnicity | #1 | #2 | #3 | #4 | #5 | Top SHAP |
|-----------|----|----|----|----|-----|----------|
| Asian | nose_width | eye_width | eye_area | mouth_chin | lip_fullness | 0.270 |
| Black | **nose_width** | mouth_chin | eye_width | eye_area | interpupillary | **0.702** |
| Caucasian | mouth_chin | nose_width | eye_width | eye_area | upper_lip | 0.284 |
| Hispanic | nose_width | eye_width | mouth_chin | eye_area | upper_lip | 0.337 |
| Indian | nose_width | eye_width | mouth_chin | eye_area | eye_spacing | 0.340 |
| Mid-Eastern | nose_width | eye_width | mouth_chin | eye_area | eye_spacing | 0.327 |

</div>

<div class="fragment">
<div class="card ok" style="margin-top: 10px;">
<strong>Both universal and culture-specific patterns exist.</strong> Top-3 features are shared globally, but magnitudes and 4th/5th features vary — beauty perception has universal geometric foundations with group-specific modulations.
</div>
</div>

---

<!-- _class: break -->

# Key Takeaways

What did we learn?

---

# Summary of Findings

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Geometry works:</strong> 30 facial ratios achieve r=0.714, MAE=0.544 — 36.6% improvement over baseline from simple measurements</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Diversity matters:</strong> Models trained on 2 ethnicities fail on 6 (r drops from 0.71 to 0.28). Combining datasets fixes this.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Data scarcity drives unfairness:</strong> Fairness gap 0.32–0.46 z-score across all model families. Indian group (N=156) consistently worst — data problem, not model problem.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Beauty has universal + specific components:</strong> Same top-3 features everywhere, but 2.6× magnitude difference across groups.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>CNN gap explained:</strong> Our r=0.71 vs. CNN r=0.89-0.92 — gap driven by texture cues and easier evaluation settings, not model weakness.</div>
</div>
</div>

---

# Limitations

<div class="grid grid-2">

<div class="fragment">
<div class="card warn">
<strong>Data Imbalance</strong><br>
80% Asian, 0.9% Indian. Fairness gap (0.32 z-score) reflects insufficient minority representation.
</div>
</div>

<div class="fragment">
<div class="card warn">
<strong>Annotation Bias</strong><br>
Predominantly East Asian raters (LiveBeauty) — ground truth itself encodes majority-group preferences.
</div>
</div>

<div class="fragment">
<div class="card warn">
<strong>No Texture (= CNN gap)</strong><br>
Geometry ignores skin, hair, grooming — explains most of the r=0.71 vs. r=0.89 gap to CNN methods.
</div>
</div>

<div class="fragment">
<div class="card warn">
<strong>Regression-to-the-Mean</strong><br>
Std ratios 0.63–0.85: predictions compress toward zero. Ranker XGBoost partially addresses this via rank targets.
</div>
</div>

</div>

---

# Discussion: What Worked and What Didn't

<div class="grid grid-2" style="margin-top: 14px;">

<div class="fragment">
<div class="card ok">
<strong>Worked Well</strong><br>
&#8226; Geometric features alone: 36.6% over baseline<br>
&#8226; Multi-dataset training: fixes cross-ethnic failure<br>
&#8226; SHAP interpretability: reveals group-specific patterns<br>
&#8226; Optuna HPO: consistent improvement across models
</div>
</div>

<div class="fragment">
<div class="card warn">
<strong>Fell Short</strong><br>
&#8226; r=0.71 vs. CNN r=0.89 — geometry alone has a ceiling<br>
&#8226; Minority fairness: can't fix 92× data imbalance with algorithms<br>
&#8226; Symmetry features: near-zero signal (MediaPipe artifact)<br>
&#8226; Regression-to-mean: all models compress predictions
</div>
</div>

</div>

<div class="fragment">
<div class="card purple" style="margin-top: 14px;">
<strong>Central insight:</strong> The gap between our models (r=0.71) and the theoretical ceiling is split into two parts: ~50% is texture (skin, hair), ~50% is evaluation difficulty (multi-ethnic vs. single-dataset). Geometry captures the structural foundation of beauty but not its surface.
</div>
</div>

---

# Discussion: On Fairness vs. Accuracy

<div class="fragment">
<div class="card" style="text-align: center; font-size: 1.05rem; padding: 18px;">
Can we have <strong>both</strong> high accuracy <strong>and</strong> ethnic fairness?
</div>
</div>

<div class="grid grid-2" style="margin-top: 16px;">

<div class="fragment">
<div class="card">
<strong>Evidence for YES</strong><br>
&#8226; Combining datasets improved both overall MAE <em>and</em> cross-ethnic r<br>
&#8226; More expressive models (MLP) reduce the fairness gap (0.32 vs. Ridge 0.46)<br>
&#8226; Same top features globally suggests shared beauty signal
</div>
</div>

<div class="fragment">
<div class="card warn">
<strong>Evidence for NO (with current data)</strong><br>
&#8226; Indian MAE remains 60% worse even with best model<br>
&#8226; Annotation bias cannot be fixed post-hoc<br>
&#8226; 92× sample imbalance creates irrecoverable statistical weakness
</div>
</div>

</div>

<div class="fragment">
<div class="card tip" style="margin-top: 14px;">
<strong>Answer:</strong> Fairness is achievable in principle, but requires <em>balanced data collection</em> — no algorithmic trick can substitute for representative training data.
</div>
</div>

---

# Outlook: Future Work

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Balanced Data Collection:</strong> Acquire equal samples per ethnic group. Evaluate whether fairness gaps narrow with equal representation — testing if the problem is truly data-driven.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Rater Demographic Integration:</strong> Separate rater-ethnicity bias from ratee-ethnicity bias. Does an Asian rater rating a Black face produce systematically different scores?</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Hybrid Geometry + Texture:</strong> Combine our 30 landmark features with deep embeddings (e.g., ArcFace) to close the gap to CNN SOTA while maintaining partial interpretability.</div>
</div>
</div>

<div class="fragment">
<div class="timeline-item">
<div class="timeline-dot"></div>
<div><strong>Fairness-Aware Training:</strong> Evaluate loss functions that penalize per-group disparities and quantify the accuracy-fairness tradeoff curve.</div>
</div>
</div>

---

<!-- _class: cover -->

<img src="images/uni-mannheim-logo.svg" style="position: absolute; top: 30px; right: 50px; width: 220px; filter: brightness(0) invert(1) opacity(0.9);">

# Questions?

## Thank you!

<div style="margin-top: 30px; font-size: 0.9rem;">
Team 3 — IE500 Data Mining, University of Mannheim
</div>


<!-- Fragment reveal script - DO NOT DELETE -->
<script>
(function() {
  if (typeof window === 'undefined') return;

  // Hide all fragments on load (only in HTML/browser — PPTX/PDF won't run this)
  document.querySelectorAll('.fragment').forEach(function(f) {
    f.classList.add('f-hidden');
  });

  function getActiveSection() {
    var selectors = ['svg[data-marpit-svg].bespoke-active','svg[data-marpit-svg].bespoke-marp-active','svg[data-marpit-svg].bespoke-current'];
    for (var s = 0; s < selectors.length; s++) {
      var el = document.querySelector(selectors[s]);
      if (el) return el.querySelector('section');
    }
    var svgs = document.querySelectorAll('svg[data-marpit-svg]');
    var cx = window.innerWidth / 2, cy = window.innerHeight / 2;
    for (var i = 0; i < svgs.length; i++) {
      var r = svgs[i].getBoundingClientRect();
      if (r.left <= cx && cx <= r.right && r.top <= cy && cy <= r.bottom) return svgs[i].querySelector('section');
    }
    return null;
  }

  function revealNext() {
    var section = getActiveSection();
    if (!section) return false;
    var frag = section.querySelector('.fragment.f-hidden');
    if (frag) { frag.classList.remove('f-hidden'); frag.classList.add('f-on'); return true; }
    return false;
  }

  function hideLastRevealed() {
    var section = getActiveSection();
    if (!section) return false;
    var frags = section.querySelectorAll('.fragment.f-on');
    if (frags.length > 0) { frags[frags.length - 1].classList.remove('f-on'); frags[frags.length - 1].classList.add('f-hidden'); return true; }
    return false;
  }

  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(m) {
      if (m.attributeName === 'class' && !m.target.classList.contains('bespoke-active')) {
        var section = m.target.querySelector('section');
        if (section) section.querySelectorAll('.fragment.f-on').forEach(function(f) { f.classList.remove('f-on'); f.classList.add('f-hidden'); });
      }
    });
  });
  document.querySelectorAll('svg[data-marpit-svg]').forEach(function(svg) {
    observer.observe(svg, { attributes: true });
  });

  document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      if (revealNext()) { e.stopImmediatePropagation(); e.preventDefault(); }
    } else if (e.key === 'ArrowLeft') {
      if (hideLastRevealed()) { e.stopImmediatePropagation(); e.preventDefault(); }
    }
  }, true);

  document.addEventListener('mousedown', function(e) {
    if (e.button !== 0) return;
    if (revealNext()) { e.stopImmediatePropagation(); e.preventDefault(); }
  }, true);
})();
</script>
