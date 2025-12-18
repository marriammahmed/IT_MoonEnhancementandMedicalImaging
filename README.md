# Astronomical Image Enhancement and Medical Image Processing

## 🌌 Project Overview

This project implements classical and scientifically grounded image enhancement techniques commonly used in astronomy, medical imaging, and computer vision. All algorithms are implemented explicitly using **NumPy-based numerical operations**, following the course requirement to avoid black-box image processing.The module integrates into a shared image viewer framework and provides two enhancement pipelines optimized for raw or minimally processed images.

### Failed Algorithms on Blackhole
1. Relativistic Doppler Boosting
2. CLAHE Enhancement
3. Sobel and Canny Edge Detection

### Implemented Algorithms
1. **Moon Detail Enhancement**: CLAHE, Unsharp Masking, Global Contrast Boost
2. **Medical Imaging**: Histogram Equaliation

---
### Blackhole attempts – Algorithm Comparison

<table>
  <tr>
    <th align="center">CLAHE Enhancement</th>
    <th align="center">Relativistic Doppler Boosting</th>
    <th align="center">Sobel Edge Detection</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://i.imgur.com/s1TAUy3.png" width="280"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/e9S9Wg5.png" width="280"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/MSaWeUw.png" width="280"/>
    </td>
  </tr>
</table>


### 📸 Successful Attempts and Results – Full Transformation Grid

<table>
  <tr>
    <th align="center">1</th>
    <th align="center">2</th>
    <th align="center">3/th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://i.imgur.com/kw77QbH.png" width="260"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/K28Eu8U.png" width="260"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/aQ2Lz0s.png" width="260"/>
    </td>
  </tr>

  <tr>
    <th align="center">4</th>
    <th align="center">5</th>
    <th align="center">6</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://i.imgur.com/9Qb1IB6.png" width="260"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/SrM9Eic.png" width="260"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/tJdRjpN.png" width="260"/>
    </td>
  </tr>

  <tr>
    <th align="center">7</th>
    <th align="center">8</th>
    <th align="center">9</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://i.imgur.com/mzgYcSH.png" width="260"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/UUuIeFq.png" width="260"/>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/CyCNhyk.png" width="260"/>
    </td>
  </tr>
</table>


---

## ❌ Failed Attempt: Black Hole Image Enhancement

### Why the Initial Approach Failed

The first experiment attempted to enhance the **Event Horizon Telescope (EHT) black hole image** using classical image processing techniques. This attempt failed for fundamental physical and computational reasons.

### Techniques Tested

#### 1️⃣ CLAHE (Contrast Limited Adaptive Histogram Equalization)

**What Was Attempted:**  
CLAHE was applied to increase local contrast in the black hole image.

**Why It Failed:**  
- EHT images are already reconstructed using interferometry and optimization algorithms
- Local contrast is already maximized during scientific reconstruction
- CLAHE amplified noise and artifacts without revealing new structure

---

#### 2️⃣ Doppler Boosting (Astrophysical Effect)

**Scientific Principle:**  
Doppler boosting increases observed brightness when relativistic plasma moves toward the observer:

\[
I_{observed} \propto I_{emitted} \cdot \delta^3
\]

**Why It Failed in Post-Processing:**  
- Doppler boosting is a **physical emission phenomenon**, not a post-processing filter
- The EHT reconstruction already encodes Doppler boosting
- Image enhancement cannot recreate relativistic effects after data reconstruction

---

#### 3️⃣ Sobel & Canny Edge Detection (Applied to Black Hole Image)

**Scientific Principle:**  
Both Sobel and Canny detect **intensity gradients**, identifying sharp spatial transitions.

**Sobel Model:**
- Computes first-order derivatives in x and y directions
- Highlights gradient magnitude

**Canny Pipeline:**
1. Gaussian smoothing
2. Gradient computation
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

**Why It Failed:**  
- Black hole images do not contain classical edges
- The emission ring is diffuse, probabilistic, and reconstruction-based
- Edge detectors falsely highlighted noise and reconstruction artifacts

**Conclusion:**  
Classical edge detection assumes object boundaries, which do not exist in interferometric black hole imagery.

---

## ✅ Successful Enhancement Pipelines

After abandoning black hole data, the project pivoted to **raw and minimally processed images**, where classical image processing is scientifically valid.

---

## 🌕 Astronomical Imaging: Moon Detail Enhancement

### 1️⃣ CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Scientific Principle:**  
CLAHE enhances **local contrast** by applying histogram equalization independently to small tiles.

**Model Used:**
- Tile-based histograms
- CDF-based intensity redistribution
- Histogram clipping to suppress noise

\[
I_{new}(x,y) = CDF_{tile}(I(x,y)) \times 255
\]

**Why It Works Here:**  
- Lunar images have uneven illumination
- Surface features benefit from local contrast enhancement
- Unlike bla


---

### 2️⃣ Unsharp Masking (High-Frequency Enhancement)

**Scientific Principle:**  
Enhances edges and textures by amplifying high-frequency components.

**Model:**
\[
I_{sharpened} = I + \alpha (I - G_\sigma(I))
\]

**Why It Works:**  
- Craters and ridges are high-frequency features
- Sharpening reveals surface texture without altering global brightness

---

### 3️⃣ Gradient-Based Contrast Boost (Astrophotography Optimization)

**Scientific Principle:**  
Global contrast boosting separates intensities around the mean:

\[
I_{new} = \mu + \beta (I - \mu)
\]

**Why It Works:**  
- Enhances depth perception of lunar terrain
- Makes geological features visually distinct

---

## 🏥 Medical-Style Imaging: Histogram Equalization

### 4️⃣ Histogram Equalization

**Scientific Principle:**  
Redistributes pixel intensities uniformly across the available dynamic range.

**Model Used:**
- Global histogram computation
- CDF normalization
- Intensity remapping

\[
I_{new} = \frac{CDF(I) - CDF_{min}}{N - CDF_{min}} \times 255
\]

**Why It Works:**  
- Extremely effective for low-contrast images
- Widely used in X-ray, CT, and MRI enhancement
- Reveals hidden details in underexposed regions


---

## 🔭 The Black Hole Imaging Challenge

### Why Not Use Black Hole Images?
Why Not Use Black Hole Images?
During development, I initially wanted to demonstrate these algorithms on the famous M87 black hole image from the Event Horizon Telescope (EHT). However, I encountered a fundamental problem that highlights the difference between raw astronomical data and public-facing science images:
The Problem with EHT Data

No Raw Images Available: The EHT doesn't produce "raw images" in the traditional sense. The data consists of voltage measurements from radio telescopes scattered across the globe, not pixel arrays.
Already Heavily Processed: The iconic orange ring image you've seen is the result of:

Complex interferometry combining data from 8 telescopes
Computational reconstruction algorithms
Noise reduction and artifact removal
Color mapping (radio wavelengths mapped to visible spectrum)
Multiple iterations of processing and validation


No Enhancement Possible: Because the public image is already optimally processed, applying additional enhancement algorithms provides no meaningful improvement. The image is already at its peak quality.

This limitation taught me an important lesson: enhancement algorithms are most effective on raw or minimally-processed data, not on publication-ready scientific imagery from major observatories like NASA, ESA, or EHT.


---

## 🧠 Key Lesson

> **Image enhancement is only effective on raw or minimally processed data.**

- Black hole images are physics-driven reconstructions, not pixel measurements
- Lunar and medical images contain real intensity gradients
- Classical image processing enhances *measured data*, not inferred data

This realization guided the final design of the project and ensured scientifically valid results.
