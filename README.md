# 🩸 OpenVET Simulator (Open-Source Viscoelastic Testing)

**A mechanistic, reduced-order computational framework for viscoelastic coagulation testing.**

Commercial rotational thromboelastometry (ROTEM) analyzers operate as proprietary "black boxes," obscuring the mechanical and computational algorithms that translate clot formation into viscoelastic amplitude. This project deconstructs that black box by providing a transparent, open-source mathematical model and interactive web application that simulates whole-blood coagulation dynamics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-ff4b4b.svg)
![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.21922911-lightgreen)](https://doi.org/10.5281/zenodo.21922911)

---

## 📖 Table of Contents
- [Overview](#overview)
- [The Scientific Monograph](#the-scientific-monograph)
- [Key Features](#key-features)
- [Installation & Usage](#installation--usage)
- [Mathematical Framework](#mathematical-framework)
- [Disclaimer](#disclaimer)

---

## 🌐 Overview

This simulator implements a reduced-order ordinary differential equation (ODE) system to model the core hemostatic network: thrombin generation, fibrin polymerization, fibrinolysis, and platelet activation. Biochemical states are mapped to viscoelastic amplitude (in millimeters) via an additive Clot Elasticity (CE) transformation.

The application allows users to select specific ROTEM assays (EXTEM, FIBTEM, APTEM, INTEM) and apply pathological presets (Hypofibrinogenemia, Thrombocytopenia, Hyperfibrinolysis) to observe their effects on the viscoelastic trace in real time.

---

## 📘 The Scientific Monograph

The mathematical and computational foundations of this software are rigorously documented in the accompanying technical monograph:

> **Mathematical, Physical, and Computational Principles of Rotational Thromboelastometry: An Open-Source Computational Framework**
> *Mohammed W. Hammami (2026)*

The monograph details the exact ODE specifications, parameter provenance (derived from Hockin et al., Weisel & Nagaswami, Ouedraogo et al., and Fogelson et al.), numerical implementation (LSODA solver configuration), and phenotypic validation against clinical literature. 

📄 **[Read the Full Monograph (PDF)](./MONOGRAPH.pdf)**

---

## ✨ Key Features

- **Dual Execution Modes:**
  - **Educational Mode:** An 8-state Hill-gated heuristic model. Fast, interactive, and tunable via UI sliders.
  - **Research Mode:** An 11-state direct Michaelis-Menten mathematical implementation using exact literature kinetics (Hockin-Mann).
- **Interactive Clinical Dashboard:** Built with Streamlit and Plotly. Includes time-scrubbing sliders, live CT/MCF/LI60 metrics, and clinical phase indicators.
- **Pathology Presets:** Simulate hypofibrinogenemia, thrombocytopenia, and the lethal hyperfibrinolytic "Death Diamond" phenotype.
- **Reproducibility Manifest:** Every simulation generates a downloadable JSON configuration file containing the exact model version, parameters, assay definition, and solver tolerances.
- **Data Export:** Export simulated traces to CSV or JSON for external analysis.

---

## 💻 Installation & Usage

To run the simulator locally, ensure you have Python 3.8+ installed.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/OpenROTEM-Simulator.git
cd OpenROTEM-Simulator
```

**2. Install the required dependencies:**
```bash
pip install streamlit numpy scipy plotly pandas
```

**3. Run the Streamlit app:**
```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

---

## 🧮 Mathematical Framework

The model translates biochemical states into viscoelastic amplitude $A(t)$ via Clot Elasticity (CE) space. To ensure biomechanical fidelity, the platelet contribution is gated by the presence of intact polymerized fibrin ($F_p$). This ensures that if the fibrin mesh is degraded by plasmin, the platelet aggregate loses structural integrity, accurately simulating the rapid amplitude decay seen in hyperfibrinolysis.

$$CE_{fibrin} = CEmax_{fib} \cdot \frac{[F_p]}{[F_p] + K_F}$$
$$CE_{platelet} = CEmax_{plt} \cdot \frac{[Pa]}{[Pa] + K_{Pa}} \cdot \left( \frac{[F_p]}{[F_p] + K_{gate}} \right)$$
$$A(t) = \frac{100 \cdot (CE_{fibrin} + CE_{platelet})}{100 + (CE_{fibrin} + CE_{platelet})}$$

*Note: Because the reduced-order model operates in normalized concentration spaces rather than strict Molar units, effective rate constants are calibrated to preserve proportional relationships while maintaining numerical stability within the ODE solver.*

---

## ⚖️ License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- **Educational & Research Use:** Allowed and encouraged.
- **Commercial Use:** Strictly prohibited without explicit written permission from the author.

---

## ⚠️ Disclaimer

This software is an **open-source computational simulator** intended strictly for educational purposes, mechanistic research, and software experimentation. 

- It is **not** a clinical diagnostic device.
- It is **not** a reverse-engineered implementation of proprietary commercial algorithms.
- It should **not** be used to inform clinical decision-making, predict patient outcomes, or guide medical therapy.
