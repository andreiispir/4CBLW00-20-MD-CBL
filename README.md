# Intelligent Deployment of Police Resources Based on Crime Forecasting

This repository contains the codebase for our project focused on improving police resource allocation through crime forecasting techniques. By analyzing historical crime data and applying machine learning models, we aim to predict future crime trends and assist in proactive police deployment.

## 📁 Branch Overview

Each branch corresponds to a major component of the project:

- **`main`** – Hosts the **dashboard** implementation that visualizes crime predictions and allocations.
- **`Andrei-Prophet`** – Contains the **Prophet** forecasting script used for modeling temporal crime patterns.
- **`Alicja`** – Includes the **ILP (Integer Linear Programming)** script for optimizing the deployment of police units.
- **`Luuk`** – Holds the **XGBoost** script, used as a second predictive model to compare against Prophet.

## 🔧 Technologies Used

- Python 3.x
- Pandas, NumPy
- Plotly / Dash (for dashboard)
- Facebook Prophet
- XGBoost
- PuLP (for ILP optimization)

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/andreiispir/4CBLW00-20-MD-CBL.git
