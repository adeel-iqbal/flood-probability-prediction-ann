# 🌊 Flood Probability Prediction using Artificial Neural Network

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

## 📋 Overview

This project implements an **Artificial Neural Network (ANN)** to predict flood probability based on environmental, geographical, and human-influenced factors. The model analyzes 20 different risk factors and outputs the likelihood of flood occurrence as a percentage.

## ✨ Features

- **High Accuracy**: Achieves ~99.98% R² score on test data
- **Comprehensive Risk Assessment**: Analyzes 20 key flood risk factors
- **Easy Integration**: Pre-trained model ready for deployment
- **Feature Importance Analysis**: Identifies most influential factors
- **Scalable Architecture**: Built with TensorFlow/Keras for production use

## 🏗️ Project Structure

```
flood-prediction/
├── flood.csv                    # Raw dataset
├── flood_predictor.ipynb        # Training model
├── flood_ann_model.h5           # Trained ANN model
├── flood_scaler.save            # MinMaxScaler for preprocessing
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adeel-iqbal/flood-probability-prediction-ann.git
   cd flood-probability-prediction-ann
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Alternative - Install manually:**
   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn joblib
   ```

## 🚀 Quick Start

### Load and Use the Model

```python
import pandas as pd
import joblib
from tensorflow import keras

# Load trained model and scaler
model = keras.models.load_model("flood_ann_model.h5")
scaler = joblib.load("flood_scaler.save")

# Example input (ordinal values for 20 risk factors)
sample_input = {
    'MonsoonIntensity': 3,
    'TopographyDrainage': 8,
    'RiverManagement': 6,
    'Deforestation': 6,
    'Urbanization': 4,
    'ClimateChange': 4,
    'DamsQuality': 6,
    'Siltation': 2,
    'AgriculturalPractices': 3,
    'Encroachments': 2,
    'IneffectiveDisasterPreparedness': 5,
    'DrainageSystems': 10,
    'CoastalVulnerability': 7,
    'Landslides': 4,
    'Watersheds': 2,
    'DeterioratingInfrastructure': 3,
    'PopulationScore': 4,
    'WetlandLoss': 3,
    'InadequatePlanning': 2,
    'PoliticalFactors': 6
}

# Prepare input and predict
input_df = pd.DataFrame([sample_input])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled).flatten()[0] * 100

print(f"Predicted Flood Probability: {prediction:.2f}%")
```

## 📊 Input Features

The model requires 20 ordinal-encoded features representing different flood risk factors:

| Feature | Range | Description |
|---------|-------|-------------|
| MonsoonIntensity | 0-16 | Intensity of monsoon rainfall |
| TopographyDrainage | 0-18 | Land topography and drainage efficiency |
| RiverManagement | 0-16 | Quality of river management systems |
| Deforestation | 0-17 | Level of deforestation in the area |
| Urbanization | 0-17 | Degree of urban development |
| ClimateChange | 0-17 | Impact of climate change factors |
| DamsQuality | 0-16 | Quality and condition of dams |
| Siltation | 0-16 | Level of river/water body siltation |
| AgriculturalPractices | 0-16 | Impact of agricultural activities |
| Encroachments | 0-18 | Illegal encroachments on water bodies |
| IneffectiveDisasterPreparedness | 0-16 | Disaster preparedness effectiveness |
| DrainageSystems | 0-17 | Quality of drainage infrastructure |
| CoastalVulnerability | 0-17 | Vulnerability of coastal areas |
| Landslides | 0-16 | Risk of landslides |
| Watersheds | 0-16 | Watershed management quality |
| DeterioratingInfrastructure | 0-17 | Infrastructure deterioration level |
| PopulationScore | 0-19 | Population density impact |
| WetlandLoss | 0-22 | Loss of wetland areas |
| InadequatePlanning | 0-16 | Urban planning inadequacy |
| PoliticalFactors | 0-16 | Political influence on flood management |

> **Note**: All values are ordinal encodings where 0 represents minimal impact and higher values represent increased severity/impact.

## 🔬 Model Architecture

- **Type**: Multi-layer Artificial Neural Network
- **Framework**: TensorFlow/Keras
- **Input Features**: 20 normalized features
- **Output**: Flood probability (0-1, converted to percentage)
- **Optimization**: Early stopping to prevent overfitting
- **Preprocessing**: MinMaxScaler normalization

## 📈 Performance Metrics

| Metric | Training | Testing |
|--------|----------|---------|
| **R² Score** | ~0.9999 | ~0.9998 |
| **MAE** | Low | Low |
| **RMSE** | Low | Low |

### Top Influential Features
1. Encroachments
2. River Management
3. Coastal Vulnerability
4. Dams Quality
5. Urbanization

## 🛠️ Usage Scenarios

### 1. Disaster Management
- Early warning systems
- Risk assessment for vulnerable areas
- Resource allocation planning

### 2. Urban Planning
- Land use planning in flood-prone areas
- Infrastructure development decisions
- Environmental impact assessment

### 3. Research & Analysis
- Climate change impact studies
- Policy effectiveness evaluation
- Historical flood pattern analysis

## 📝 API Integration

```python
def predict_flood_probability(features_dict):
    """
    Predict flood probability from input features
    
    Args:
        features_dict (dict): Dictionary containing 20 flood risk features
    
    Returns:
        float: Predicted flood probability as percentage
    """
    input_df = pd.DataFrame([features_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled).flatten()[0] * 100
    return round(prediction, 2)
```

## 📞 Contact

**Adeel Iqbal Memon**
- 📧 Email: adeelmemon096@yahoo.com
- 💼 LinkedIn: [linkedin.com/in/adeeliqbalmemon](https://linkedin.com/in/adeeliqbalmemon)
- 🐱 GitHub: [github.com/adeel-iqbal](https://github.com/adeel-iqbal)

## 🙏 Acknowledgments

- Dataset contributors and flood management experts
- TensorFlow/Keras development team
- Open source community for tools and libraries

## ⚠️ Disclaimer

This model is designed for research and educational purposes. The model achieved excellent performance on the training dataset, but real-world flood prediction requires:

- **Professional Validation**: Consult with hydrologists and disaster management experts
- **Local Context**: Consider local geographical and meteorological conditions  
- **Official Data**: Use alongside official meteorological and hydrological monitoring
- **Continuous Updates**: Regular retraining with new data for accuracy maintenance

The predictions should be used as supplementary information for informed decision-making, not as the sole basis for critical flood management decisions.

---

⭐ **Star this repository if you found it helpful!**
