# CSH-film-cooling-effectiveness-prediction
# VP-GSM Surrogate Model for Film Cooling Effectiveness

This repository provides the implementation of two machine-learning-based surrogate models for predicting film cooling effectiveness of converging slot holes (CSH):  
1. **Direct Prediction BPNN**  
2. **Gaussian-Constrained Prediction (VP-GSM)**  

Representative results are provided for Case 1 (M=1.0, DR=2.0, α=30°, L/D=3.2, s/l=3.0), showing the workflow from parameter input to contour visualization.

---

## File Structure
- **shuchu_yuntu_zhijie_tecplot_xd=1.py**  
  Direct BPNN prediction program.  
  Outputs a CSV file (`output_zhijie_1.csv`) with three columns: `x`, `y`, and predicted adiabatic film cooling effectiveness `η`.

- **shuchu_yuntu_yueshu_tecplot_xd=1.py**  
  Gaussian-constrained VP-GSM prediction program.  
  Outputs a CSV file with the same format.

- **output_zhijie_16.csv**  
  Example direct prediction results for Case 1.

- **1_lengxiao.csv** 
  CFD benchmark results for Case 1, extracted from Fluent/CFD-Post.

- **Training Scripts**  
  - `BPNN_zhijie.py` / `raw_single_zhijie.py`  
  - `BPNN_yueshu.py` / `raw_single_yueshu.py`  

---

## Input Parameters
The models take **five normalized parameters** as input:
1. Blowing Ratio (M), range: 0.5 – 1.5  
2. Density Ratio (DR), range: 1.5 – 2.5  
3. Injection Angle θ (°), range: 30° – 45°  
4. Length-to-Diameter Ratio (L/D), range: 2.8 – 3.6  
5. Outlet Slot Length-to-Width Ratio (s/l), range: 3 – 9  

Example input vector for Case 1:  
```python
a1 = [0.5, 0.5, 0, 0.5, 0]

## Pre-trained Models
The folder `trained_models/` contains the pre-trained weights for four surrogate models:  

- `NEW_收敛孔_二维_展向约束_从xd1开始.pkl`, `NEW_收敛孔_二维_直接预测_从xd1开始_3000.pkl`, `NEW_收敛孔_二维_直接预测_从xd1开始_8000.pkl`, `NEW_收敛孔_二维_直接预测_从xd1开始_10000.pkl.pkl`  
- Users can directly load these `.pkl` files when running the prediction scripts without retraining.  

Example (Direct BPNN prediction with pre-trained weights):  
```python
import pickle
model = pickle.load(open("trained_models/NEW_收敛孔_二维_直接预测从xd1开始_10000.pkl","rb"))
