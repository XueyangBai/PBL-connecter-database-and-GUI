#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
import random

# 加载模型和标准化器
model_path = "catboost_model.cbm"
scaler_path = "scaler.pkl"
data_path = "Random parameter list.csv"

best_model = CatBoostRegressor()
best_model.load_model(model_path)
scaler = joblib.load(scaler_path)

data = pd.read_csv(data_path)

tk_feature_names = ['n', 'D', 'A_r', 'A_B', 'A_F', 'α', 'β', 'f_c', 'f_y']

unit_dict = {
    'n': '',
    'D': 'mm',
    'A_r': 'mm²',
    'A_B': 'mm²',
    'A_F': 'mm²',
    'α': '',
    'β': '',
    'f_c': 'MPa',
    'f_y': 'MPa'
}

max_values = {
    "n": 5,
    "D": 90,
    "A_r": 2000,
    "A_B": 5000,
    "A_F": 180000,
    "f_c": 150,
    "f_y": 500
}

def fill_random_values():
    random_row = data.sample(n=1).iloc[0]
    for i, entry in enumerate(entries):
        entry.delete(0, tk.END)
        entry.insert(0, str(random_row[tk_feature_names[i]]))

def validate_input():
    try:
        user_input = [float(entry.get()) for entry in entries]
        n_value = user_input[tk_feature_names.index("n")]
        
        if not n_value.is_integer():
            raise ValueError("n 必须为整数！")
        
        D_value = user_input[tk_feature_names.index("D")]
        alpha_value = user_input[tk_feature_names.index("α")]
        beta_value = user_input[tk_feature_names.index("β")]
        
        if n_value == 0 and D_value != 0:
            raise ValueError("当 n 为 0 时，D 必须为 0！")
        if alpha_value not in [0, 1] or beta_value not in [0, 1]:
            raise ValueError("α 和 β 只能输入 0 或 1！")
        if n_value < 0 or any(val < 0 for val in user_input):
            raise ValueError("所有输入参数不能为负数！")

        for feature, max_val in max_values.items():
            feature_index = tk_feature_names.index(feature)
            if user_input[feature_index] > max_val:
                raise ValueError(f"{feature} 不能超过 {max_val}！")
                
        return user_input
    except Exception as e:
        messagebox.showerror("输入错误", str(e))
        return None

def predict():
    user_input = validate_input()
    if user_input is not None:
        user_input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(user_input_array)
        model_prediction = best_model.predict(scaled_input)[0]

        n, D, A_R, A_B, A_F, alpha, beta, f_c, f_y = user_input
        formula_prediction = (
            (1.56 * n * np.pi * D**2 * np.sqrt(f_c) +
             (0.536 * A_R * f_y - 4e-7 * (A_R * f_y)**2) +
             1.095 * alpha * A_B * f_c +
             0.135 * beta * A_F * np.sqrt(f_c) +
             192566.3) / 1000
        )

        result_label.config(text=f"Vu_pred (Model): {model_prediction:.4f} kN\nVu_pred (Formula): {formula_prediction:.4f} kN", foreground="blue")

root = tk.Tk()
root.title("PBL Connectors Shear Capacity Prediction")
root.geometry("1200x850")
root.resizable(False, False)

biaoti_img = Image.open("title.png").resize((980, 110))
biaoti_photo = ImageTk.PhotoImage(biaoti_img)
tk.Label(root, image=biaoti_photo).pack()

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

left_frame = tk.Frame(frame)
left_frame.pack(side="left", padx=20, pady=20)

tk.Label(left_frame, text="Input parameters:", font=("Arial", 20, "bold")).pack()

random_button = ttk.Button(left_frame, text="Random parameters", command=fill_random_values)
random_button.pack(pady=10)

entries = []
for feature in tk_feature_names:
    row_frame = tk.Frame(left_frame)
    row_frame.pack(fill="x", pady=10)
    
    unit = unit_dict[feature]
    label_text = feature if unit == '' else f"{feature} ({unit})"
    label = tk.Label(row_frame, text=label_text, width=8, font=("Times New Roman", 14, "italic"), anchor="w")
    label.pack(side="left", padx=10)
    
    entry = ttk.Entry(row_frame, font=("Arial", 10), width=10)
    entry.pack(side="left", fill="x", expand=True, padx=3)
    
    entries.append(entry)

fill_random_values()

predict_button = ttk.Button(left_frame, text="predict", command=predict)
predict_button.pack(pady=10)

result_label = tk.Label(left_frame, text="", font=("Arial", 12))
result_label.pack()

shiyi_img = Image.open("parameter meanings.png").resize((870, 650))
shiyi_photo = ImageTk.PhotoImage(shiyi_img)
tk.Label(frame, image=shiyi_photo).pack(side="right", padx=20)

footer_label = tk.Label(root, text="This GUI is developed by college of Civil Engineering of Nanjing Forestry University", 
                        font=("Helvetica", 14), fg="darkblue")
footer_label.pack(side="bottom", pady=5)

root.mainloop()

