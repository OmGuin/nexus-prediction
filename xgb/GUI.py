import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from feature_importance import getGraph
#from optimize import optimize_score
from config import Config

model = joblib.load('xgb/trained_xgb.pkl')  

FEATURES = Config.FEATURES
FORMAL_FEATURES = Config.FORMAL_FEATURES
canvas = None

bg_color = "#ede7f6"        # light purple background
fg_color = "#4a148c"        # deep purple text
entry_bg = "#f3e5f5"        # entry background
button_bg = "#7e57c2"       # medium purple
button_fg = "#ffffff"       # white text


def display_shap_plot(input):
    global canvas
    shap_fig = getGraph(input)
    try:
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(shap_fig, master = right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", f"Error displaying SHAP plot: {e}")



def predict_irscore():
    try:
        input_data = [0]
        
        for field in FEATURES:
            value = entries[field].get()
            if value == '':
                raise ValueError(f"Missing value for {field}")
            input_data.append(float(value))


        full_features = Config.FULL_FEATURES

        df = pd.DataFrame([input_data], columns = full_features)
        prediction = model.predict(df)[0]
        result_var.set(f"Predicted IR Score:\n {prediction:.2f}")
        display_shap_plot(input_data)
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong: {e}")


root = tk.Tk()
root.configure(bg="#ede7f6")
root.title("Insulin Resistance Score Predictor")
root.geometry("1300x600")

style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background=bg_color)
style.configure("TLabel", font=("Segoe UI", 16), background=bg_color, foreground=fg_color)
style.configure("TButton", font=("Segoe UI", 16), background=button_bg, foreground=button_fg)
style.map("TButton", background=[("active", "#9575cd")])
style.configure("TEntry", font=("Segoe UI", 16))


left_frame = ttk.Frame(root, padding=20)
left_frame.grid(row=0, column=0, sticky="nsew")

entries = {}

for i, feature in enumerate(FORMAL_FEATURES):
    ttk.Label(left_frame, text=feature + ":").grid(row=i, column=0, sticky="e", pady=5)
    entry = ttk.Entry(left_frame, width=10, font=("Segoe UI", 14))
    entry.grid(row=i, column=1, pady=5, padx=10)
    entries[FEATURES[i]] = entry

# Predict Button and Result
ttk.Button(left_frame, text="Predict IR Score", command=predict_irscore).grid(row=len(FEATURES), column=0, columnspan=2, pady=15)
result_var = tk.StringVar(value="Predicted IR Score \nwill appear here.")
ttk.Label(left_frame, textvariable=result_var, foreground="#333", font=("Segoe UI", 20, "bold")).grid(row=len(FEATURES)+1, column=0, columnspan=2, pady=20)


right_frame = ttk.Frame(root, padding=20)
right_frame.grid(row=0, column=1, sticky="nsew")


root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
fig = plt.Figure(figsize=(8, 5.5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack()
canvas.draw()

def on_closing():
    root.quit() 
    root.destroy() 

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()