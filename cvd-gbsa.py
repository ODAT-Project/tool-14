#Developed by ODAT project
#please see https://odat.info
#please see https://github.com/ODAT-Project
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#here i removed permutation_importance and concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index #for Cox C-index
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import gc
import traceback

STYLE_CONFIG = {
    "font_family": "Segoe UI", "font_size_normal": 10, "font_size_header": 14,
    "font_size_section": 12, "bg_root": "#F0F0F0", "bg_widget": "#FFFFFF",
    "bg_entry": "#FFFFFF", "fg_text": "#333333", "fg_header": "#000000",
    "accent_color": "#0078D4", "accent_text_color": "#FFFFFF", "border_color": "#CCCCCC",
    "listbox_select_bg": "#0078D4", "listbox_select_fg": "#FFFFFF",
    "disabled_bg": "#E0E0E0", "disabled_fg": "#A0A0A0", "error_text_color": "#D32F2F",
}

class DynamicCVDApp:
    GBSA_RISK_SCORE_COL_NAME = 'GBSA_Risk_Score_Covariate'

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Dynamic CVD Risk Predictor -- GBSA-CPH Enhanced")
        self.root.geometry("1450x980")
        self.root.configure(bg=STYLE_CONFIG["bg_root"])

        self.data_df = None
        self.gbsa_model = None
        self.cph_model = None
        self.scaler_gbsa = None
        self.scaler_cph_linear = None
        self.num_imputer_gbsa = None
        self.num_imputer_cph_linear = None

        self.trained_gbsa_feature_names = []
        self.trained_cph_linear_feature_names = []
        self.all_base_features_for_input = []

        self.trained_feature_medians_gbsa = {}
        self.trained_feature_medians_cph_linear = {}
        self.scaled_columns_gbsa = []
        self.scaled_columns_cph_linear = []

        #storing processed training data
        self.X_gbsa_processed_train = None #useful for understanding what data model was trained on
        self.y_survival_train = None

        self.target_event_col_var = tk.StringVar()
        self.time_horizon_var = tk.StringVar(value="5")
        self.time_to_event_col_var = tk.StringVar()

        self.n_estimators_var = tk.StringVar(value="100")
        self.max_depth_var = tk.StringVar(value="3")
        self.learning_rate_var = tk.StringVar(value="0.1")
        self.cph_penalizer_var = tk.StringVar(value="0.1")

        self.y_test_full_df_for_metrics = None
        
        self.prediction_input_widgets = {}
        self.dynamic_input_scrollable_frame = None
        self.more_plots_window = None

        self.setup_styles()
        self.create_menu()
        self.create_main_layout()
        self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False)

    def setup_styles(self):
        s = ttk.Style(self.root)
        s.theme_use("default")
        
        font_normal = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"])
        font_bold = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"], "bold")
        font_header = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_header"], "bold")
        font_section = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_section"], "bold")
        
        s.configure(".", font=font_normal, background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"], bordercolor=STYLE_CONFIG["border_color"], lightcolor=STYLE_CONFIG["bg_widget"], darkcolor=STYLE_CONFIG["bg_widget"])
        s.configure("TFrame", background=STYLE_CONFIG["bg_root"])
        s.configure("Content.TFrame", background=STYLE_CONFIG["bg_widget"])
        s.configure("TLabel", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"])
        s.configure("Header.TLabel", font=font_header, foreground=STYLE_CONFIG["fg_header"], background=STYLE_CONFIG["bg_root"])
        s.configure("Section.TLabel", font=font_section, foreground=STYLE_CONFIG["fg_header"], background=STYLE_CONFIG["bg_widget"])
        s.configure("TButton", font=font_bold, padding=6, background=STYLE_CONFIG["accent_color"], foreground=STYLE_CONFIG["accent_text_color"])
        
        s.map("TButton", background=[("active", STYLE_CONFIG["accent_color"]), ("disabled", STYLE_CONFIG["disabled_bg"])], foreground=[("active", STYLE_CONFIG["accent_text_color"]), ("disabled", STYLE_CONFIG["disabled_fg"])])
        s.configure("TEntry", fieldbackground=STYLE_CONFIG["bg_entry"], foreground=STYLE_CONFIG["fg_text"], insertcolor=STYLE_CONFIG["fg_text"])
        s.configure("TCombobox", fieldbackground=STYLE_CONFIG["bg_entry"], foreground=STYLE_CONFIG["fg_text"], selectbackground=STYLE_CONFIG["bg_entry"], selectforeground=STYLE_CONFIG["fg_text"], arrowcolor=STYLE_CONFIG["fg_text"])
        
        self.root.option_add('*TCombobox*Listbox.background', STYLE_CONFIG["bg_entry"])
        self.root.option_add('*TCombobox*Listbox.foreground', STYLE_CONFIG["fg_text"])
        self.root.option_add('*TCombobox*Listbox.selectBackground', STYLE_CONFIG["listbox_select_bg"])
        self.root.option_add('*TCombobox*Listbox.selectForeground', STYLE_CONFIG["listbox_select_fg"])
        
        s.configure("TScrollbar", background=STYLE_CONFIG["bg_widget"], troughcolor=STYLE_CONFIG["bg_root"], arrowcolor=STYLE_CONFIG["fg_text"])
        s.configure("TCheckbutton", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"])
        
        s.map("TCheckbutton", indicatorcolor=[("selected", STYLE_CONFIG["accent_color"]), ("!selected", STYLE_CONFIG["border_color"])])
        
        s.configure("TPanedwindow", background=STYLE_CONFIG["bg_root"])
        s.configure("TLabelFrame", background=STYLE_CONFIG["bg_widget"], bordercolor=STYLE_CONFIG["border_color"])
        s.configure("TLabelFrame.Label", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_header"], font=font_section)

    def create_menu(self):
        menubar = tk.Menu(self.root, bg=STYLE_CONFIG["bg_widget"], fg=STYLE_CONFIG["fg_text"], activebackground=STYLE_CONFIG["accent_color"], activeforeground=STYLE_CONFIG["accent_text_color"])
        
        file_menu = tk.Menu(menubar, tearoff=0, bg=STYLE_CONFIG["bg_widget"], fg=STYLE_CONFIG["fg_text"], activebackground=STYLE_CONFIG["accent_color"], activeforeground=STYLE_CONFIG["accent_text_color"])
        file_menu.add_command(label="Load CSV...", command=self.load_csv_file, accelerator="Ctrl+O")
        file_menu.add_separator(); file_menu.add_command(label="About", command=self.show_about_dialog)
        file_menu.add_separator(); file_menu.add_command(label="Quit", command=self.root.quit, accelerator="Ctrl+Q")
        
        menubar.add_cascade(label="File", menu=file_menu)
        
        self.root.config(menu=menubar)
        self.root.bind_all("<Control-o>", lambda e: self.load_csv_file())
        self.root.bind_all("<Control-q>", lambda e: self.root.quit())

    def create_main_layout(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        train_config_pane = ttk.Frame(main_pane, padding="10", style="Content.TFrame")
        main_pane.add(train_config_pane, weight=1)
        
        predict_results_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(predict_results_pane, weight=1)
        
        self.prediction_input_outer_frame = ttk.Frame(predict_results_pane, padding="10", style="Content.TFrame")
        
        predict_results_pane.add(self.prediction_input_outer_frame, weight=2)
        results_display_frame = ttk.Frame(predict_results_pane, padding="10", style="Content.TFrame")
        predict_results_pane.add(results_display_frame, weight=3)
        
        self.create_train_config_widgets(train_config_pane)
        self.create_dynamic_prediction_inputs_placeholder(self.prediction_input_outer_frame)
        self.create_results_display_widgets(results_display_frame)

    def log_training_message(self, message, is_error=False):
        if not hasattr(self, 'training_log_text') or not self.training_log_text.winfo_exists():
            print(f"LOG {'(Error)' if is_error else ''}: {message}"); return
        try:
            self.training_log_text.configure(state=tk.NORMAL)
            
            tag = "error_tag" if is_error else "normal_tag"
            
            self.training_log_text.tag_configure("error_tag", foreground=STYLE_CONFIG["error_text_color"])
            self.training_log_text.tag_configure("normal_tag", foreground=STYLE_CONFIG["fg_text"])
            self.training_log_text.insert(tk.END, message + "\n", tag)
            
            self.training_log_text.see(tk.END)
            self.training_log_text.configure(state=tk.DISABLED)
            self.root.update_idletasks()
        except tk.TclError: print(f"LOG (TCL Error) {'(Error)' if is_error else ''}: {message}")

    def create_train_config_widgets(self, parent_frame):
        ttk.Label(parent_frame, text="Model Training Configuration", style="Header.TLabel", background=STYLE_CONFIG["bg_widget"]).pack(pady=(0,10), anchor=tk.W)
        load_button = ttk.Button(parent_frame, text="Load CSV File", command=self.load_csv_file)
        load_button.pack(pady=5, fill=tk.X)
        
        self.loaded_file_label = ttk.Label(parent_frame, text="No file loaded.")
        self.loaded_file_label.pack(pady=(2,5), anchor=tk.W)

        target_config_frame = ttk.LabelFrame(parent_frame, text="Target Variable & Time")
        target_config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_config_frame, text="Event Column (1=event, 0=censor):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.target_event_selector = ttk.Combobox(target_config_frame, textvariable=self.target_event_col_var, state="readonly", width=28, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.target_event_selector.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        
        ttk.Label(target_config_frame, text="Time to Event/Censor Column (days):").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.time_to_event_selector = ttk.Combobox(target_config_frame, textvariable=self.time_to_event_col_var, state="readonly", width=28, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.time_to_event_selector.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        
        ttk.Label(target_config_frame, text="Prediction Horizon (Years, for results):").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.time_horizon_entry = ttk.Entry(target_config_frame, textvariable=self.time_horizon_var, width=10, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.time_horizon_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)
        
        target_config_frame.columnconfigure(1, weight=1)

        gbsa_fs_frame = ttk.LabelFrame(parent_frame, text="Features for GBSA Risk Score Model")
        gbsa_fs_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(gbsa_fs_frame, text="Select features for non-linear risk score (GBSA):").pack(anchor=tk.W, padx=5, pady=(5,0))
        
        gbsa_listbox_container = ttk.Frame(gbsa_fs_frame, style="Content.TFrame")
        gbsa_listbox_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.gbsa_feature_listbox = tk.Listbox(gbsa_listbox_container, selectmode=tk.MULTIPLE, exportselection=False, height=6, bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"], selectbackground=STYLE_CONFIG["listbox_select_bg"], selectforeground=STYLE_CONFIG["listbox_select_fg"], highlightthickness=1, highlightbackground=STYLE_CONFIG["border_color"], font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        gbsa_feature_listbox_scrollbar = ttk.Scrollbar(gbsa_listbox_container, orient=tk.VERTICAL, command=self.gbsa_feature_listbox.yview)
        self.gbsa_feature_listbox.configure(yscrollcommand=gbsa_feature_listbox_scrollbar.set)
        
        gbsa_feature_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gbsa_feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        gbsa_params_frame = ttk.LabelFrame(parent_frame, text="Risk Score Model (GBSA) Hyperparameters")
        gbsa_params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(gbsa_params_frame, text="Estimators:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.n_estimators_entry = ttk.Entry(gbsa_params_frame, textvariable=self.n_estimators_var, width=8, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.n_estimators_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)
        
        ttk.Label(gbsa_params_frame, text="Max Depth:").grid(row=0, column=2, padx=5, pady=3, sticky=tk.W)
        self.max_depth_entry = ttk.Entry(gbsa_params_frame, textvariable=self.max_depth_var, width=8, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.max_depth_entry.grid(row=0, column=3, padx=5, pady=3, sticky=tk.W)

        ttk.Label(gbsa_params_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.learning_rate_entry = ttk.Entry(gbsa_params_frame, textvariable=self.learning_rate_var, width=8, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.learning_rate_entry.grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)


        cph_fs_frame = ttk.LabelFrame(parent_frame, text="Features for Linear Part of CPH Model")
        
        cph_fs_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(cph_fs_frame, text="Select features for CPH linear terms:").pack(anchor=tk.W, padx=5, pady=(5,0))
        
        cph_listbox_container = ttk.Frame(cph_fs_frame, style="Content.TFrame")
        cph_listbox_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cph_linear_feature_listbox = tk.Listbox(cph_listbox_container, selectmode=tk.MULTIPLE, exportselection=False, height=5, bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"], selectbackground=STYLE_CONFIG["listbox_select_bg"], selectforeground=STYLE_CONFIG["listbox_select_fg"], highlightthickness=1, highlightbackground=STYLE_CONFIG["border_color"], font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        cph_linear_feature_listbox_scrollbar = ttk.Scrollbar(cph_listbox_container, orient=tk.VERTICAL, command=self.cph_linear_feature_listbox.yview)
        
        self.cph_linear_feature_listbox.configure(yscrollcommand=cph_linear_feature_listbox_scrollbar.set)
        
        cph_linear_feature_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.cph_linear_feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cph_params_frame = ttk.LabelFrame(parent_frame, text="CPH Model Hyperparameters")
        cph_params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(cph_params_frame, text="L2 Penalizer:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.cph_penalizer_entry = ttk.Entry(cph_params_frame, textvariable=self.cph_penalizer_var, width=8, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.cph_penalizer_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)

        self.train_button = ttk.Button(parent_frame, text="Train Hybrid GBSA-CPH Model", command=self.train_model_action)
        self.train_button.pack(pady=(10,5), fill=tk.X)

        ttk.Label(parent_frame, text="Training Log & Report:", style="Section.TLabel").pack(anchor=tk.W, pady=(10,0))
        
        self.training_log_text = scrolledtext.ScrolledText(parent_frame, height=5, wrap=tk.WORD, bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"], insertbackground=STYLE_CONFIG["fg_text"], font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]-1), highlightthickness=1, highlightbackground=STYLE_CONFIG["border_color"])
        self.training_log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.training_log_text.configure(state=tk.DISABLED)

    def create_dynamic_prediction_inputs_placeholder(self, parent_frame):
        ttk.Label(parent_frame, text="Patient Data for Prediction", style="Header.TLabel", background=STYLE_CONFIG["bg_widget"]).pack(pady=(0,10), anchor=tk.W)
        
        self.dynamic_input_canvas = tk.Canvas(parent_frame, borderwidth=0, background=STYLE_CONFIG["bg_widget"], highlightthickness=0)
        
        vsb = ttk.Scrollbar(parent_frame, orient="vertical", command=self.dynamic_input_canvas.yview)
        
        self.dynamic_input_canvas.configure(yscrollcommand=vsb.set); vsb.pack(side="right", fill="y")
        self.dynamic_input_canvas.pack(side="left", fill="both", expand=True)
        
        self.dynamic_input_scrollable_frame = ttk.Frame(self.dynamic_input_canvas, style="Content.TFrame")
        self.dynamic_input_canvas.create_window((0, 0), window=self.dynamic_input_scrollable_frame, anchor="nw")
        
        self.dynamic_input_scrollable_frame.bind("<Configure>", lambda e: self.dynamic_input_canvas.configure(scrollregion=self.dynamic_input_canvas.bbox("all")))
        self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="Train a model to enable prediction inputs.", style="TLabel")
        
        self.placeholder_pred_label.pack(padx=10, pady=20)
        
        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (CPH)", command=self.assess_risk_action)
        self.assess_button.pack_forget()

    def create_dynamic_prediction_inputs(self):
        if self.dynamic_input_scrollable_frame:
            for widget in self.dynamic_input_scrollable_frame.winfo_children(): widget.destroy()
        self.prediction_input_widgets = {}
        self.all_base_features_for_input = sorted(list(set(self.trained_gbsa_feature_names + self.trained_cph_linear_feature_names)))
        if not self.all_base_features_for_input:
            self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="No features available. Train model.", style="TLabel")
            self.placeholder_pred_label.pack(padx=10, pady=20)
            if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.pack_forget()
            return
        for feature_name in self.all_base_features_for_input:
            row_frame = ttk.Frame(self.dynamic_input_scrollable_frame, style="Content.TFrame")
            
            row_frame.pack(fill=tk.X, pady=1, padx=2)
            
            display_name = feature_name if len(feature_name) < 35 else feature_name[:32] + "..."
            
            label = ttk.Label(row_frame, text=f"{display_name}:", width=35, anchor="w"); label.pack(side=tk.LEFT, padx=(0,5))
            
            entry = ttk.Entry(row_frame, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
            
            default_val = "0"
            
            if feature_name in self.trained_feature_medians_gbsa: default_val = self.trained_feature_medians_gbsa.get(feature_name, "0")
            
            elif feature_name in self.trained_feature_medians_cph_linear: default_val = self.trained_feature_medians_cph_linear.get(feature_name, "0")
            
            entry.insert(0, str(default_val)); entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
            
            self.prediction_input_widgets[feature_name] = entry
        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (CPH)", command=self.assess_risk_action)
        
        self.assess_button.pack(pady=(15,10), fill=tk.X, padx=5)
        
        self.dynamic_input_scrollable_frame.update_idletasks()
        
        self.dynamic_input_canvas.config(scrollregion=self.dynamic_input_canvas.bbox("all"))

    def create_results_display_widgets(self, parent_frame):
        top_frame = ttk.Frame(parent_frame, style="Content.TFrame"); top_frame.pack(fill=tk.X, pady=5)
        
        pred_res_frame = ttk.LabelFrame(top_frame, text="CPH Prediction Result")
        pred_res_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        self.risk_prob_label = ttk.Label(pred_res_frame, text="N/A", font=(STYLE_CONFIG["font_family"], 22, "bold"), foreground=STYLE_CONFIG["accent_color"])
        self.risk_prob_label.pack(pady=(5,2))
        
        self.risk_interpretation_label = ttk.Label(pred_res_frame, text="Train model & assess.", font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        self.risk_interpretation_label.pack(pady=(0,5))
        
        self.more_plots_button = ttk.Button(top_frame, text="View Survival Plots", command=self.show_more_plots_window, state=tk.DISABLED)
        self.more_plots_button.pack(side=tk.RIGHT, padx=(5,0), pady=10, anchor="ne")
        
        plot_frame = ttk.LabelFrame(parent_frame, text="Model Performance Visuals"); plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.fig = plt.Figure(figsize=(8, 7), dpi=90, facecolor=STYLE_CONFIG["bg_widget"]); self.fig.subplots_adjust(hspace=0.6, wspace=0.4)
        self.ax_importance_gbsa = self.fig.add_subplot(2, 2, 1); self.ax_cph_coeffs = self.fig.add_subplot(2, 2, 2)
        self.ax_calibration_like = self.fig.add_subplot(2, 2, 3); self.ax_survival_curve = self.fig.add_subplot(2, 2, 4) 
        
        for ax in [self.ax_importance_gbsa, self.ax_cph_coeffs, self.ax_calibration_like, self.ax_survival_curve]:
            ax.tick_params(colors=STYLE_CONFIG["fg_text"]); ax.xaxis.label.set_color(STYLE_CONFIG["fg_text"])
            ax.yaxis.label.set_color(STYLE_CONFIG["fg_text"]); ax.title.set_color(STYLE_CONFIG["fg_header"])
            ax.set_facecolor(STYLE_CONFIG["bg_entry"])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame); self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(bg=STYLE_CONFIG["bg_widget"]); self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.update_plots(clear_only=True)

    def toggle_train_predict_sections_enabled(self, data_loaded=False, model_trained=False):
        train_state = tk.NORMAL if data_loaded else tk.DISABLED
        predict_state = tk.NORMAL if model_trained else tk.DISABLED
        
        if hasattr(self, 'train_button'): self.train_button.config(state=train_state)
        
        if hasattr(self, 'gbsa_feature_listbox'): self.gbsa_feature_listbox.config(state=train_state)
        
        if hasattr(self, 'cph_linear_feature_listbox'): self.cph_linear_feature_listbox.config(state=train_state)
        
        if hasattr(self, 'target_event_selector'): self.target_event_selector.config(state="readonly" if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'time_to_event_selector'): self.time_to_event_selector.config(state="readonly" if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'time_horizon_entry'): self.time_horizon_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'n_estimators_entry'): self.n_estimators_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'max_depth_entry'): self.max_depth_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        if hasattr(self, 'learning_rate_entry'): self.learning_rate_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'cph_penalizer_entry'): self.cph_penalizer_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.config(state=predict_state)
        
        if hasattr(self, 'more_plots_button'): self.more_plots_button.config(state=predict_state)
        
        for _feature_name, widget in self.prediction_input_widgets.items():
            if hasattr(widget, 'config'): widget.config(state=tk.NORMAL if model_trained else tk.DISABLED)

    def downcast_numerics(self, df):
        self.log_training_message("  Attempting to downcast numeric types for memory optimization...")
        
        f_cols = df.select_dtypes('float').columns; i_cols = df.select_dtypes('integer').columns
        
        df[f_cols] = df[f_cols].apply(pd.to_numeric, downcast='float')
        df[i_cols] = df[i_cols].apply(pd.to_numeric, downcast='integer')
        
        gc.collect(); return df

    def _populate_ui_lists_after_load(self, column_names):
        self.gbsa_feature_listbox.delete(0, tk.END); self.cph_linear_feature_listbox.delete(0, tk.END)
        
        for col_name in column_names: 
            self.gbsa_feature_listbox.insert(tk.END, col_name)
            self.cph_linear_feature_listbox.insert(tk.END, col_name)
        
        self.target_event_selector['values'] = column_names; self.time_to_event_selector['values'] = column_names
        default_target = 'Cardiovascular_mortality'; default_time_col = 'Time_to_CVD_mortality_days'
        
        if default_target in column_names: self.target_event_col_var.set(default_target)
        
        elif column_names: self.target_event_col_var.set(column_names[0])
        
        if default_time_col in column_names: self.time_to_event_col_var.set(default_time_col)
        
        elif 'Time_to_mortality_days' in column_names: self.time_to_event_col_var.set('Time_to_mortality_days')
        
        elif column_names and len(column_names) > 1: self.time_to_event_col_var.set(column_names[1])
        
        self.log_training_message(f"  UI lists populated with {len(column_names)} columns."); self.root.update_idletasks()

    def load_csv_file(self):
        filepath = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filepath: self.log_training_message("File loading cancelled by user."); return
        try:
            self.data_df = pd.read_csv(filepath, low_memory=False)
            
            self.data_df = self.downcast_numerics(self.data_df.copy()); gc.collect()
            
            self.loaded_file_label.config(text=f"Loaded: {filepath.split('/')[-1]} ({self.data_df.shape[0]} rows, {self.data_df.shape[1]} cols)")
            
            self.log_training_message(f"Successfully loaded and downcasted {filepath.split('/')[-1]}.")
            
            self.log_training_message(f"  Shape: {self.data_df.shape}. Columns found: {len(self.data_df.columns)}")
            
            column_names = sorted([str(col) for col in self.data_df.columns if str(col).strip()])
            
            if not column_names:
                self.log_training_message("No columns found in CSV or header is missing.", is_error=True)
                messagebox.showerror("CSV Error", "No columns detected in CSV.")
                self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False); return
            self.root.after(10, self._populate_ui_lists_after_load, column_names)
            
            self.gbsa_model = None; self.cph_model = None; self.X_gbsa_processed_train = None; self.y_survival_train = None
            
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=False)
            
            if self.dynamic_input_scrollable_frame:
                for widget in self.dynamic_input_scrollable_frame.winfo_children(): widget.destroy()
            
            self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="Train a model to enable prediction inputs.", style="TLabel")
            
            self.placeholder_pred_label.pack(padx=10, pady=20)
            
            if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.pack_forget()
            self.update_plots(clear_only=True)
            
            self.risk_interpretation_label.config(text="Data loaded. Configure and train model."); self.risk_prob_label.config(text="N/A")
        except Exception as e:
            self.log_training_message(f"Error loading CSV: {str(e)}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            
            messagebox.showerror("Error Loading CSV", f"Failed to load or parse CSV file.\nError: {e}")
            
            self.data_df = None; self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False)

    def _preprocess_features(self, df_subset, feature_names, stored_medians, scaler_to_use, scaled_cols_list, imputer_to_use, fit_transform=True):
        processed_df = df_subset[feature_names].copy()
        for col in processed_df.columns: processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        if fit_transform:
            for col in processed_df.columns: stored_medians[col] = processed_df[col].median()
            imputer_to_use.fit(processed_df)
        imputed_values = imputer_to_use.transform(processed_df)
        processed_df = pd.DataFrame(imputed_values, columns=processed_df.columns, index=processed_df.index)
        if fit_transform:
            scaled_cols_list.clear()
            for col in processed_df.columns:
                if processed_df[col].dtype != np.float64:
                    try:
                        numeric_col = pd.to_numeric(processed_df[col], errors='coerce')
                        processed_df[col] = numeric_col.astype(np.float64) if not numeric_col.isnull().all() else numeric_col.astype(np.float64)
                    except Exception: continue 
                if pd.api.types.is_numeric_dtype(processed_df[col]) and processed_df[col].nunique(dropna=False) > 2:
                    scaled_cols_list.append(col)
        if scaled_cols_list:
            valid_scaled_cols = []
            for col_to_scale in scaled_cols_list:
                if col_to_scale in processed_df.columns:
                    if processed_df[col_to_scale].dtype != np.float64:
                        try:
                            processed_df[col_to_scale] = processed_df[col_to_scale].astype(np.float64)
                            valid_scaled_cols.append(col_to_scale)
                        except ValueError: self.log_training_message(f"Warning: Column {col_to_scale} could not be cast to float64 for scaling. Skipping.", is_error=True)
                    else: valid_scaled_cols.append(col_to_scale)
            if valid_scaled_cols: 
                df_to_scale = processed_df[valid_scaled_cols]
                if not df_to_scale.empty: 
                    if fit_transform: scaler_to_use.fit(df_to_scale)
                    scaled_values = scaler_to_use.transform(df_to_scale)
                    processed_df.loc[:, valid_scaled_cols] = scaled_values
            elif scaled_cols_list: self.log_training_message("Warning: No valid float64 columns found for scaling after dtype checks.", is_error=True)
        return processed_df

    def train_model_action(self):
        if self.data_df is None: messagebox.showerror("Error", "No data loaded."); return
        selected_gbsa_indices = self.gbsa_feature_listbox.curselection()
        
        selected_cph_linear_indices = self.cph_linear_feature_listbox.curselection()
        
        target_event_col = self.target_event_col_var.get()
        
        time_to_event_col = self.time_to_event_col_var.get()

        if not selected_gbsa_indices: messagebox.showerror("Error", "No features selected for GBSA Model. Please select at least one."); return
        
        if not target_event_col or not time_to_event_col: messagebox.showerror("Error", "Target event and time-to-event columns must be selected."); return
        
        self.trained_gbsa_feature_names = [self.gbsa_feature_listbox.get(i) for i in selected_gbsa_indices]
        
        self.trained_cph_linear_feature_names = [self.cph_linear_feature_listbox.get(i) for i in selected_cph_linear_indices]
        
        for col_list_name, col_list in [("GBSA", self.trained_gbsa_feature_names), ("CPH", self.trained_cph_linear_feature_names)]:
            if target_event_col in col_list or time_to_event_col in col_list:
                messagebox.showerror("Error", f"Target/Time column cannot be in {col_list_name} feature list."); return
        if self.GBSA_RISK_SCORE_COL_NAME in self.trained_cph_linear_feature_names:
             messagebox.showerror("Error", f"'{self.GBSA_RISK_SCORE_COL_NAME}' is reserved."); return
        try:
            n_estimators = int(self.n_estimators_var.get())
            max_depth_str = self.max_depth_var.get()
            max_depth = int(max_depth_str) if max_depth_str.strip() and max_depth_str.lower() != 'none' and max_depth_str.lower() != "" else None
            learning_rate = float(self.learning_rate_var.get())
            cph_penalizer = float(self.cph_penalizer_var.get())
            if n_estimators <= 0 or (max_depth is not None and max_depth <= 0) or learning_rate <=0 or cph_penalizer < 0:
                raise ValueError("Hyperparameters out of valid range.")
        except ValueError: messagebox.showerror("Hyperparameter Error", "Invalid hyperparameter values."); return

        self.log_training_message("--- Starting Hybrid Model Training ---")
        
        self.log_training_message(f"Target Event: '{target_event_col}', Time Column: '{time_to_event_col}'")
        
        self.log_training_message(f"GBSA Features ({len(self.trained_gbsa_feature_names)}): {self.trained_gbsa_feature_names[:5]}...")
        
        self.log_training_message(f"CPH Linear Features ({len(self.trained_cph_linear_feature_names)}): {self.trained_cph_linear_feature_names[:5] if self.trained_cph_linear_feature_names else 'None'}...")
        
        full_df_processed = self.data_df.copy(); gc.collect()
        try:
            full_df_processed[target_event_col] = pd.to_numeric(full_df_processed[target_event_col], errors='raise')
            full_df_processed[time_to_event_col] = pd.to_numeric(full_df_processed[time_to_event_col], errors='raise')
        except Exception as e:
            self.log_training_message(f"Error: Target or Time column non-numeric: {e}", is_error=True); messagebox.showerror("Data Error", f"Target/Time column has non-numeric data: {e}"); return
        
        initial_rows = len(full_df_processed)
        
        full_df_processed.dropna(subset=[target_event_col, time_to_event_col], inplace=True)
        
        full_df_processed = full_df_processed[full_df_processed[time_to_event_col] > 0]
        
        if len(full_df_processed) < initial_rows: self.log_training_message(f"  Dropped {initial_rows - len(full_df_processed)} rows (NaNs/invalid times in target/time).")
        
        if full_df_processed.empty: messagebox.showerror("Data Error", "No valid data after cleaning target/time columns."); return

        self.log_training_message("\n--- Stage 1: Training Risk Score Model (Gradient Boosting Survival Analysis) ---")
        try:
            X_gbsa_full = full_df_processed[self.trained_gbsa_feature_names]
            
            self.scaler_gbsa = StandardScaler(); self.num_imputer_gbsa = SimpleImputer(strategy='median')
            
            self.trained_feature_medians_gbsa = {}; self.scaled_columns_gbsa = []
            
            X_gbsa_processed = self._preprocess_features(X_gbsa_full, self.trained_gbsa_feature_names, self.trained_feature_medians_gbsa, self.scaler_gbsa, self.scaled_columns_gbsa, self.num_imputer_gbsa, fit_transform=True)
            
            self.log_training_message(f"  GBSA processed features: {X_gbsa_processed.shape}")
            
            self.log_training_message(f"  GBSA Scaled columns: {self.scaled_columns_gbsa if self.scaled_columns_gbsa else 'None'}")
            
            y_event_status = full_df_processed[target_event_col].astype(bool)
            
            y_time_to_event = full_df_processed[time_to_event_col].astype(float)
            
            y_survival_structured = Surv.from_arrays(event=y_event_status, time=y_time_to_event)
            
            self.gbsa_model = GradientBoostingSurvivalAnalysis(
                n_estimators=n_estimators, 
                learning_rate=learning_rate, 
                max_depth=max_depth, 
                random_state=42,
                subsample=0.7 
            ) 
            
            self.gbsa_model.fit(X_gbsa_processed, y_survival_structured)
            
            self.log_training_message("  Gradient Boosting Survival Analysis model trained.")
            
            self.X_gbsa_processed_train = X_gbsa_processed.copy(); self.y_survival_train = y_survival_structured
            
            gbsa_risk_scores_all_data = self.gbsa_model.predict(X_gbsa_processed)
            
            full_df_processed[self.GBSA_RISK_SCORE_COL_NAME] = gbsa_risk_scores_all_data
            
            self.log_training_message(f"  '{self.GBSA_RISK_SCORE_COL_NAME}' generated for all {len(gbsa_risk_scores_all_data)} samples.")
            
            if len(gbsa_risk_scores_all_data) > 0:
                self.log_training_message(f"    Sample of '{self.GBSA_RISK_SCORE_COL_NAME}' (first 5): {np.round(gbsa_risk_scores_all_data[:5], 4)}")
                self.log_training_message(f"    Stats for '{self.GBSA_RISK_SCORE_COL_NAME}': Min={np.min(gbsa_risk_scores_all_data):.4f}, Max={np.max(gbsa_risk_scores_all_data):.4f}, Mean={np.mean(gbsa_risk_scores_all_data):.4f}, StdDev={np.std(gbsa_risk_scores_all_data):.4f}")
        except Exception as e:
            self.log_training_message(f"Error in GBSA training: {type(e).__name__}: {str(e)}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            messagebox.showerror("GBSA Training Error", f"Failed GBSA stage ({type(e).__name__}): {e}"); self.gbsa_model = None; return

        self.log_training_message("\n--- Stage 2: Training Cox Proportional Hazards (CPH) Model ---")
        try:
            if self.trained_cph_linear_feature_names:
                X_cph_linear_part = full_df_processed[self.trained_cph_linear_feature_names]
                
                self.scaler_cph_linear = StandardScaler(); self.num_imputer_cph_linear = SimpleImputer(strategy='median')
                
                self.trained_feature_medians_cph_linear = {}; self.scaled_columns_cph_linear = []
                
                X_cph_linear_processed = self._preprocess_features(X_cph_linear_part, self.trained_cph_linear_feature_names, self.trained_feature_medians_cph_linear, self.scaler_cph_linear, self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=True)
                
                self.log_training_message(f"  CPH linear processed features: {X_cph_linear_processed.shape}")
                
                self.log_training_message(f"  CPH linear Scaled columns: {self.scaled_columns_cph_linear if self.scaled_columns_cph_linear else 'None'}")
                
                df_for_cph_fitting = X_cph_linear_processed.copy()
                
                df_for_cph_fitting[self.GBSA_RISK_SCORE_COL_NAME] = full_df_processed[self.GBSA_RISK_SCORE_COL_NAME]
            else: 
                df_for_cph_fitting = pd.DataFrame({self.GBSA_RISK_SCORE_COL_NAME: full_df_processed[self.GBSA_RISK_SCORE_COL_NAME]})
            self.log_training_message(f"  Features for CPH fitter: {df_for_cph_fitting.columns.tolist()}")
            
            df_for_cph_fitting[time_to_event_col] = full_df_processed[time_to_event_col].values
            
            df_for_cph_fitting[target_event_col] = full_df_processed[target_event_col].astype(int).values
            
            train_cph_df, test_cph_df = train_test_split(df_for_cph_fitting, test_size=0.25, random_state=42, stratify=df_for_cph_fitting[target_event_col].astype(int) if df_for_cph_fitting[target_event_col].nunique() > 1 else None)
            
            if train_cph_df.empty or (not test_cph_df.empty and len(test_cph_df) < 5) : 
                 self.log_training_message("Train or test split for CPH is empty or too small.", is_error=True); messagebox.showerror("Data Split Error", "CPH training/test set is empty or too small."); return
            self.cph_model = CoxPHFitter(penalizer=cph_penalizer, l1_ratio=0.0)
            
            self.cph_model.fit(train_cph_df, duration_col=time_to_event_col, event_col=target_event_col)
            
            self.log_training_message("  CPH model trained."); self.log_training_message(f"  CPH Concordance on training: {self.cph_model.concordance_index_:.4f}")
            
            self.log_training_message("\n  Checking Proportional Hazards Assumption (on training data):")
            try:
                ph_test_results = self.cph_model.check_assumptions(train_cph_df, p_value_threshold=0.05, show_plots=False)
                
                self.log_training_message("    --- PH Assumption Test Summary ---")
                
                if isinstance(ph_test_results, pd.DataFrame):
                    if 'p' in ph_test_results.columns and not ph_test_results.empty:
                        for covariate_name_ph in ph_test_results.index:
                            p_value = ph_test_results.loc[covariate_name_ph, 'p']
                            test_name_detail = f" ({ph_test_results.loc[covariate_name_ph, 'test_name']})" if 'test_name' in ph_test_results.columns else ""
                            self.log_training_message(f"        {covariate_name_ph}{test_name_detail}: p={p_value:.4f} {'(Potential Violation)' if p_value < 0.05 else ''}")
                    else:
                        self.log_training_message(f"    PH results DataFrame structure unexpected or empty. Columns: {ph_test_results.columns.tolist() if not ph_test_results.empty else 'Empty DF'}", is_error=True)
                        if not ph_test_results.empty: self.log_training_message(f"    Full PH results (DataFrame):\n{ph_test_results.to_string()}", is_error=True)
                        else: self.log_training_message("    PH results DataFrame is empty.", is_error=True)
                
                elif isinstance(ph_test_results, list):
                    if not ph_test_results: self.log_training_message("    PH assumption check returned an empty list. Test might not be applicable or no violations found to list.", is_error=False)
                    else: self.log_training_message(f"    PH assumption check returned a list. Content (first 500 chars): {str(ph_test_results)[:500]}...", is_error=True)
                
                elif hasattr(ph_test_results, '__str__') and "StatisticalResult" in str(type(ph_test_results)): self.log_training_message(str(ph_test_results))
                
                else: self.log_training_message(f"    PH assumption check did not return DataFrame/List/StatisticalResult. Got: {type(ph_test_results)}", is_error=True)
            
            except Exception as e_ph:
                self.log_training_message(f"    Error during PH assumption check: {type(e_ph).__name__}: {e_ph}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            
            self.y_test_full_df_for_metrics = test_cph_df.copy()
            
            if not self.y_test_full_df_for_metrics.empty:
                cph_model_features_for_pred_cindex = self.cph_model.params_.index.tolist()
                
                test_df_for_cindex_pred = self.y_test_full_df_for_metrics[[col for col in cph_model_features_for_pred_cindex if col in self.y_test_full_df_for_metrics.columns]]
                
                if not test_df_for_cindex_pred.empty and all(col in test_df_for_cindex_pred.columns for col in cph_model_features_for_pred_cindex):
                    cph_test_c_index_val = concordance_index(self.y_test_full_df_for_metrics[time_to_event_col], -self.cph_model.predict_partial_hazard(test_df_for_cindex_pred), self.y_test_full_df_for_metrics[target_event_col])
                    
                    self.log_training_message(f"  CPH Concordance on test set: {cph_test_c_index_val:.4f}")
                else: self.log_training_message(f"  CPH Concordance on test set: Could not compute due to missing columns in test_df for prediction. Needed: {cph_model_features_for_pred_cindex}", is_error=True)
            else: self.log_training_message("  Test set for CPH is empty, cannot calculate test C-index.", is_error=True)
            
            self.generate_training_report(n_estimators, max_depth, learning_rate, cph_penalizer)
            
            self.create_dynamic_prediction_inputs(); self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=True)
            
            self.update_plots()
        except Exception as e:
            error_type = type(e).__name__; error_msg = str(e)
            
            self.log_training_message(f"Error in CPH training stage ({error_type}): {error_msg}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            
            messagebox.showerror("CPH Training Error", f"Failed CPH stage ({error_type}): {error_msg}"); self.cph_model = None
            
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=bool(self.gbsa_model))
        finally:
            if 'full_df_processed' in locals(): del full_df_processed; 
            
            if 'X_gbsa_full' in locals(): del X_gbsa_full
            
            if 'X_gbsa_processed' in locals() and self.X_gbsa_processed_train is not X_gbsa_processed : del X_gbsa_processed
            
            if 'df_for_cph_fitting' in locals(): del df_for_cph_fitting
            
            if 'train_cph_df' in locals(): del train_cph_df
            
            if 'test_cph_df' in locals(): del test_cph_df
            
            gc.collect()

    def generate_training_report(self, n_est, max_d, learn_rate, cph_pen):
        report_lines = []; target_col_name = self.target_event_col_var.get(); time_col_name = self.time_to_event_col_var.get()
        
        report_lines.append("\n--- Combined Model Training Report ---")
        
        report_lines.append(f"Dataset Shape (after initial cleaning): {self.data_df.shape if self.data_df is not None else 'N/A'}")
        
        report_lines.append(f"Target Event Column: {target_col_name}, Time Column: {time_col_name}")
        
        report_lines.append("\nRisk Score Model (Gradient Boosting Survival Analysis):")
        
        report_lines.append(f"  Features used: {len(self.trained_gbsa_feature_names)}")
        
        report_lines.append(f"  Hyperparameters: Estimators={n_est}, MaxDepth={max_d if max_d is not None else 'Unlimited'}, LearningRate={learn_rate}")
        
        report_lines.append("\nCPH Model:")
        
        cph_features_in_model = self.trained_cph_linear_feature_names + [self.GBSA_RISK_SCORE_COL_NAME]
        
        report_lines.append(f"  Features used (incl. GBSA score): {len(cph_features_in_model)}")
        
        report_lines.append(f"  Linear CPH Features: {self.trained_cph_linear_feature_names if self.trained_cph_linear_feature_names else 'None'}")
        
        report_lines.append(f"  CPH L2 Penalizer: {cph_pen}")
        
        if self.cph_model and hasattr(self.cph_model, 'summary') and self.cph_model.summary is not None and not self.cph_model.summary.empty:
            report_lines.append("\nCPH Model Summary (Hazard Ratios):")
            
            summary_df_to_print = self.cph_model.summary.reset_index()
            
            original_index_name = self.cph_model.summary.index.name if self.cph_model.summary.index.name is not None else 'covariate'
            
            if 'index' in summary_df_to_print.columns and original_index_name != 'index': summary_df_to_print.rename(columns={'index': original_index_name}, inplace=True)
            
            elif 'index' in summary_df_to_print.columns and original_index_name == 'index' and 'covariate' not in summary_df_to_print.columns:
                 summary_df_to_print.rename(columns={'index': 'covariate'}, inplace=True); original_index_name = 'covariate'
            summary_cols_expected = [original_index_name, 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
            
            actual_cols_present = [col for col in summary_cols_expected if col in summary_df_to_print.columns]
            
            if original_index_name in actual_cols_present :
                summary_text = summary_df_to_print[actual_cols_present].to_string(index=False); report_lines.extend(summary_text.split('\n'))
            
            else: report_lines.append(f"  Could not format CPH summary. Covariate col '{original_index_name}' or key stats missing. Available: {summary_df_to_print.columns.tolist()}")
        else: report_lines.append("  CPH model summary not available.")
        
        if self.y_test_full_df_for_metrics is not None and not self.y_test_full_df_for_metrics.empty and self.cph_model:
            try:
                cph_model_features_for_pred_report = self.cph_model.params_.index.tolist()
                
                test_df_for_report_pred = self.y_test_full_df_for_metrics[[col for col in cph_model_features_for_pred_report if col in self.y_test_full_df_for_metrics.columns]]
                
                if not test_df_for_report_pred.empty and all(col in test_df_for_report_pred.columns for col in cph_model_features_for_pred_report):
                    cph_test_c_index_val_report = concordance_index(self.y_test_full_df_for_metrics[time_col_name], -self.cph_model.predict_partial_hazard(test_df_for_report_pred), self.y_test_full_df_for_metrics[target_col_name])
                    report_lines.append(f"\nCPH Performance on Test Set:"); report_lines.append(f"  Concordance Index (C-index): {cph_test_c_index_val_report:.4f}")
                else: report_lines.append(f"\nCPH Performance on Test Set: Could not calculate C-index due to missing columns for prediction.")
            except Exception as e: report_lines.append(f"\nCPH Performance on Test Set: Error calculating C-index: {type(e).__name__} - {e}")
        
        report_lines.append("--- End of Report ---"); report_lines.append("-----------------------"); report_lines.append("--- by ODAT project ---")
        
        file_save_status_message = ""
        
        try:
            with open("report.txt", "w", encoding='utf-8') as f_report:
                for line in report_lines: f_report.write(line + "\n")
            file_save_status_message = "\n--- Report content also saved to report.txt ---"
        except Exception as e: file_save_status_message = f"\n--- Error saving report to report.txt: {e} ---"
        
        for line in report_lines: self.log_training_message(line, is_error=False)
        
        is_file_save_error = "Error saving report" in file_save_status_message
        
        self.log_training_message(file_save_status_message, is_error=is_file_save_error)

    def assess_risk_action(self):
        if not self.gbsa_model or not self.cph_model or not self.all_base_features_for_input:
            messagebox.showerror("Error", "Model not fully trained or features unclear."); return
        input_values_from_gui = {}
        try:
            for feature_name, widget in self.prediction_input_widgets.items():
                value_str = widget.get()
                try: input_values_from_gui[feature_name] = float(value_str)
                except ValueError:
                    if value_str.lower() in ["true", "yes", "1", "male"]: input_values_from_gui[feature_name] = 1.0
                    elif value_str.lower() in ["false", "no", "0", "female"]: input_values_from_gui[feature_name] = 0.0
                    else: input_values_from_gui[feature_name] = np.nan 
        except Exception as e: messagebox.showerror("Input Error", f"Error reading input values: {e}"); return
        
        input_df_for_processing = pd.DataFrame([input_values_from_gui], columns=self.all_base_features_for_input)
        
        input_df_gbsa_filtered = input_df_for_processing[self.trained_gbsa_feature_names].copy()
        
        input_gbsa_processed = self._preprocess_features(input_df_gbsa_filtered, self.trained_gbsa_feature_names, self.trained_feature_medians_gbsa, self.scaler_gbsa, self.scaled_columns_gbsa, self.num_imputer_gbsa, fit_transform=False)
        
        try: new_gbsa_risk_score = self.gbsa_model.predict(input_gbsa_processed)[0]
        except Exception as e:
            self.log_training_message(f"GBSA prediction error: {str(e)}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            messagebox.showerror("Prediction Error", f"Could not generate GBSA risk score: {e}"); return
        
        cph_input_dict = {self.GBSA_RISK_SCORE_COL_NAME: new_gbsa_risk_score}
        
        if self.trained_cph_linear_feature_names:
            input_df_cph_linear_filtered = input_df_for_processing[self.trained_cph_linear_feature_names].copy()
            input_cph_linear_processed = self._preprocess_features(input_df_cph_linear_filtered, self.trained_cph_linear_feature_names, self.trained_feature_medians_cph_linear, self.scaler_cph_linear, self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=False)
            for col in input_cph_linear_processed.columns: cph_input_dict[col] = input_cph_linear_processed[col].iloc[0]
        input_data_for_cph_model_df = pd.DataFrame([cph_input_dict])
        
        try: cph_model_cols = self.cph_model.params_.index.tolist()
        except AttributeError: self.log_training_message("CPH model params_ not found for prediction.", is_error=True); messagebox.showerror("Prediction Error", "CPH model structure unclear."); return
        
        for col in cph_model_cols:
            if col not in input_data_for_cph_model_df.columns:
                 input_data_for_cph_model_df[col] = 0.0; self.log_training_message(f"Warning: Covariate '{col}' for CPH prediction missing, imputed as 0.", is_error=True)
        try: input_data_for_cph_model_df = input_data_for_cph_model_df[cph_model_cols]
        
        except KeyError as e_key: self.log_training_message(f"KeyError CPH prediction: {e_key}. Needed: {cph_model_cols}, Got: {input_data_for_cph_model_df.columns.tolist()}", is_error=True); messagebox.showerror("Prediction Error", f"Column mismatch CPH prediction: {e_key}"); return
        
        try:
            time_horizon_str = self.time_horizon_var.get().strip()
            pred_horizon_years = 5 
            if time_horizon_str:
                try: pred_horizon_years = float(time_horizon_str)
                except ValueError: pred_horizon_years = 5 
            if pred_horizon_years <= 0: pred_horizon_years = 5 
            
            pred_horizon_days = pred_horizon_years * 365.25 
            
            survival_prob_at_horizon = self.cph_model.predict_survival_function(input_data_for_cph_model_df, times=[pred_horizon_days]).iloc[0,0]
            
            risk_at_horizon = 1.0 - survival_prob_at_horizon
            
            self.risk_prob_label.config(text=f"{pred_horizon_years:.0f}-Year Risk: {risk_at_horizon*100:.1f}%")
            
            if risk_at_horizon > 0.20: self.risk_interpretation_label.config(text="Higher Risk", foreground=STYLE_CONFIG["error_text_color"])
            
            elif risk_at_horizon > 0.10: self.risk_interpretation_label.config(text="Moderate Risk", foreground="#FFA000") 
            
            else: self.risk_interpretation_label.config(text="Lower Risk", foreground="#388E3C") 
        except Exception as e:
            self.log_training_message(f"CPH prediction error: {str(e)}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            
            messagebox.showerror("Prediction Error", f"Could not make CPH prediction: {e}"); self.risk_prob_label.config(text="Error"); self.risk_interpretation_label.config(text="")

    def update_plots(self, clear_only=False):
        ax_list = [self.ax_importance_gbsa, self.ax_cph_coeffs, self.ax_calibration_like, self.ax_survival_curve]
        
        for ax in ax_list: ax.clear()
        
        title_font_dict = {'color': STYLE_CONFIG["fg_header"], 'fontsize': STYLE_CONFIG["font_size_normal"] + 2, 'weight': 'bold'}
        label_font_dict = {'color': STYLE_CONFIG["fg_text"], 'fontsize': STYLE_CONFIG["font_size_normal"]}
        tick_color = STYLE_CONFIG["fg_text"]; bar_color = STYLE_CONFIG["accent_color"]

        #GBSA feature importance plot
        self.ax_importance_gbsa.set_title("GBSA: Feature Importances", fontdict=title_font_dict)
        
        if clear_only or not self.gbsa_model:
            self.ax_importance_gbsa.text(0.5, 0.5, "Train GBSA model first.", ha='center', va='center', color=tick_color)
        elif not self.trained_gbsa_feature_names:
            self.ax_importance_gbsa.text(0.5, 0.5, "No features were selected for GBSA training.", ha='center', va='center', color=tick_color)
        elif self.X_gbsa_processed_train is None : #check training data
             self.ax_importance_gbsa.text(0.5, 0.5, "GBSA training data not available for context.", ha='center', va='center', color=tick_color)
        elif self.X_gbsa_processed_train.shape[1] == 0 :
             self.ax_importance_gbsa.text(0.5, 0.5, "No features in GBSA training data.", ha='center', va='center', color=tick_color)
        else:
            try:
                if not hasattr(self.gbsa_model, 'feature_importances_'):
                    self.ax_importance_gbsa.text(0.5, 0.5, "GBSA model does not have\nfeature_importances_ attribute.", ha='center', va='center', color=tick_color)
                else:
                    importances = self.gbsa_model.feature_importances_
                    feature_names_for_plot = np.array(self.trained_gbsa_feature_names)

                    if len(importances) != len(feature_names_for_plot):
                        self.ax_importance_gbsa.text(0.5, 0.5, "Mismatch between #importances\nand #feature names.", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
                    else:
                        sorted_idx = np.argsort(importances)[::-1] #sort in descending order
                        
                        importances_sorted = importances[sorted_idx]
                        feature_names_sorted = feature_names_for_plot[sorted_idx]

                        num_features_to_plot = min(len(importances_sorted), 10)
                        
                        self.ax_importance_gbsa.barh(range(num_features_to_plot), importances_sorted[:num_features_to_plot][::-1], align="center", color=bar_color)
                        self.ax_importance_gbsa.set_yticks(range(num_features_to_plot))
                        self.ax_importance_gbsa.set_yticklabels(feature_names_sorted[:num_features_to_plot][::-1], fontdict={'fontsize': STYLE_CONFIG["font_size_normal"]-2})
                        self.ax_importance_gbsa.set_xlabel("Feature Importance (Mean Decrease in Loss)", fontdict=label_font_dict)
                        self.log_training_message("Displaying GBSA built-in feature importances.", is_error=False)
            
            except Exception as e_fi:
                self.log_training_message(f"Error displaying GBSA feature importances: {type(e_fi).__name__} - {e_fi}", is_error=True)
                self.log_training_message(traceback.format_exc(), is_error=True)
                self.ax_importance_gbsa.text(0.5, 0.5, f"Error GBSA importances:\n{str(e_fi)[:40]}...", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
        
        self.ax_importance_gbsa.tick_params(axis='x', colors=tick_color); self.ax_importance_gbsa.tick_params(axis='y', colors=tick_color); self.ax_importance_gbsa.set_facecolor(STYLE_CONFIG["bg_entry"])
        
        #Cox coeffs. plot
        self.ax_cph_coeffs.set_title("CPH Model: Log(Hazard Ratios)", fontdict=title_font_dict)
        if clear_only or not self.cph_model or self.cph_model.params_ is None or self.cph_model.params_.empty:
            self.ax_cph_coeffs.text(0.5, 0.5, "Train CPH model for coefficients.", ha='center', va='center', color=tick_color)
        else:
            coeffs = self.cph_model.params_
            conf_int_df = self.cph_model.confidence_intervals_
            if not coeffs.empty and conf_int_df is not None and not conf_int_df.empty:
                sorted_coeffs = coeffs.sort_values(ascending=False); y_pos = np.arange(len(sorted_coeffs))
                param_lower_ci_col = next((col for col in conf_int_df.columns if 'lower' in col and '%' in col), None)
                param_upper_ci_col = next((col for col in conf_int_df.columns if 'upper' in col and '%' in col), None)
                if param_lower_ci_col and param_upper_ci_col and param_lower_ci_col in conf_int_df.columns and param_upper_ci_col in conf_int_df.columns:
                    valid_indices_for_ci = sorted_coeffs.index.intersection(conf_int_df.index)
                    if len(valid_indices_for_ci) == len(sorted_coeffs):
                        lower_ci_values = conf_int_df.loc[sorted_coeffs.index, param_lower_ci_col]; upper_ci_values = conf_int_df.loc[sorted_coeffs.index, param_upper_ci_col]
                        errors = [(sorted_coeffs - lower_ci_values).abs().values, (upper_ci_values - sorted_coeffs).abs().values]
                        self.ax_cph_coeffs.barh(y_pos, sorted_coeffs.values, align='center', color=bar_color, xerr=errors, capsize=3, ecolor=STYLE_CONFIG["fg_text"])
                    else: self.ax_cph_coeffs.barh(y_pos, sorted_coeffs.values, align='center', color=bar_color)
                else: self.ax_cph_coeffs.barh(y_pos, sorted_coeffs.values, align='center', color=bar_color)
                self.ax_cph_coeffs.set_yticks(y_pos); self.ax_cph_coeffs.set_yticklabels(sorted_coeffs.index, fontdict={'fontsize': STYLE_CONFIG["font_size_normal"]-2})
                self.ax_cph_coeffs.axvline(0, color=STYLE_CONFIG["border_color"], linestyle='--', linewidth=0.8); self.ax_cph_coeffs.set_xlabel("Log(Hazard Ratio)", fontdict=label_font_dict)
            else: self.ax_cph_coeffs.text(0.5, 0.5, "CPH coeffs/CI not available.", ha='center', va='center', color=tick_color)
        self.ax_cph_coeffs.tick_params(axis='x', colors=tick_color); self.ax_cph_coeffs.tick_params(axis='y', colors=tick_color); self.ax_cph_coeffs.set_facecolor(STYLE_CONFIG["bg_entry"])

        #predicted risk distb. plot
        self.ax_calibration_like.set_title("CPH: Predicted Risk Distribution (Test)", fontdict=title_font_dict)
        if clear_only or not self.cph_model or self.y_test_full_df_for_metrics is None or self.y_test_full_df_for_metrics.empty:
            self.ax_calibration_like.text(0.5, 0.5, "Train model for risk distribution.", ha='center', va='center', color=tick_color)
        else:
            try:
                cph_model_features = self.cph_model.params_.index.tolist()
                df_to_predict_on = self.y_test_full_df_for_metrics[[col for col in cph_model_features if col in self.y_test_full_df_for_metrics.columns]]
                if not df_to_predict_on.empty and all(col in df_to_predict_on.columns for col in cph_model_features):
                    partial_hazards_test = self.cph_model.predict_partial_hazard(df_to_predict_on)
                    event_test = self.y_test_full_df_for_metrics[self.target_event_col_var.get()]
                    sns.histplot(partial_hazards_test[event_test == 0], label='Censored (0) on Test', kde=True, ax=self.ax_calibration_like, color="skyblue", stat="density", bins=20, element="step")
                    sns.histplot(partial_hazards_test[event_test == 1], label='Event (1) on Test', kde=True, ax=self.ax_calibration_like, color="salmon", stat="density", bins=20, element="step")
                    self.ax_calibration_like.set_xlabel("Predicted Partial Hazard (Risk Score)", fontdict=label_font_dict); self.ax_calibration_like.set_ylabel("Density", fontdict=label_font_dict)
                    self.ax_calibration_like.legend(facecolor=STYLE_CONFIG["bg_entry"], edgecolor=STYLE_CONFIG["border_color"], labelcolor=tick_color)
                else:
                    missing_cols_str = ", ".join([col for col in cph_model_features if col not in self.y_test_full_df_for_metrics.columns])
                    self.ax_calibration_like.text(0.5, 0.5, f"Plot error: Missing CPH features\nin test data: {missing_cols_str[:50]}...", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
            except Exception as e: self.ax_calibration_like.text(0.5, 0.5, f"Risk dist. plot error:\n{str(e)[:30]}...", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
        self.ax_calibration_like.tick_params(colors=tick_color); self.ax_calibration_like.set_facecolor(STYLE_CONFIG["bg_entry"])

        #baseline survival plot
        self.ax_survival_curve.set_title("CPH: Baseline Survival Function", fontdict=title_font_dict)
        if clear_only or not self.cph_model or not hasattr(self.cph_model, 'baseline_survival_') or self.cph_model.baseline_survival_ is None or self.cph_model.baseline_survival_.empty:
            self.ax_survival_curve.text(0.5, 0.5, "Train CPH model for baseline survival.", ha='center', va='center', color=tick_color)
        else:
            try:
                self.cph_model.baseline_survival_.plot(ax=self.ax_survival_curve, color=bar_color, legend=False)
                time_unit = self.time_to_event_col_var.get().split('_')[-1] if self.time_to_event_col_var.get() else 'Days'
                if time_unit.lower() in ["col", "column", "var", "variable"]: time_unit = "Days"
                self.ax_survival_curve.set_xlabel(f"Time ({time_unit})", fontdict=label_font_dict); self.ax_survival_curve.set_ylabel("Survival Probability", fontdict=label_font_dict)
                self.ax_survival_curve.set_ylim(0, 1.05)
            except Exception as e: self.ax_survival_curve.text(0.5, 0.5, f"Baseline plot error:\n{str(e)[:30]}", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
        self.ax_survival_curve.tick_params(colors=tick_color); self.ax_survival_curve.set_facecolor(STYLE_CONFIG["bg_entry"])
        
        try: self.fig.tight_layout(pad=2.5)
        except Exception: pass 
        
        self.canvas.draw()


    def show_more_plots_window(self):
        if self.more_plots_window is not None and self.more_plots_window.winfo_exists():
            self.more_plots_window.lift(); return
        
        if not self.cph_model or self.y_test_full_df_for_metrics is None or self.y_test_full_df_for_metrics.empty:
            messagebox.showinfo("No Data", "CPH Model must be trained and test data available to view more plots.")
            return

        self.more_plots_window = tk.Toplevel(self.root)
        
        self.more_plots_window.title("CPH Model Diagnostic Plots (Test Set)")
        
        self.more_plots_window.geometry("950x450") 
        
        self.more_plots_window.configure(bg=STYLE_CONFIG["bg_root"])
        
        plot_canvas_frame = ttk.Frame(self.more_plots_window)
        
        plot_canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        fig_more = plt.Figure(figsize=(9, 4), dpi=100, facecolor=STYLE_CONFIG["bg_widget"])
        
        fig_more.subplots_adjust(hspace=0.35, top=0.90, bottom=0.12, left=0.1, right=0.95)
        
        fig_more.suptitle("CPH Model: Survival by Predicted Risk Quartile (Test Set)",
                            fontsize=STYLE_CONFIG["font_size_section"]+1, color=STYLE_CONFIG["fg_header"], weight="bold")
        
        ax_surv_by_risk = fig_more.add_subplot(1, 1, 1)
        
        canvas_more = FigureCanvasTkAgg(fig_more, master=plot_canvas_frame)
        
        label_font_dict = {'color': STYLE_CONFIG["fg_text"], 'fontsize': STYLE_CONFIG["font_size_normal"] - 1}
        
        tick_color = STYLE_CONFIG["fg_text"]
        
        base_test_df_for_plots = self.y_test_full_df_for_metrics.copy()
        
        time_col = self.time_to_event_col_var.get()
        
        event_col = self.target_event_col_var.get()
        
        cph_model_all_covariates = []
        
        if self.cph_model and self.cph_model.params_ is not None:
            cph_model_all_covariates = self.cph_model.params_.index.tolist()

        ax_surv_by_risk.clear()
        
        try:
            from lifelines import KaplanMeierFitter
            
            if not cph_model_all_covariates: raise ValueError("No covariates in CPH model for risk quartiles.")
            
            df_for_risk_pred = base_test_df_for_plots[[col for col in cph_model_all_covariates if col in base_test_df_for_plots.columns]].copy()
            
            df_for_risk_pred.dropna(subset=cph_model_all_covariates, how='any', inplace=True)
            
            if not df_for_risk_pred.empty:
                partial_hazards = self.cph_model.predict_partial_hazard(df_for_risk_pred)
                
                temp_plot_df = base_test_df_for_plots.loc[partial_hazards.index].copy() 
                
                temp_plot_df['risk_score'] = partial_hazards.values
                
                n_unique_scores = temp_plot_df['risk_score'].nunique()
                
                if n_unique_scores >= 4: temp_plot_df['risk_group'] = pd.qcut(temp_plot_df['risk_score'], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], duplicates='drop')
                
                elif n_unique_scores >= 2: temp_plot_df['risk_group'] = pd.qcut(temp_plot_df['risk_score'], 2, labels=["Low Risk", "High Risk"], duplicates='drop')
                
                elif n_unique_scores == 1: temp_plot_df['risk_group'] = "Overall (Single Risk Value)"
                
                else: raise ValueError("Not enough unique risk scores for groups.")
                
                for grp, df_grp in temp_plot_df.groupby('risk_group', observed=True):
                    if not df_grp.empty and time_col in df_grp and event_col in df_grp:
                        kmf = KaplanMeierFitter(); kmf.fit(df_grp[time_col], event_observed=df_grp[event_col], label=str(grp))
                        kmf.plot_survival_function(ax=ax_surv_by_risk)
                time_unit_plot = time_col.split('_')[-1] if time_col else 'Days'
                
                if time_unit_plot.lower() in ["col", "column", "var", "variable"]: time_unit_plot = "Days"
                
                ax_surv_by_risk.set_xlabel(f"Time ({time_unit_plot})", fontdict=label_font_dict); ax_surv_by_risk.set_ylabel("Survival Probability", fontdict=label_font_dict)
                
                ax_surv_by_risk.legend(title="Risk Group", facecolor=STYLE_CONFIG["bg_entry"], edgecolor=STYLE_CONFIG["border_color"], labelcolor=tick_color, fontsize='x-small')
            else: ax_surv_by_risk.text(0.5, 0.5, "Not enough data for risk quartiles\nafter NaN removal in covariates.", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
        except Exception as e:
            ax_surv_by_risk.text(0.5, 0.5, f"Error plotting survival by risk:\n{str(e)[:40]}", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
            
            self.log_training_message(f"Error in show_more_plots (survival by risk): {traceback.format_exc()}", is_error=True)
        ax_surv_by_risk.tick_params(colors=tick_color); ax_surv_by_risk.set_facecolor(STYLE_CONFIG["bg_entry"])
        
        canvas_more.draw()
        
        canvas_more.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(self.more_plots_window, style="Content.TFrame")
        
        btn_frame.pack(pady=(5,10), fill=tk.X)
        
        ttk.Button(btn_frame, text="Close", command=self.more_plots_window.destroy).pack()

    def show_about_dialog(self):
        messagebox.showinfo("About Advanced Dynamic CVD Predictor",
                             "Advanced Dynamic CVD Predictor -- GBSA-CPH Enhanced\n\n"
                             "Gradient Boosting Survival Analysis (GBSA) model generates a non-linear risk score. CPH model incorporates this score along with other linear predictors for survival analysis.\n\n"
                             "Developed by ODAT project.")

#main fun init here
if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicCVDApp(root)
    root.mainloop()