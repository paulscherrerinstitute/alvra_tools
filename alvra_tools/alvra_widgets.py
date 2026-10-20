import os, glob
import ipywidgets as widgets
from IPython.display import display, clear_output
from alvra_tools.analysis_apo import Rebin_and_filter, Rebin_with_scanvar_and_filter, Rebin_and_filter_2Dscans
from alvra_tools.utils import model_decay_1exp, model_decay_2exp

def Rebin_widget(data, Rebin_and_filter=Rebin_and_filter, 
                       Rebin_with_scanvar_and_filter=Rebin_with_scanvar_and_filter,
                       Rebin_and_filter_2Dscans=Rebin_and_filter_2Dscans):

    out = {
        "signal1": None, 
        "signal2": None, 
    }

    #out = {
    #    "results1": None, 
    #    "results2": None, 
    #    "params": None
    #}

    mode_w = widgets.ToggleButtons(
        options=[
            ("Scan", "scanvar"),
            ("Rebin", "rebin"),
            ("Rebin 2D", "2D")
        ],
        description="Mode:"
    )

    # Scalars
    binsize_w = widgets.FloatText(value=50.0, description='Binsize:')
    minvalue_w = widgets.FloatText(value=0.0, description='Min:')
    maxvalue_w = widgets.FloatText(value=10.0, description='Max:')
    quantile_w = widgets.FloatSlider(value=0.7, min=0, max=1, step=0.01, description='Quantile:')

    # Strings (dropdown or text depending on use case)
    signal1_w = widgets.Dropdown(
        options=['diode1', 'diode2', 'diode3', 'laser'],
        description='Signal 1'
    )

    signal2_w = widgets.Dropdown(
        options=['None', 'diode1', 'diode2', 'diode3', 'laser'],
        description='Signal 2'
    )
    
    izero_w = widgets.Dropdown(
        options=['Izero110', 'Izero122', 'Izero125','diode1', 'diode2', 'diode3', 'laser', 'laserIzero'],
        description='Izero:'
    )

    TT_w = widgets.Dropdown(
        options=['126', '124'],
        description='TT:'
    )

    # Booleans
    withTT_w = widgets.Checkbox(value=False, description='With TT correction')
    YAGscan_w = widgets.Checkbox(value=False, description='YAG scan?')

    # Button
    run_button = widgets.Button(description="Run", button_style='success')

    # Output area
    output = widgets.Output()


    def on_run_clicked(b):
        with output:
            output.clear_output()

            try:
                if mode_w.value == "rebin":
                    func = Rebin_and_filter
                    kwargs = dict(
                        binsize=binsize_w.value,
                        minvalue=minvalue_w.value,
                        maxvalue=maxvalue_w.value,
                        quantile=quantile_w.value,
                        izero=izero_w.value,
                        TT=TT_w.value,
                        YAGscan=YAGscan_w.value,
                        withTT=withTT_w.value
                    )
                elif mode_w.value == "scanvar":
                    func = Rebin_with_scanvar_and_filter
                    kwargs = dict(
                        quantile=quantile_w.value,
                        izero=izero_w.value,
                        TT=TT_w.value,
                        YAGscan=YAGscan_w.value,
                        withTT=withTT_w.value
                    )
                else:
                    func = Rebin_and_filter_2Dscans
                    kwargs = dict(
                        binsize=binsize_w.value,
                        minvalue=minvalue_w.value,
                        maxvalue=maxvalue_w.value,
                        quantile=quantile_w.value,
                        izero=izero_w.value,
                        TT=TT_w.value,
                        withTT=withTT_w.value
                    )

                print (f"Running signal: {signal1_w.value} / Izero: {izero_w.value}")   
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
                cmd = f"{func.__name__}(signal={repr(signal1_w.value)}, data=data, {kwargs_str})"
                print (cmd)
                out["results1"] = func(signal=signal1_w.value, data=data, **kwargs)
                out["results2"] = None
                if signal2_w.value != "None":
                    print (f"Running signal: {signal2_w.value} / Izero: {izero_w.value}") 
                    cmd2 = f"{func.__name__}(signal={repr(signal2_w.value)}, data=data, {kwargs_str})"
                    print (cmd2)
                    out["results2"] = func(signal=signal2_w.value, data=data, **kwargs)

                out["params"] = {
                    "signal1": signal1_w.value,
                    "signal2": signal2_w.value,
                    "izero": izero_w.value, # you fill this in above already -- probably just remove
                    **kwargs
                }

                out["signal1"] = {
                    "results": out["results1"],
                    "params": out["params"],
                    "which": "signal1"
                }

                if out["results2"] is not None:
                    out["signal2"] = {
                        "results": out["results2"],
                        "params": out["params"],
                        "which": "signal2"
                    }

                else:
                    out["signal2"] = None            
        
                print("Done!")
            except Exception as e:
                print ("Error {}".format(e))

    run_button.on_click(on_run_clicked)
    #run_button_scanvar.on_click(on_run_scanvar_clicked)

    rebin_only_widgets = widgets.VBox([
        widgets.HBox([binsize_w]),
        widgets.HBox([minvalue_w, maxvalue_w])
    ])

    common_widgets = widgets.VBox([
        quantile_w,
        widgets.HBox([signal1_w, signal2_w]), 
        izero_w,
        TT_w,
        widgets.HBox([withTT_w, YAGscan_w])
    ])

    def update_visibility(change):
        if change["new"] == "rebin":
            rebin_only_widgets.layout.display = "block"
        elif change["new"] == '2D':
            rebin_only_widgets.layout.display = "block"
        else:
            rebin_only_widgets.layout.display = "none"

    mode_w.observe(update_visibility, names="value")
    update_visibility({"new": mode_w.value})

    



    ui = widgets.VBox([
        widgets.HTML("<h3>Rebin Parameters</h3>"),
        mode_w,
        rebin_only_widgets,
        common_widgets,
        run_button,
        output
    ])

    return ui, out


def interactive_lineouts(data, meta):

    def update(energy, delay):
        energylist = [energy]
        delayslist = [delay]

        plt.close('all')

        plot_lineouts(data, meta,
                      energylist=energylist,
                      delayslist=delayslist)

        plt.show()

    energy_slider = widgets.FloatSlider(
        min=min(data['results']['scanvar_rebin']),
        max=max(data['results']['scanvar_rebin']),
        step=0.1,
        description='Energy'
    )

    delay_slider = widgets.FloatSlider(
        min=min(data['results']['delay_rebin']),
        max=max(data['results']['delay_rebin']),
        step=10,
        description='Delay'
    )

    ui = widgets.VBox([energy_slider, delay_slider])

    out = widgets.interactive_output(update, {
        'energy': energy_slider,
        'delay': delay_slider
    })

    display(ui, out)


class RunSelector:
    def __init__(self, LoadDir):
        self.LoadDir = LoadDir
        self.groups = []
        self.runlist = None
        self.single_runs, self.multi_runs = scan_runs(LoadDir)
        self._build_ui()

    def _build_ui(self):

        self.single_select = widgets.SelectMultiple(
            options=[(name, num) for name, num in self.single_runs],
            description='Single',
            rows=10
        )
        
        self.multi_select = widgets.SelectMultiple(
            options=[(name, nums) for name, nums in self.multi_runs],
            description='Multi',
            rows=10
        )

        self.group_list = widgets.Select(description='Selected',rows=10)
        self.output = widgets.Output()

        self.btn_new = widgets.Button(description="Add Runs")
        self.btn_remove = widgets.Button(description="Remove Runs")
        self.btn_clear = widgets.Button(description= "Clear all Runs")
        self.btn_export = widgets.Button(description="Print and Set runlist2load", layout=widgets.Layout(width='200px'), button_style='success')

        self.btn_new.on_click(self.add_new_group)
        self.btn_remove.on_click(self.remove_group)
        self.btn_clear.on_click(self.clear_all_groups)   
        self.btn_export.on_click(self.export_result)

        self.ui = widgets.VBox([
            widgets.HBox([self.single_select, self.multi_select, self.group_list]),
            widgets.HBox([self.btn_new, self.btn_remove, self.btn_clear, self.btn_export]),
            self.output
        ])

    def add_new_group(self, _):
        new_group = []

        # add singles
        for val in self.single_select.value:
            new_group.append(val)

        # add multis
        for val in self.multi_select.value:
            new_group.append(val)

        if not new_group:
            return

        self.groups.append(new_group)
        self.refresh_groups()

        self.single_select.value = ()
        self.multi_select.value = ()

    def remove_group(self, _):
        if self.group_list.index is None:
            return

        self.groups.pop(self.group_list.index)
        self.refresh_groups()

    def refresh_groups(self):
        display_list = []
        for g in self.groups:
            flat = []
            for item in g:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            display_list.append(str(flat))

        self.group_list.options = display_list

    def export_result(self, _):
        self.runlist = []
        for g in self.groups:
            merged = []
            for item in g:
                if isinstance(item, list):
                    merged.extend(item)
                else:
                    merged.append(item)
            self.runlist.append(merged)

        with self.output:
            self.output.clear_output()
            print ('runlist2load ready:')
            print (self.runlist)

    def clear_all_groups(self, _):
        self.groups.clear()
        self.refresh_groups()

        self.single_select.value = ()
        self.multi_select.value = ()

    def display(self):
        display(self.ui)


def scan_runs(LoadDir):
    single_dir = LoadDir + '_singlerun/'
    multi_dir  = LoadDir + '_multiruns/'

    single_runs = []
    multi_runs  = []

    # --- scan singleruns ---
    if os.path.exists(single_dir):
        paths = glob.glob(os.path.join(single_dir, "run*"))
        paths = sorted(paths, key=lambda p: int(os.path.basename(p).replace("run", "")),reverse=True)
        for path in paths:
            name = os.path.basename(path)
            num = int(name.replace("run", ""))
            single_runs.append((name, num))		

    # --- scan multiruns ---
    if os.path.exists(multi_dir):
        paths = glob.glob(os.path.join(multi_dir, "run*"))
        paths = sorted(paths, key=lambda p: tuple(int(x) for x in os.path.basename(p).replace("run", "").split("_")), reverse=True)
        for path in paths: #sorted(glob.glob(os.path.join(multi_dir, "run*"))):
            name = os.path.basename(path)
            nums = [int(x) for x in name.replace("run", "").split("_")]
            multi_runs.append((name, nums))

    return single_runs, multi_runs

def RunSelectorUI(LoadDir):
    selector = RunSelector(LoadDir)
    selector.display()
    return selector

##################################################

class DecayFitWidget:

    def __init__(self, plotter, dataplot1, dataplot2, meta):

        self.plotter = plotter
        self.data = {
            "dataplot1": dataplot1,
            "dataplot2": dataplot2,
        }
        self.meta = meta

        # -----------------------------
        # Data selection
        # -----------------------------
        self.data_dropdown = widgets.Dropdown(
            options=["dataplot1", "dataplot2"],
            value="dataplot2",
            description="Data:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="200px"),
        )

        self.which_pp_dropdown = widgets.Dropdown(
            options=["pp_r", "pp_d"],
            value="pp_r",
            description="Signal:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="180px"),
        )

        # -----------------------------
        # Function selection
        # -----------------------------
        self.function_dropdown = widgets.Dropdown(
            options=[
                "1-exp decay",
                "2-exp decay",
            ],
            value="2-exp decay",
            description="Function:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="300px"),
        )

        # -----------------------------
        # Buttons
        # -----------------------------
        self.fit_button = widgets.Button(
            description="Fit",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="100px"),
        )

        self.reset_button = widgets.Button(
            description="Reset",
            icon="refresh",
            layout=widgets.Layout(width="100px"),
        )

        # -----------------------------
        # Parameter area
        # -----------------------------
        self.parameter_box = widgets.VBox()

        # -----------------------------
        # Plot output
        # -----------------------------
        self.plot_output = widgets.Output()

        # General output for messages
        self.output = widgets.Output()

        # -----------------------------
        # Connections
        # -----------------------------
        self.function_dropdown.observe(
            self._function_changed,
            names="value"
        )

        self.fit_button.on_click(self._fit)

        self.reset_button.on_click(self._reset)

        # Build initial parameters
        self._build_parameter_widgets()

        # -----------------------------
        # Layout
        # -----------------------------
        self.widget = widgets.VBox([
            widgets.HTML("<h4>Decay fit</h4>"),

            # Data + signal on same line
            widgets.HBox([
                self.data_dropdown,
                self.which_pp_dropdown,
            ]),

            # Function
            self.function_dropdown,

            widgets.HTML("<br><b>Parameters</b>"),

            self.parameter_box,

            widgets.HBox([
                self.fit_button,
                self.reset_button,
            ]),

            self.output,

            # Plot appears here
            self.plot_output,
        ])


    # ============================================================
    # Parameter definitions
    # ============================================================

    def _get_function_info(self):

        if self.function_dropdown.value == "1-exp decay":

            return {
                "function": model_decay_1exp,
                "parameters": [
                    ("x0", 0),
                    ("sigma", 100),
                    ("amp1", 0.1),
                    ("tau1", 1000),
                    ("C", 0),
                ],
            }

        elif self.function_dropdown.value == "2-exp decay":

            return {
                "function": model_decay_2exp,
                "parameters": [
                    ("x0", 0),
                    ("sigma", 100),
                    ("amp1", 0.1),
                    ("tau1", 1000),
                    ("C", 0),
                    ("amp2", 0.05),
                    ("tau2", 1000),
                ],
            }
            

    # ============================================================
    # Build parameter widgets
    # ============================================================

    def _build_parameter_widgets(self):

        function_info = self._get_function_info()

        self.parameter_widgets = []

        rows = []

        # Header
        rows.append(
            widgets.HBox([
                widgets.Label("Parameter", layout=widgets.Layout(width="100px")),
                widgets.Label("Initial value", layout=widgets.Layout(width="120px")),
                widgets.Label("Fixed", layout=widgets.Layout(width="60px")),
            ])
        )

        for name, default in function_info["parameters"]:

            value_widget = widgets.FloatText(
                value=default,
                description="",
                layout=widgets.Layout(width="120px"),
            )

            fixed_widget = widgets.Checkbox(
                value=False,
                description="",
                indent=False,
                layout=widgets.Layout(width="60px"),
            )

            row = widgets.HBox([
                widgets.Label(
                    name,
                    layout=widgets.Layout(width="100px")
                ),
                value_widget,
                fixed_widget,
            ])

            rows.append(row)

            self.parameter_widgets.append({
                "name": name,
                "value": value_widget,
                "fixed": fixed_widget,
            })

        self.parameter_box.children = rows


    # ============================================================
    # Function changed
    # ============================================================

    def _function_changed(self, change):

        self._build_parameter_widgets()


    # ============================================================
    # Reset
    # ============================================================

    def _reset(self, button=None):

        self._build_parameter_widgets()

        with self.output:
            clear_output()

        with self.plot_output:
            clear_output()


    # ============================================================
    # Get current settings
    # ============================================================

    def get_settings(self):

        function_info = self._get_function_info()

        p0 = []
        fixed = {}

        for i, p in enumerate(self.parameter_widgets):

            value = p["value"].value

            p0.append(value)

            if p["fixed"].value:
                fixed[i] = value

        return {
            "data": self.data[self.data_dropdown.value],
            "data_name": self.data_dropdown.value,

            "which_pp": self.which_pp_dropdown.value,

            "fitfunction": function_info["function"],

            "p0": p0,

            "fixed": fixed,
        }


    # ============================================================
    # Fit
    # ============================================================

    def _fit(self, button=None):

        settings = self.get_settings()

        with self.output:
            clear_output()

            print("Fitting...")
            print(f"Data:     {settings['data_name']}")
            print(f"Signal:   {settings['which_pp']}")
            print(f"Function: {settings['fitfunction'].__name__}")
            print(f"p0:       {settings['p0']}")
            print(f"Fixed:    {settings['fixed']}")

        # Remove previous plot
        with self.plot_output:
            clear_output()

            try:

                fig, axes = self.plotter.fit_decay(
                    settings["data"],
                    self.meta,
                    fitfunction=settings["fitfunction"],
                    p0=settings["p0"],
                    fixed_params=settings["fixed"],
                    which_pp=settings["which_pp"],
                    errbars=True,
                )

                display(fig)
                

            except Exception as e:

                print("Fit failed:")
                print(e)


    # ============================================================
    # Display widget
    # ============================================================

    def display(self):

        display(self.widget)


