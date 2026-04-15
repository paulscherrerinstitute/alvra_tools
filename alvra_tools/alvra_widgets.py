import os, glob
import ipywidgets as widgets
from IPython.display import display
from alvra_tools.analysis_apo import Rebin_and_filter, Rebin_with_scanvar_and_filter, Rebin_and_filter_2Dscans

def Rebin_widget(data, Rebin_and_filter=Rebin_and_filter, 
                       Rebin_with_scanvar_and_filter=Rebin_with_scanvar_and_filter,
                       Rebin_and_filter_2Dscans=Rebin_and_filter_2Dscans):

    out = {
        "results1": None, 
        "results2": None, 
        "params": None
    }

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
        options=['Izero110', 'Izero122', 'diode1', 'diode2', 'diode3'],
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

    





