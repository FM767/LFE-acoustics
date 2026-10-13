# teste sem interface: stub do tkinter, corre calcular + graficos_resultados com os dados reais
import sys, types, os
from pathlib import Path
for n in ["tkinter","tkinter.ttk","tkinter.messagebox"]:
    m=types.ModuleType(n); sys.modules[n]=m
tk=sys.modules["tkinter"]; tk.Tk=object; tk.ttk=sys.modules["tkinter.ttk"]; tk.messagebox=sys.modules["tkinter.messagebox"]
import matplotlib; matplotlib.use("Agg"); matplotlib.use=lambda *a,**k:None
bt=types.ModuleType("matplotlib.backends.backend_tkagg"); bt.FigureCanvasTkAgg=None; bt.NavigationToolbar2Tk=None
sys.modules["matplotlib.backends.backend_tkagg"]=bt
AQUI=Path(sys.argv[1]); sys.path.insert(0,str(AQUI))
import importlib.util
spec=importlib.util.spec_from_file_location("app",AQUI/"app.py"); app=importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
app.PASTA=Path(sys.argv[2])
import json
class Fake: pass
self=Fake(); self.plano=json.loads((app.PASTA/"plano.json").read_text(encoding="utf-8"))
self.pasta=lambda pid: app.PASTA/"dados"/pid
self.feito=lambda pid: (self.pasta(pid)/"meta.json").exists()
L=app.App.calcular(self)
from matplotlib.figure import Figure
figs={n:Figure(figsize=(8,4.5)) for n in ("T(f) por a","Profundidade vs N","f_centro vs 1/a")}
txt=app.App.graficos_resultados(None,L,figs)
print(txt)
out=Path(sys.argv[3]); out.mkdir(exist_ok=True)
for n,f in figs.items(): f.savefig(out/(n.replace("/","_").replace(" ","_")+".png"),dpi=80)
