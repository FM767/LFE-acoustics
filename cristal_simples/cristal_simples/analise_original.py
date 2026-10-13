"""
analise.py — as contas da experiência (sem interface).

Ideia:
  1. Gravar ruído branco SEM cristal (referência) e COM cristal, 5 vezes cada.
  2. Espectro de cada gravação pelo método de Welch.
  3. Transmissão T(f) = espectro com cristal / espectro de referência, em dB.
  4. Hiato = zona onde T fica 6 dB abaixo do nível normal, perto de c/(2a).
  5. Cada número = média das 5 repetições ± desvio-padrão / √5.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

# ------------------------------------------------------------ parâmetros fixos
FS = 48000          # amostragem (Hz)
NPERSEG = 4096      # tamanho de cada segmento de Welch → Δf = 11.7 Hz
FMIN, FMAX = 500, 12000   # banda analisada (Hz)
SUAVIZAR_HZ = 100   # média móvel aplicada a T(f)
LIMIAR_DB = 6       # hiato = T abaixo de T0 − 6 dB
JANELA_PROF = 0.10  # profundidade = atenuação média em c/2a ± 10 %


def velocidade_som(temp_C):
    return 331.3 * np.sqrt(1 + temp_C / 273.15)


def f_bragg(a_mm, temp_C, direcao="ΓX"):
    """1.º hiato: ΓX → f = c/(2a);  ΓM (cristal rodado 45°) → f = c/(√2·a)."""
    f = velocidade_som(temp_C) / (2 * a_mm / 1000)
    return f * np.sqrt(2) if direcao == "ΓM" else f


# ------------------------------------------------------------ espectros
def espectro(y, dur):
    """PSD de Welch de uma gravação (usa só o troço onde há ruído)."""
    y = y[int(0.5 * FS):int(dur * FS)]
    f, P = welch(y, fs=FS, window="hann", nperseg=NPERSEG)
    banda = (f >= FMIN) & (f <= FMAX)
    return f[banda], P[banda]


def ler_espectros(pasta):
    pasta = Path(pasta)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    Ps = []
    for fich in sorted(pasta.glob("*rep*.wav")):
        y, _ = sf.read(fich, dtype="float64")
        f, P = espectro(y, meta["dur"])
        Ps.append(P)
    return meta, f, np.array(Ps)


def suavizar(T, f):
    n = max(1, int(round(SUAVIZAR_HZ / (f[1] - f[0]))))
    return uniform_filter1d(T, n, mode="nearest")


# ------------------------------------------------------------ hiato
def encontrar_hiato(f, Ts, fb):
    """Zona contínua abaixo de T0 − limiar mais perto de c/2a.
    Devolve (f_inf, f_sup) ou None."""
    T0 = np.median(Ts)
    abaixo = Ts < T0 - LIMIAR_DB
    zonas, ini = [], None
    for i, b in enumerate(abaixo):
        if b and ini is None:
            ini = i
        if not b and ini is not None:
            zonas.append((ini, i - 1))
            ini = None
    if ini is not None:
        zonas.append((ini, len(f) - 1))
    # junta zonas separadas por menos de 100 Hz
    juntas = []
    for z in zonas:
        if juntas and f[z[0]] - f[juntas[-1][1]] < 100:
            juntas[-1] = (juntas[-1][0], z[1])
        else:
            juntas.append(z)
    boas = [z for z in juntas if abs((f[z[0]] + f[z[1]]) / 2 - fb) < 0.3 * fb]
    if not boas:
        return None
    z = min(boas, key=lambda z: abs((f[z[0]] + f[z[1]]) / 2 - fb))
    return f[z[0]], f[z[1]]


def numeros(f, T, fb):
    """Grandezas de UMA curva T(f) (em dB)."""
    Ts = suavizar(T, f)
    T0 = np.median(Ts)
    jan = (f > fb * (1 - JANELA_PROF)) & (f < fb * (1 + JANELA_PROF))
    out = dict(profundidade=T0 - Ts[jan].mean(), fc=np.nan, largura=np.nan,
               f_inf=np.nan, f_sup=np.nan)
    h = encontrar_hiato(f, Ts, fb)
    if h:
        out.update(f_inf=h[0], f_sup=h[1], fc=(h[0] + h[1]) / 2, largura=h[1] - h[0])
    return out


def media_erro(valores):
    v = np.array([x for x in valores if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan
    if v.size == 1:
        return v[0], np.nan
    return v.mean(), v.std(ddof=1) / np.sqrt(v.size)


def analisar(pasta_cristal, pasta_ref):
    """T(f) média ± σ e os números do hiato (média ± erro das repetições)."""
    meta, f, Pc = ler_espectros(pasta_cristal)
    _, _, Pr = ler_espectros(pasta_ref)
    Ti = np.array([10 * np.log10(Pc[i] / Pr[i % len(Pr)]) for i in range(len(Pc))])
    T = Ti.mean(axis=0)
    sT = Ti.std(axis=0, ddof=1) / np.sqrt(len(Ti)) if len(Ti) > 1 else np.zeros_like(T)
    fb = f_bragg(meta["a_mm"], meta["temp_C"], meta.get("dir", "ΓX"))
    por_rep = [numeros(f, t, fb) for t in Ti]
    res = dict(meta=meta, f=f, T=T, sT=sT, Ts=suavizar(T, f), fb=fb, n=len(Ti),
               T0=float(np.median(suavizar(T, f))), hiato=encontrar_hiato(f, suavizar(T, f), fb))
    for k in ("profundidade", "fc", "largura", "f_inf", "f_sup"):
        res[k], res["s_" + k] = media_erro([r[k] for r in por_rep])
    return res


# ------------------------------------------------------------ regressões
def reta(x, y, sy):
    """Regressão linear ponderada y = m x + b. Devolve m, σm, b, σb, χ²/ν."""
    x, y, sy = map(np.asarray, (x, y, sy))
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, np.nanmax(sy) if np.isfinite(sy).any() else 1)
    w = 1 / sy ** 2
    S, Sx, Sy, Sxx, Sxy = w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx ** 2
    m, b = (S * Sxy - Sx * Sy) / D, (Sxx * Sy - Sx * Sxy) / D
    chi2 = (w * (y - m * x - b) ** 2).sum()
    nu = len(x) - 2
    return dict(m=m, sm=np.sqrt(S / D), b=b, sb=np.sqrt(Sxx / D),
                chi2nu=chi2 / nu if nu > 0 else np.nan)


# ------------------------------------------------------------ simulação (treino)
def simular(n, passo, temp_C=20, lado_mm=15.8, rng=np.random.default_rng()):
    """Gravação falsa: coluna + (cristal com hiato de Bragg) + ruído."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    ganho = 1 / (1 + (200 / (f + 1)) ** 4) / (1 + (f / 14000) ** 4)   # coluna
    if passo["tipo"] in ("cristal", "extra"):
        a, N = passo["a_mm"], passo["N"]
        ff = (lado_mm / a) ** 2
        # pequenas diferenças de montagem de repetição para repetição
        fb = f_bragg(a, temp_C, passo.get("dir", "ΓX")) * (1 - 0.1 * ff) * (1 + 0.003 * rng.standard_normal())
        meia = fb * ff
        N = N * (1 + 0.04 * rng.standard_normal())
        dentro = 1 / (1 + np.exp(-(f - fb + meia) / 60)) / (1 + np.exp((f - fb - meia) / 60))
        TdB = -N * 50 * ff * dentro
        TdB = 10 * np.log10(10 ** (TdB / 10) + 10 ** (-2.4))       # satura a −24 dB
        TdB += 0.8 * np.cos(2 * np.pi * f * 2 * N * a / 1000 / velocidade_som(temp_C))
        ganho = ganho * 10 ** (TdB / 20)
    y = np.fft.irfft(X * ganho, n) * 0.05 * (1 + 0.03 * rng.standard_normal())
    y += 2e-4 * rng.standard_normal(n)
    return y.astype(np.float32)
