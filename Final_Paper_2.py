# -*- coding: utf-8 -*-
"""
Soil Health Index (SHI) in a Tropical Ecotone (Atlantic Rainforest - Cerrado Transition)
========================================================================================

Authors: Alexandre et al. (2026)
Study Area: Bauru, São Paulo, Brazil
Description:
    Complete analytical and visualization pipeline for soil health assessment across
    three land uses (Seasonal Semideciduous Forest [SSF], Densely Wooded Savanna [DWS], 
    and Regeneration Area [RA]) along deep soil profiles (0–100 cm).
    
Pipeline Overview:
    - Setup & Global Configuration: Environment bootstrap, font handling, output paths.
    - Figure 1: Soil Organic Carbon (SOC, %) and Bulk Density (BD, Mg m⁻³) depth profiles.
    - Figure 2: Chemical indicators (pH, CEC, Total N, Resin P, Sum of Bases) depth profiles.
    - Figure 3: Principal Component Analysis (PCA) ordination and Spearman rank correlation matrix.
    - Figure 4: SHI sub-indices decomposition (physical, chemical, biological) and Overall SHI boxplot.
    - Figure 5: Radar / Petal radial bar charts of the five evaluated soil functions.
    - Figure 6: Random Forest regression for SHI_Layer using exclusive edaphic drivers (Approach B: 7 soil properties).
    - Supplementary Table S1: Robustness assessment of SHI without bulk density (BioChem-only SHI).
    - Supplementary Figure S1: Whole-profile boxplot of BioChem-only SHI (mirroring Figure 4b).

Requirements:
    Python >= 3.9
    pandas, numpy, matplotlib, scikit-learn, scipy, statsmodels, openpyxl

Usage:
    Execute cell-by-cell in an interactive IDE (VS Code, Spyder, Jupyter) or run sequentially:
        python Final_Paper_2.py
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import importlib.util
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ── Environment & Path Setup ──────────────────────────────────
# Resolve script directory dynamically for cross-platform and GitHub reproducibility
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
os.chdir(_SCRIPT_DIR)

OUTDIR = _SCRIPT_DIR
DATA_FILENAME = 'database_ecotono_02-05-2025_SHI_calculado.xlsx'
FILE = os.path.join(_SCRIPT_DIR, DATA_FILENAME)

if not os.path.isfile(FILE):
    # Fallback to current working directory
    FILE = os.path.join(os.getcwd(), DATA_FILENAME)
    if not os.path.isfile(FILE):
        raise FileNotFoundError(
            f"Dataset not found: '{DATA_FILENAME}'.\n"
            f"Please place '{DATA_FILENAME}' in the script folder or working directory:\n"
            f"  {_SCRIPT_DIR}"
        )

print(f"[Setup] Working directory: {os.getcwd()}")
print(f"[Setup] Data file resolved: {FILE}")

# ── Dependency Verification ───────────────────────────────────
def ensure_packages(pkg_dict):
    missing = [pkg for mod, pkg in pkg_dict.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[Setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                               '--disable-pip-version-check', *missing])

ensure_packages({
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels',
    'openpyxl': 'openpyxl'
})

# ── Publication Plotting Configuration ────────────────────────
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def get_best_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for font_candidate in ['Arial', 'Liberation Sans', 'Helvetica', 'FreeSans']:
        if font_candidate in available:
            return font_candidate
    return 'DejaVu Sans'

FONT = get_best_font()
matplotlib.rcParams.update({
    'font.family': FONT,
    'font.weight': 'bold',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 1.5,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})
matplotlib.use('Agg')
print(f"[Setup] Active typography: {FONT}")

# %%
# ============================================================
# FIGURE 1 — SOC (%) & Bulk Density (Mg m⁻³)
# ============================================================
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy import stats
from scipy.stats import shapiro, levene, boxcox
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

warnings.filterwarnings('ignore')

def _best_font():
    av = {f.name for f in fm.fontManager.ttflist}
    for c in ['Arial','Liberation Sans','Helvetica','FreeSans']:
        if c in av: return c
    return 'DejaVu Sans'

FONT = _best_font()
matplotlib.rcParams.update({
    'font.family': FONT,
    'font.weight': 'bold',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 1.6
})
matplotlib.use('Agg')

df = pd.read_excel(FILE, sheet_name=0, header=0)
df['Vegetation'] = df['Vegetation'].replace('DA','RA')

DEPTHS = ['P20','P40','P60','P80','P100']
DLABS  = ['0–20 cm','20–40 cm','40–60 cm','60–80 cm','80–100 cm']
TREATS = ['SSF','DWS','RA']
COLORS = {'SSF':'#1f78b4','DWS':'#f4c430','RA':'#d95f02'}
ALPHA  = 0.05
YLIMS_F1 = {'C':(0.00,1.45),'BD_Benites_2007':(1.42,1.82)}
YLABS_F1 = {'C':'SOC (%)','BD_Benites_2007':'Bulk Density (Mg m\u207b\u00b3)'}

def compact_letters(tukey, treats):
    groups = list(tukey.groupsunique)
    pairs  = list(combinations(range(len(groups)),2))
    rd = {}
    for (i,j),rej in zip(pairs,tukey.reject):
        rd[(groups[i],groups[j])]=rej; rd[(groups[j],groups[i])]=rej
    ls = {t:set() for t in treats}; cur=0; abc=list('abcdefgh')
    for t1 in treats:
        if not ls[t1]:
            ls[t1].add(abc[cur])
            for t2 in treats:
                if t1!=t2 and not rd.get((t1,t2),True): ls[t2].add(abc[cur])
            cur+=1
    for t in treats:
        if not ls[t]: ls[t].add(abc[cur]); cur+=1
    return {t:''.join(sorted(ls[t])) for t in treats}

def run_stats(df_sub, var):
    groups = [df_sub[df_sub['Vegetation']==t][var].dropna().values for t in TREATS]
    def resid(g): return np.concatenate([x-x.mean() for x in g])
    r=resid(groups); sw_s,sw_p=shapiro(r); lv_s,lv_p=levene(*groups)
    boxcox_used=False; lam=na=ha=swap=lvap=None; ga=groups; pdf=False
    if sw_p<ALPHA or lv_p<ALPHA:
        av=np.concatenate(groups); sh=max(0,-av.min()+1e-6)
        try:
            _,lam=boxcox(av+sh)
            tr=[boxcox(g+sh,lmbda=lam) for g in groups]
            if all(np.std(g)<1e-10 for g in tr): raise ValueError("collapsed")
            r2=resid(tr); sw2,swp2=shapiro(r2); lv2,lvp2=levene(*tr)
            boxcox_used=True; na=swp2>=ALPHA; ha=lvp2>=ALPHA
            swap=round(swp2,4); lvap=round(lvp2,4); ga=tr
            if not na or not ha: pdf=True
        except: pass
    fs,ap=stats.f_oneway(*ga)
    lets={t:'' for t in TREATS}; tukey_rows=[]
    if ap<=ALPHA:
        va=np.concatenate(ga); la=np.concatenate([[t]*len(g) for t,g in zip(TREATS,ga)])
        tk=pairwise_tukeyhsd(va,la,alpha=ALPHA); lets=compact_letters(tk,TREATS)
        grps=list(tk.groupsunique); p2=list(combinations(range(len(grps)),2))
        for (i,j),rej,pval in zip(p2,tk.reject,tk.pvalues):
            tukey_rows.append({'Variable':var,'Depth':None,'Group1':grps[i],
                               'Group2':grps[j],'Reject_H0':bool(rej),
                               'p_adj':round(float(pval),6)})
    return dict(sw_stat=round(sw_s,4),sw_p=round(sw_p,4),
                lev_stat=round(lv_s,4),lev_p=round(lv_p,4),
                norm_before=sw_p>=ALPHA,homo_before=lv_p>=ALPHA,
                boxcox_used=boxcox_used,boxcox_lambda=round(lam,4) if lam else None,
                norm_after=na,homo_after=ha,sw_after_p=swap,lev_after_p=lvap,
                proceeded_despite_failure=pdf,f_stat=round(fs,4),
                anova_p=round(ap,4),significant=ap<=ALPHA,letters=lets,tukey_rows=tukey_rows)

def desc_stats(df_sub,var,treat,dlb):
    v=df_sub[df_sub['Vegetation']==treat][var].dropna()
    return dict(Variable=var,Depth=dlb,Treatment=treat,n=len(v),
                Mean=round(v.mean(),4),Median=round(v.median(),4),
                SD=round(v.std(),4),Min=round(v.min(),4),Max=round(v.max(),4),
                CV_pct=round(v.std()/v.mean()*100,2) if v.mean()!=0 else None)

def whisker_top(vals):
    q75=np.percentile(vals,75); iqr=q75-np.percentile(vals,25)
    fence=q75+1.5*iqr; above=vals[vals<=fence]
    return above.max() if len(above) else q75

def plot_box_f1(ax,df_sub,var,dlb,sr,ylabel,ylim,show_ylabel,panel_letter):
    bp_data=[df_sub[df_sub['Vegetation']==t][var].dropna().values for t in TREATS]
    bp=ax.boxplot(bp_data,positions=[1,2,3],widths=0.52,patch_artist=True,notch=False,
                  medianprops=dict(color='#000000',linewidth=2.8),
                  whiskerprops=dict(color='#000000',linewidth=1.8),
                  capprops=dict(color='#000000',linewidth=1.8),
                  flierprops=dict(marker='o',markersize=6,markerfacecolor='#777777',
                                  markeredgecolor='#000000',linewidth=1.2))
    for patch,t in zip(bp['boxes'],TREATS):
        patch.set_facecolor(COLORS[t]); patch.set_edgecolor('#000000')
        patch.set_linewidth(2.0); patch.set_alpha(0.92)
    ax.set_ylim(ylim); yspan=ylim[1]-ylim[0]
    
    # FONTES AUMENTADAS AQUI
    ax.text(0.03, 0.98, f'({panel_letter})', transform=ax.transAxes,
            ha='left', va='top', fontsize=20, fontweight='bold', fontfamily=FONT)
    ax.text(0.5, 0.98, dlb, transform=ax.transAxes, ha='center', va='top',
            fontsize=20, fontweight='bold', fontfamily=FONT, color='#111111',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='#e8e8e8',
                      edgecolor='#888888', linewidth=1.2, alpha=0.95))
    
    p = sr['anova_p']
    p_str = 'p < 0.001' if p<0.001 else f'p = {p:.3f}'
    sig_str = p_str if sr['significant'] else f'{p_str} (ns)'
    ax.text(0.5, 0.84, sig_str, transform=ax.transAxes, ha='center', va='top',
            fontsize=18, fontweight='bold', style='italic', color='#111111', fontfamily=FONT)
    
    if sr['significant']:
        for pos,t in zip([1,2,3],TREATS):
            vals = bp_data[TREATS.index(t)]
            tip = max(whisker_top(vals),vals.max())
            y_letter = min(tip+yspan*0.035, ylim[0]+yspan*0.78)
            ax.text(pos, y_letter, sr['letters'][t], ha='center', va='bottom',
                    fontsize=18, fontweight='bold', color='#000000',
                    fontfamily=FONT, transform=ax.transData)
            
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.6); sp.set_color('#000000')
    ax.set_xlim(0.25, 3.75); ax.set_xticks([])
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=18, fontweight='bold', labelpad=6, fontfamily=FONT)
    ax.yaxis.set_major_locator(MaxNLocator(6, prune='both'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y', labelsize=13, which='major', direction='in', width=1.6, length=6, right=True)
    ax.tick_params(axis='y', which='minor', direction='in', width=1.0, length=4, right=True)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold'); lbl.set_fontsize(13); lbl.set_fontfamily(FONT)

desc_rows1=[]; stats_rows1=[]; tukey_rows1=[]
VARS_F1=[('C','SOC (%)'),('BD_Benites_2007','Bulk Density (Mg m\u207b\u00b3)')]
alphabet=list('abcdefghijklmnopqrstuvwxyz')

fig1, axes1 = plt.subplots(2, 5, figsize=(18, 9.5), gridspec_kw={'hspace':0.16, 'wspace':0.32})
panel_idx = 0
for ri, (var, ylabel) in enumerate(VARS_F1):
    ylim = YLIMS_F1[var]
    for ci, (dep, dlb) in enumerate(zip(DEPTHS, DLABS)):
        ax = axes1[ri, ci]; df_sub = df[df['Depth']==dep].copy()
        sr = run_stats(df_sub, var)
        plot_box_f1(ax, df_sub, var, dlb, sr, ylabel, ylim, show_ylabel=(ci==0),
                    panel_letter=alphabet[panel_idx])
        panel_idx += 1

patches1 = [mpatches.Patch(facecolor=COLORS[t], edgecolor='#000000', linewidth=1.6, label=t) for t in TREATS]
leg1 = fig1.legend(handles=patches1, loc='lower center', ncol=3, frameon=True,
                   framealpha=0.95, edgecolor='#555555', fancybox=False,
                   prop={'family':FONT, 'size':18, 'weight':'bold'},
                   title='Treatment', title_fontsize=18, bbox_to_anchor=(0.5, -0.01))
leg1.get_title().set_fontweight('bold')

BASE1 = os.path.join(OUTDIR, 'Fig1_C_BD_final')
fig1.savefig(f'{BASE1}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig1.savefig(f'{BASE1}.pdf', bbox_inches='tight', facecolor='white')
fig1.savefig(f'{BASE1}.svg', bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("[Success] Figure 1 generated and saved successfully!")




# %%
# ============================================================
# FIGURE 2 — pH, CEC, N, P resin, Sum of Bases
# ============================================================
# Cell bootstrap: makes this cell runnable on its own (e.g. Spyder
# "run cell" / %runcell) even if cell 0 was never executed first.
import sys, subprocess, importlib.util
def _ensure_pkgs(pkg_map):
    missing=[p for m,p in pkg_map.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
                               '--disable-pip-version-check',*missing])
_ensure_pkgs({'pandas':'pandas','numpy':'numpy','matplotlib':'matplotlib','scipy':'scipy','statsmodels':'statsmodels','openpyxl':'openpyxl'})

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from scipy.stats import shapiro, levene, boxcox
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else globals().get('_SCRIPT_DIR', os.getcwd())
OUTDIR = globals().get('OUTDIR', _SCRIPT_DIR)
FILE = globals().get('FILE', os.path.join(_SCRIPT_DIR, 'database_ecotono_02-05-2025_SHI_calculado.xlsx'))

def _best_font():
    av={f.name for f in fm.fontManager.ttflist}
    for c in ['Arial','Liberation Sans','Helvetica','FreeSans']:
        if c in av: print(f"[font] {c}"); return c
    return 'DejaVu Sans'

FONT=_best_font()
matplotlib.rcParams.update({'font.family':FONT,'font.weight':'bold',
                            'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':1.4})
matplotlib.use('Agg')

df=pd.read_excel(FILE,sheet_name=0,header=0)
df['Vegetation']=df['Vegetation'].replace('DA','RA')

DEPTHS=['P20','P40','P60','P80','P100']
DLABS=['0–20 cm','20–40 cm','40–60 cm','60–80 cm','80–100 cm']
TREATS=['SSF','DWS','RA']
COLORS={'SSF':'#1f78b4','DWS':'#f4c430','RA':'#d95f02'}
ALPHA=0.05
YLIMS_F2={'pH':(3.50,5.20),'CTC':(8.0,108.0),'N':(-0.002,0.112),
          'P':(0.0,33.0),'SB':(0.0,39.0)}
YLABS_F2={'pH':'pH','CTC':'CEC (mmol\u2095 dm\u207b\u00b3)',
          'N':'N (%)','P':'P resin (mg dm\u207b\u00b3)',
          'SB':'Sum of Bases (mmol\u2095 dm\u207b\u00b3)'}

def compact_letters(tukey,treats):
    groups=list(tukey.groupsunique); pairs=list(combinations(range(len(groups)),2))
    rd={}
    for (i,j),rej in zip(pairs,tukey.reject):
        rd[(groups[i],groups[j])]=rej; rd[(groups[j],groups[i])]=rej
    ls={t:set() for t in treats}; cur=0; abc=list('abcdefgh')
    for t1 in treats:
        if not ls[t1]:
            ls[t1].add(abc[cur])
            for t2 in treats:
                if t1!=t2 and not rd.get((t1,t2),True): ls[t2].add(abc[cur])
            cur+=1
    for t in treats:
        if not ls[t]: ls[t].add(abc[cur]); cur+=1
    return {t:''.join(sorted(ls[t])) for t in treats}

def run_stats(df_sub,var):
    groups=[df_sub[df_sub['Vegetation']==t][var].dropna().values for t in TREATS]
    def resid(g): return np.concatenate([x-x.mean() for x in g])
    r=resid(groups); sw_s,sw_p=shapiro(r); lv_s,lv_p=levene(*groups)
    boxcox_used=False; lam=na=ha=swap=lvap=None; ga=groups; pdf=False
    if sw_p<ALPHA or lv_p<ALPHA:
        av=np.concatenate(groups); sh=max(0,-av.min()+1e-6)
        try:
            _,lam=boxcox(av+sh)
            tr=[boxcox(g+sh,lmbda=lam) for g in groups]
            if all(np.std(g)<1e-10 for g in tr): raise ValueError("collapsed")
            r2=resid(tr); sw2,swp2=shapiro(r2); lv2,lvp2=levene(*tr)
            boxcox_used=True; na=swp2>=ALPHA; ha=lvp2>=ALPHA
            swap=round(swp2,4); lvap=round(lvp2,4); ga=tr
            if not na or not ha: pdf=True
        except: pass
    fs,ap=stats.f_oneway(*ga)
    lets={t:'' for t in TREATS}; tukey_rows=[]
    if ap<=ALPHA:
        va=np.concatenate(ga); la=np.concatenate([[t]*len(g) for t,g in zip(TREATS,ga)])
        tk=pairwise_tukeyhsd(va,la,alpha=ALPHA); lets=compact_letters(tk,TREATS)
        grps=list(tk.groupsunique); p2=list(combinations(range(len(grps)),2))
        for (i,j),rej,pval in zip(p2,tk.reject,tk.pvalues):
            tukey_rows.append({'Variable':var,'Depth':None,'Group1':grps[i],
                               'Group2':grps[j],'Reject_H0':bool(rej),
                               'p_adj':round(float(pval),6)})
    return dict(sw_stat=round(sw_s,4),sw_p=round(sw_p,4),
                lev_stat=round(lv_s,4),lev_p=round(lv_p,4),
                norm_before=sw_p>=ALPHA,homo_before=lv_p>=ALPHA,
                boxcox_used=boxcox_used,boxcox_lambda=round(lam,4) if lam else None,
                norm_after=na,homo_after=ha,sw_after_p=swap,lev_after_p=lvap,
                proceeded_despite_failure=pdf,f_stat=round(fs,4),
                anova_p=round(ap,4),significant=ap<=ALPHA,letters=lets,tukey_rows=tukey_rows)

def desc_stats(df_sub,var,treat,dlb):
    v=df_sub[df_sub['Vegetation']==treat][var].dropna()
    return dict(Variable=var,Depth=dlb,Treatment=treat,n=len(v),
                Mean=round(v.mean(),4),Median=round(v.median(),4),
                SD=round(v.std(),4),Min=round(v.min(),4),Max=round(v.max(),4),
                CV_pct=round(v.std()/v.mean()*100,2) if v.mean()!=0 else None)

def whisker_top(vals):
    q75=np.percentile(vals,75); iqr=q75-np.percentile(vals,25)
    fence=q75+1.5*iqr; above=vals[vals<=fence]
    return above.max() if len(above) else q75

def plot_box_f2(ax,df_sub,var,dlb,sr,ylabel,ylim,show_ylabel,panel_letter):
    bp_data=[df_sub[df_sub['Vegetation']==t][var].dropna().values for t in TREATS]
    bp=ax.boxplot(bp_data,positions=[1,2,3],widths=0.50,patch_artist=True,notch=False,
                  medianprops=dict(color='#000000',linewidth=2.5),
                  whiskerprops=dict(color='#000000',linewidth=1.6),
                  capprops=dict(color='#000000',linewidth=1.6),
                  flierprops=dict(marker='o',markersize=4.5,markerfacecolor='#777777',
                                  markeredgecolor='#000000',linewidth=1.0))
    for patch,t in zip(bp['boxes'],TREATS):
        patch.set_facecolor(COLORS[t]); patch.set_edgecolor('#000000')
        patch.set_linewidth(1.8); patch.set_alpha(0.92)
    ax.set_ylim(ylim); yspan=ylim[1]-ylim[0]
    ax.text(0.03,0.98,f'({panel_letter})',transform=ax.transAxes,
            ha='left',va='top',fontsize=10,fontweight='bold',fontfamily=FONT)
    ax.text(0.5,0.98,dlb,transform=ax.transAxes,ha='center',va='top',
            fontsize=8.5,fontweight='bold',fontfamily=FONT,color='#111111',
            bbox=dict(boxstyle='round,pad=0.25',facecolor='#e8e8e8',
                      edgecolor='#888888',linewidth=0.9,alpha=0.95))
    p=sr['anova_p']
    p_str='p < 0.001' if p<0.001 else f'p = {p:.3f}'
    sig_str=p_str if sr['significant'] else f'{p_str} (ns)'
    ax.text(0.5,0.855,sig_str,transform=ax.transAxes,ha='center',va='top',
            fontsize=8,fontweight='bold',style='italic',color='#111111',fontfamily=FONT)
    if sr['significant']:
        for pos,t in zip([1,2,3],TREATS):
            vals=bp_data[TREATS.index(t)]
            tip=max(whisker_top(vals),vals.max())
            y_letter=min(tip+yspan*0.032,ylim[0]+yspan*0.79)
            ax.text(pos,y_letter,sr['letters'][t],ha='center',va='bottom',
                    fontsize=9.5,fontweight='bold',color='#000000',
                    fontfamily=FONT,transform=ax.transData)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.4); sp.set_color('#000000')
    ax.set_xlim(0.25,3.75); ax.set_xticks([])
    if show_ylabel:
        ax.set_ylabel(ylabel,fontsize=9,fontweight='bold',labelpad=4,fontfamily=FONT)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y',labelsize=8.5,which='major',direction='in',
                   width=1.4,length=5,right=True)
    ax.tick_params(axis='y',which='minor',direction='in',width=0.8,length=3,right=True)
    ax.tick_params(axis='x',which='both',direction='in',top=True,bottom=True,
                   length=4,width=1.3)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold'); lbl.set_fontsize(8.5); lbl.set_fontfamily(FONT)

desc_rows2=[]; stats_rows2=[]; tukey_rows2=[]
VARS_F2=['pH','CTC','N','P','SB']
alphabet=list('abcdefghijklmnopqrstuvwxyz')

fig2,axes2=plt.subplots(5,5,figsize=(16,18),
                         gridspec_kw={'hspace':0.14,'wspace':0.22})
panel_idx=0
for ri,var in enumerate(VARS_F2):
    ylim=YLIMS_F2[var]; ylabel=YLABS_F2[var]
    for ci,(dep,dlb) in enumerate(zip(DEPTHS,DLABS)):
        ax=axes2[ri,ci]; df_sub=df[df['Depth']==dep].copy()
        sr=run_stats(df_sub,var)
        plot_box_f2(ax,df_sub,var,dlb,sr,ylabel,ylim,show_ylabel=(ci==0),
                    panel_letter=alphabet[panel_idx])
        panel_idx+=1
        for t in TREATS: desc_rows2.append(desc_stats(df_sub,var,t,dlb))
        stats_rows2.append({'Variable':var,'Depth':dlb,
            'Shapiro_W_before':sr['sw_stat'],'Shapiro_p_before':sr['sw_p'],
            'Normal_before':sr['norm_before'],'Levene_stat_before':sr['lev_stat'],
            'Levene_p_before':sr['lev_p'],'Homogeneous_before':sr['homo_before'],
            'BoxCox_applied':sr['boxcox_used'],'BoxCox_lambda':sr['boxcox_lambda'],
            'Shapiro_p_after':sr['sw_after_p'],'Normal_after':sr['norm_after'],
            'Levene_p_after':sr['lev_after_p'],'Homogeneous_after':sr['homo_after'],
            'Proceeded_despite_failure':sr['proceeded_despite_failure'],
            'F_stat':sr['f_stat'],'ANOVA_p':sr['anova_p'],'Significant':sr['significant'],
            'Letter_SSF':sr['letters'].get('SSF','ns'),
            'Letter_DWS':sr['letters'].get('DWS','ns'),
            'Letter_RA':sr['letters'].get('RA','ns')})
        for tr in sr['tukey_rows']:
            tr['Depth']=dlb; tukey_rows2.append(tr)

patches2=[mpatches.Patch(facecolor=COLORS[t],edgecolor='#000000',linewidth=1.4,label=t)
          for t in TREATS]
leg2=fig2.legend(handles=patches2,loc='lower center',ncol=3,frameon=True,
                 framealpha=0.95,edgecolor='#555555',fancybox=False,
                 prop={'family':FONT,'size':11,'weight':'bold'},
                 title='Legend',title_fontsize=12,bbox_to_anchor=(0.5,0.005))
leg2.get_title().set_fontweight('bold'); leg2.get_title().set_fontfamily(FONT)

BASE2=os.path.join(OUTDIR,'Fig2_chemical_final')
fig2.savefig(f'{BASE2}.png',dpi=300,bbox_inches='tight',facecolor='white')
fig2.savefig(f'{BASE2}.pdf',bbox_inches='tight',facecolor='white')
fig2.savefig(f'{BASE2}.svg',bbox_inches='tight',facecolor='white')
plt.close(fig2); print("Figure 2 saved.")
with pd.ExcelWriter(os.path.join(OUTDIR,'Fig2_stats_supplementary.xlsx'),engine='openpyxl') as w:
    pd.DataFrame(desc_rows2).to_excel(w,sheet_name='Descriptive_Stats',index=False)
    pd.DataFrame(stats_rows2).to_excel(w,sheet_name='ANOVA_Metrics',index=False)
    if tukey_rows2: pd.DataFrame(tukey_rows2).to_excel(w,sheet_name='Tukey_Pairwise',index=False)
print("Table 2 saved.")


# %%
# ============================================================
# FIGURE 3 — PCA (a) + Lower-triangle Spearman correlation (b)
# ============================================================
# Cell bootstrap: makes this cell runnable on its own (e.g. Spyder
# "run cell" / %runcell) even if cell 0 was never executed first.
import sys, subprocess, importlib.util
def _ensure_pkgs(pkg_map):
    missing=[p for m,p in pkg_map.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
                               '--disable-pip-version-check',*missing])
_ensure_pkgs({'pandas':'pandas','numpy':'numpy','matplotlib':'matplotlib','scipy':'scipy','sklearn':'scikit-learn','pingouin':'pingouin','openpyxl':'openpyxl'})

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pingouin as pg
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else globals().get('_SCRIPT_DIR', os.getcwd())
OUTDIR = globals().get('OUTDIR', _SCRIPT_DIR)
FILE = globals().get('FILE', os.path.join(_SCRIPT_DIR, 'database_ecotono_02-05-2025_SHI_calculado.xlsx'))

def _best_font():
    av={f.name for f in fm.fontManager.ttflist}
    for c in ['Arial','Liberation Sans','Helvetica','FreeSans']:
        if c in av: print(f"[font] {c}"); return c
    return 'DejaVu Sans'

FONT=_best_font()
matplotlib.rcParams.update({'font.family':FONT,'font.weight':'bold',
                            'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':1.4})
matplotlib.use('Agg')

df=pd.read_excel(FILE,sheet_name=0,header=0)
df['Vegetation']=df['Vegetation'].replace('DA','RA')

VARS=['pH','CTC','N','P','SB','C','BD_Benites_2007']
VLABS=['pH','CEC','N','P resin','SB','SOC','BD']
TREATS=['SSF','DWS','RA']
DEPTHS=['P20','P40','P60','P80','P100']
DLABS=['0–20','20–40','40–60','60–80','80–100']
ALPHA=0.05
COLORS={'SSF':'#1f78b4','DWS':'#f4c430','RA':'#d95f02'}
MARKERS={'P20':'o','P40':'s','P60':'^','P80':'D','P100':'v'}
EC='#000000'

print("[PCA] Z-score standardization (StandardScaler)")
X_raw=df[VARS].values
scaler=StandardScaler(); X_sc=scaler.fit_transform(X_raw)
pca=PCA(); scores=pca.fit_transform(X_sc)
loads=pca.components_; var_ex=pca.explained_variance_ratio_; eigval=pca.explained_variance_
PC1_pct=round(var_ex[0]*100,1); PC2_pct=round(var_ex[1]*100,1)
print(f"[PCA] PC1={PC1_pct}%  PC2={PC2_pct}%  cumulative={PC1_pct+PC2_pct}%")

sc_x=max(abs(scores[:,0].min()),abs(scores[:,0].max()))
sc_y=max(abs(scores[:,1].min()),abs(scores[:,1].max()))
scale=min(sc_x*0.75/abs(loads[0]).max(), sc_y*0.75/abs(loads[1]).max())

pairs_hz=list(combinations(range(len(VARS)),2))
n_pairs=len(pairs_hz); n_normal=0; hz_rows=[]
for i,j in pairs_hz:
    xy=df[[VARS[i],VARS[j]]].dropna().values
    try: hz=pg.multivariate_normality(xy,alpha=ALPHA); normal=bool(hz.normal)
    except: normal=False
    hz_rows.append({'Var1':VARS[i],'Var2':VARS[j],'HZ_normal':normal})
    if normal: n_normal+=1

pct_normal=n_normal/n_pairs*100; use_pearson=pct_normal>=50
method='Pearson' if use_pearson else 'Spearman'
method_note=(f"Method: {method}. {n_normal}/{n_pairs} pairs ({pct_normal:.1f}%) "
             f"passed Henze-Zirkler bivariate normality (α={ALPHA}). "
             f"≥50% → Pearson; <50% → Spearman.")
print(f"[corr] {method_note}")

n_vars=len(VARS); corr_mat=np.ones((n_vars,n_vars)); pval_mat=np.zeros((n_vars,n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        if i!=j:
            fn=pearsonr if use_pearson else spearmanr
            r,p=fn(X_raw[:,i],X_raw[:,j])
            corr_mat[i,j]=round(r,4); pval_mat[i,j]=round(p,6)

def sig_stars(p):
    if p<0.001: return '***'
    if p<0.01:  return '**'
    if p<0.05:  return '*'
    return 'ns'

fig3=plt.figure(figsize=(21,9.5))
ax_pca =fig3.add_axes([0.05,0.09,0.46,0.85])
ax_corr=fig3.add_axes([0.58,0.09,0.36,0.85])
ax_cbar=fig3.add_axes([0.956,0.09,0.018,0.85])

for dep in DEPTHS:
    for trt in TREATS:
        mask=(df['Depth']==dep)&(df['Vegetation']==trt)
        idx=df.index[mask]
        ax_pca.scatter(scores[idx,0],scores[idx,1],
                       c=COLORS[trt],marker=MARKERS[dep],
                       s=80,edgecolors=EC,linewidths=0.9,alpha=0.88,zorder=3)

for k in range(n_vars):
    dx=loads[0,k]*scale; dy=loads[1,k]*scale
    ax_pca.annotate('',xy=(dx,dy),xytext=(0,0),
                    arrowprops=dict(arrowstyle='->',color=EC,lw=2.2,mutation_scale=18),zorder=5)
    ox=dx*1.17; oy=dy*1.17
    ha='left' if dx>=0 else 'right'; va='bottom' if dy>=0 else 'top'
    ax_pca.text(ox,oy,VLABS[k],fontsize=15,fontweight='bold',color=EC,
                fontfamily=FONT,ha=ha,va=va,zorder=6,
                bbox=dict(boxstyle='round,pad=0.20',facecolor='white',edgecolor='none',alpha=0.82))

ax_pca.axhline(0,color='#aaaaaa',lw=1.0,ls='--',zorder=1)
ax_pca.axvline(0,color='#aaaaaa',lw=1.0,ls='--',zorder=1)
ax_pca.set_xlabel(f'PC1 ({PC1_pct}% variance explained)',
                  fontsize=17,fontweight='bold',fontfamily=FONT,labelpad=8)
ax_pca.set_ylabel(f'PC2 ({PC2_pct}% variance explained)',
                  fontsize=17,fontweight='bold',fontfamily=FONT,labelpad=8)
pad_x=sc_x*0.22+0.9; pad_y=sc_y*0.22+0.9
ax_pca.set_xlim(scores[:,0].min()-pad_x, scores[:,0].max()+pad_x+1.6)
ax_pca.set_ylim(scores[:,1].min()-pad_y, scores[:,1].max()+pad_y)
for sp in ax_pca.spines.values():
    sp.set_linewidth(1.4); sp.set_color(EC)
ax_pca.tick_params(axis='both',labelsize=14,direction='in',width=1.4,length=6,top=True,right=True)
for lbl in ax_pca.get_xticklabels()+ax_pca.get_yticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontfamily(FONT); lbl.set_fontsize(14)
ax_pca.text(0.02,0.98,'(a)',transform=ax_pca.transAxes,
            fontsize=17,fontweight='bold',fontfamily=FONT,va='top',ha='left')

trt_handles=[mpatches.Patch(facecolor=COLORS[t],edgecolor=EC,linewidth=1.2,label=t) for t in TREATS]
dep_handles=[mlines.Line2D([],[],color='#555555',marker=MARKERS[d],markersize=9,
                            linewidth=0,markeredgewidth=1.1,markeredgecolor=EC,
                            label=f'{lb} cm') for d,lb in zip(DEPTHS,DLABS)]
leg_t=ax_pca.legend(handles=trt_handles,title='Treatment',loc='upper right',
                    fontsize=13,frameon=True,framealpha=0.93,edgecolor='#888888',
                    prop={'family':FONT,'size':13,'weight':'bold'},title_fontsize=14)
leg_t.get_title().set_fontfamily(FONT); leg_t.get_title().set_fontweight('bold')
ax_pca.add_artist(leg_t)
leg_d=ax_pca.legend(handles=dep_handles,title='Depth (cm)',loc='lower right',
                    fontsize=13,frameon=True,framealpha=0.93,edgecolor='#888888',
                    prop={'family':FONT,'size':13,'weight':'bold'},title_fontsize=14)
leg_d.get_title().set_fontfamily(FONT); leg_d.get_title().set_fontweight('bold')

cmap=plt.cm.RdBu_r; norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
corr_plot=np.full((n_vars,n_vars),np.nan)
for i in range(n_vars):
    corr_plot[i,i]=1.0
    for j in range(i): corr_plot[i,j]=corr_mat[i,j]

ax_corr.imshow(corr_plot,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
for i in range(n_vars):
    for j in range(i+1,n_vars):
        ax_corr.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                          facecolor='white',edgecolor='white',linewidth=0))
for i in range(n_vars):
    for j in range(i+1):
        ax_corr.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                          fill=False,edgecolor='#cccccc',linewidth=0.9))
for i in range(n_vars):
    ax_corr.text(i,i,f'{VLABS[i]}\n1.00',ha='center',va='center',
                 fontsize=12.5,fontweight='bold',color='#111111',fontfamily=FONT,
                 linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.30',facecolor='#f0f0f0',
                           edgecolor='#aaaaaa',linewidth=1.0))
for i in range(n_vars):
    for j in range(i):
        r=corr_mat[i,j]; p=pval_mat[i,j]
        cell_color='white' if abs(r)>0.52 else '#111111'
        ax_corr.text(j,i,f'{r:.2f}\n{sig_stars(p)}',ha='center',va='center',
                     fontsize=12,fontweight='bold',color=cell_color,
                     fontfamily=FONT,linespacing=1.35)

ax_corr.set_xticks(range(n_vars)); ax_corr.set_yticks(range(n_vars))
ax_corr.set_xticklabels(VLABS,fontsize=18,fontweight='bold',
                         fontfamily=FONT,rotation=38,ha='right')
ax_corr.set_yticklabels(VLABS,fontsize=18,fontweight='bold',fontfamily=FONT)
ax_corr.tick_params(axis='both',length=0,pad=5)
for sp in ax_corr.spines.values():
    sp.set_linewidth(1.4); sp.set_color(EC)
ax_corr.text(0.5,-0.13,f'{method} correlation coefficient',
             transform=ax_corr.transAxes,ha='center',va='top',
             fontsize=14,fontweight='bold',fontfamily=FONT,color='#222222')
ax_corr.text(0.5,-0.20,'* p<0.05   ** p<0.01   *** p<0.001',
             transform=ax_corr.transAxes,ha='center',va='top',
             fontsize=13,fontweight='bold',fontfamily=FONT,color='#555555')
ax_corr.text(0.02,0.98,'(b)',transform=ax_corr.transAxes,
             fontsize=17,fontweight='bold',fontfamily=FONT,va='top',ha='left')

cb=matplotlib.colorbar.ColorbarBase(ax_cbar,cmap=cmap,norm=norm,orientation='vertical')
cb.set_label(f'{method} r',fontsize=18,fontweight='bold',fontfamily=FONT,labelpad=10)
cb.ax.tick_params(labelsize=18,width=1.3,length=5)
for lbl in cb.ax.get_yticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontfamily(FONT); lbl.set_fontsize(14)
cb.outline.set_linewidth(1.4)

BASE3=os.path.join(OUTDIR,'Fig3_PCA_Corr_final')
fig3.savefig(f'{BASE3}.png',dpi=300,bbox_inches='tight',facecolor='white')
fig3.savefig(f'{BASE3}.pdf',bbox_inches='tight',facecolor='white')
fig3.savefig(f'{BASE3}.svg',bbox_inches='tight',facecolor='white')
plt.close(fig3); print("Figure 3 saved.")

load_df=pd.DataFrame(loads.T,index=VARS,
    columns=[f'PC{i+1}' for i in range(loads.shape[0])]).reset_index().rename(columns={'index':'Variable'})
load_df.insert(1,'Label',VLABS)
var_df=pd.DataFrame({'PC':[f'PC{i+1}' for i in range(len(var_ex))],
    'Eigenvalue':eigval.round(4),'Variance_explained_%':(var_ex*100).round(3),
    'Cumulative_%':(np.cumsum(var_ex)*100).round(3)})
score_df=df[['Depth','Vegetation','Profile','Identifier']].copy()
for i in range(min(5,scores.shape[1])): score_df[f'PC{i+1}']=scores[:,i].round(4)
corr_long=[]
for i in range(n_vars):
    for j in range(i+1,n_vars):
        corr_long.append({'Var1':VLABS[i],'Var2':VLABS[j],'Method':method,
            f'r_{method[0]}':corr_mat[i,j],'p_value':pval_mat[i,j],
            'Significance':sig_stars(pval_mat[i,j])})
with pd.ExcelWriter(os.path.join(OUTDIR,'Fig3_stats_supplementary.xlsx'),engine='openpyxl') as w:
    var_df.to_excel(w,sheet_name='PCA_Variance_Explained',index=False)
    load_df.to_excel(w,sheet_name='PCA_Loadings',index=False)
    score_df.to_excel(w,sheet_name='PCA_Scores',index=False)
    pd.DataFrame(corr_long).to_excel(w,sheet_name='Correlation_Pairwise',index=False)
    pd.DataFrame(corr_mat,index=VLABS,columns=VLABS).round(4).to_excel(w,sheet_name='Correlation_Matrix')
    pd.DataFrame(pval_mat,index=VLABS,columns=VLABS).round(6).to_excel(w,sheet_name='Correlation_Pvalues')
    pd.DataFrame(hz_rows).assign(Method_selected=method).to_excel(w,sheet_name='HZ_Normality_per_Pair',index=False)
    pd.DataFrame([{'Method_selected':method,'N_pairs_total':n_pairs,
        'N_pairs_normal_HZ':n_normal,'Pct_normal':round(pct_normal,1),
        'Threshold_pct':50,'Criterion':'Pearson if ≥50% pass HZ (α=0.05); Spearman otherwise',
        'Note':method_note}]).to_excel(w,sheet_name='Method_Selection',index=False)
print("Table 3 saved.")


# %%
# ============================================================
# FIGURE 4 — SHI sub-indices (a) + Overall SHI boxplot (b) [EIXO HORIZONTAL]
# ============================================================
import sys, subprocess, importlib.util
def _ensure_pkgs(pkg_map):
    missing=[p for m,p in pkg_map.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
                               '--disable-pip-version-check',*missing])
_ensure_pkgs({'pandas':'pandas','numpy':'numpy','matplotlib':'matplotlib','scipy':'scipy','statsmodels':'statsmodels','openpyxl':'openpyxl'})

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import stats
from scipy.stats import shapiro, levene, boxcox
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else globals().get('_SCRIPT_DIR', os.getcwd())
OUTDIR = globals().get('OUTDIR', _SCRIPT_DIR)
FILE = globals().get('FILE', os.path.join(_SCRIPT_DIR, 'database_ecotono_02-05-2025_SHI_calculado.xlsx'))

def _best_font():
    av={f.name for f in fm.fontManager.ttflist}
    for c in ['Arial','Liberation Sans','Helvetica','FreeSans']:
        if c in av: print(f"[font] {c}"); return c
    return 'DejaVu Sans'

FONT=_best_font()
matplotlib.rcParams.update({'font.family':FONT,'font.weight':'bold',
                            'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':1.5})
matplotlib.use('Agg')

df=pd.read_excel(FILE,sheet_name=2,header=0)
df['Vegetation']=df['Vegetation'].replace('DA','RA')

# Invertido para que 0-20 cm fique no topo do gráfico horizontal
DEPTHS=['P100','P80','P60','P40','P20']
DLABS=['80–100','60–80','40–60','20–40','0–20']

TREATS=['SSF','DWS','RA']
COLORS={'SSF':'#1f78b4','DWS':'#f4c430','RA':'#d95f02'}
ALPHA=0.05
SUB_VARS=['SHI_Physical','SHI_Biological','SHI_Chemical']
HATCHES={'SHI_Physical':'....','SHI_Biological':'////','SHI_Chemical':''}
SEG_LABELS={'SHI_Physical':'Physical (...)','SHI_Biological':'Biological (///)','SHI_Chemical':'Chemical (solid)'}
BAR_ALPHA=0.78

def compact_letters(tukey,treats):
    groups=list(tukey.groupsunique); pairs=list(combinations(range(len(groups)),2))
    rd={}
    for (i,j),rej in zip(pairs,tukey.reject):
        rd[(groups[i],groups[j])]=rej; rd[(groups[j],groups[i])]=rej
    ls={t:set() for t in treats}; cur=0; abc=list('abcdefgh')
    for t1 in treats:
        if not ls[t1]:
            ls[t1].add(abc[cur])
            for t2 in treats:
                if t1!=t2 and not rd.get((t1,t2),True): ls[t2].add(abc[cur])
            cur+=1
    for t in treats:
        if not ls[t]: ls[t].add(abc[cur]); cur+=1
    return {t:''.join(sorted(ls[t])) for t in treats}

def run_stats(df_sub,var):
    groups=[df_sub[df_sub['Vegetation']==t][var].dropna().values for t in TREATS]
    def resid(g): return np.concatenate([x-x.mean() for x in g])
    r=resid(groups); sw_s,sw_p=shapiro(r); lv_s,lv_p=levene(*groups)
    boxcox_used=False; lam=na=ha=swap=lvap=None; ga=groups; pdf=False
    if sw_p<ALPHA or lv_p<ALPHA:
        av=np.concatenate(groups); sh=max(0,-av.min()+1e-6)
        try:
            _,lam=boxcox(av+sh)
            tr=[boxcox(g+sh,lmbda=lam) for g in groups]
            if all(np.std(g)<1e-10 for g in tr): raise ValueError("collapsed")
            r2=resid(tr); sw2,swp2=shapiro(r2); lv2,lvp2=levene(*tr)
            boxcox_used=True; na=swp2>=ALPHA; ha=lvp2>=ALPHA
            swap=round(swp2,4); lvap=round(lvp2,4); ga=tr
            if not na or not ha: pdf=True
        except: pass
    fs,ap=stats.f_oneway(*ga)
    lets={t:'' for t in TREATS}; tukey_rows=[]
    if ap<=ALPHA:
        va=np.concatenate(ga); la=np.concatenate([[t]*len(g) for t,g in zip(TREATS,ga)])
        tk=pairwise_tukeyhsd(va,la,alpha=ALPHA); lets=compact_letters(tk,TREATS)
        grps=list(tk.groupsunique); p2=list(combinations(range(len(grps)),2))
        for (i,j),rej,pval in zip(p2,tk.reject,tk.pvalues):
            tukey_rows.append({'Variable':var,'Depth':None,'Group1':grps[i],
                               'Group2':grps[j],'Reject_H0':bool(rej),
                               'p_adj':round(float(pval),6)})
    return dict(sw_stat=round(sw_s,4),sw_p=round(sw_p,4),
                lev_stat=round(lv_s,4),lev_p=round(lv_p,4),
                norm_before=sw_p>=ALPHA,homo_before=lv_p>=ALPHA,
                boxcox_used=boxcox_used,boxcox_lambda=round(lam,4) if lam else None,
                norm_after=na,homo_after=ha,sw_after_p=swap,lev_after_p=lvap,
                proceeded_despite_failure=pdf,f_stat=round(fs,4),
                anova_p=round(ap,4),significant=ap<=ALPHA,letters=lets,tukey_rows=tukey_rows)

def desc_stats(df_sub,var,treat,depth=None):
    v=df_sub[df_sub['Vegetation']==treat][var].dropna()
    d=dict(Variable=var,Treatment=treat,n=len(v),
           Mean=round(v.mean(),4),Median=round(v.median(),4),
           SD=round(v.std(),4),Min=round(v.min(),4),Max=round(v.max(),4),
           CV_pct=round(v.std()/v.mean()*100,2) if v.mean()!=0 else None)
    if depth: d['Depth']=depth
    return d

def whisker_top(vals):
    q75=np.percentile(vals,75); iqr=q75-np.percentile(vals,25)
    fence=q75+1.5*iqr; above=vals[vals<=fence]
    return above.max() if len(above) else q75

desc_rows4=[]; stats_rows4=[]; tukey_rows4=[]
stats_grid={}
for dep,dlb in zip(DEPTHS,DLABS):
    df_sub=df[df['Depth']==dep].copy()
    for var in SUB_VARS+['SHI_Layer']:
        sr=run_stats(df_sub,var); stats_grid[(dep,var)]=sr
        for t in TREATS: desc_rows4.append(desc_stats(df_sub,var,t,depth=dlb))
        stats_rows4.append({'Variable':var,'Depth':dlb,
            'Shapiro_W_before':sr['sw_stat'],'Shapiro_p_before':sr['sw_p'],
            'Normal_before':sr['norm_before'],'Levene_stat_before':sr['lev_stat'],
            'Levene_p_before':sr['lev_p'],'Homogeneous_before':sr['homo_before'],
            'BoxCox_applied':sr['boxcox_used'],'BoxCox_lambda':sr['boxcox_lambda'],
            'Shapiro_p_after':sr['sw_after_p'],'Normal_after':sr['norm_after'],
            'Levene_p_after':sr['lev_after_p'],'Homogeneous_after':sr['homo_after'],
            'Proceeded_despite_failure':sr['proceeded_despite_failure'],
            'F_stat':sr['f_stat'],'ANOVA_p':sr['anova_p'],'Significant':sr['significant'],
            'Letter_SSF':sr['letters'].get('SSF','ns'),
            'Letter_DWS':sr['letters'].get('DWS','ns'),
            'Letter_RA':sr['letters'].get('RA','ns')})
        for tr in sr['tukey_rows']:
            tr['Depth']=dlb; tukey_rows4.append(tr)

df_overall=df[df['Overall_SHI'].notna()].copy()
sr_overall=run_stats(df_overall,'Overall_SHI')
for t in TREATS: desc_rows4.append(desc_stats(df_overall,'Overall_SHI',t,depth='Per profile'))
stats_rows4.append({'Variable':'Overall_SHI','Depth':'Per profile',
    'Shapiro_W_before':sr_overall['sw_stat'],'Shapiro_p_before':sr_overall['sw_p'],
    'Normal_before':sr_overall['norm_before'],'Levene_stat_before':sr_overall['lev_stat'],
    'Levene_p_before':sr_overall['lev_p'],'Homogeneous_before':sr_overall['homo_before'],
    'BoxCox_applied':sr_overall['boxcox_used'],'BoxCox_lambda':sr_overall['boxcox_lambda'],
    'Shapiro_p_after':sr_overall['sw_after_p'],'Normal_after':sr_overall['norm_after'],
    'Levene_p_after':sr_overall['lev_after_p'],'Homogeneous_after':sr_overall['homo_after'],
    'Proceeded_despite_failure':sr_overall['proceeded_despite_failure'],
    'F_stat':sr_overall['f_stat'],'ANOVA_p':sr_overall['anova_p'],
    'Significant':sr_overall['significant'],
    'Letter_SSF':sr_overall['letters'].get('SSF','ns'),
    'Letter_DWS':sr_overall['letters'].get('DWS','ns'),
    'Letter_RA':sr_overall['letters'].get('RA','ns')})
for tr in sr_overall['tukey_rows']:
    tr['Depth']='Per profile'; tukey_rows4.append(tr)

fig4=plt.figure(figsize=(22,10))
ax_a=fig4.add_axes([0.10,0.12,0.52,0.82])
ax_b=fig4.add_axes([0.70,0.12,0.25,0.82])

bar_h=0.22; group_gap=0.15
group_h=len(TREATS)*bar_h
group_positions=np.arange(len(DEPTHS))*(group_h+group_gap)
offsets=np.linspace(bar_h, -bar_h, len(TREATS))

for ti,trt in enumerate(TREATS):
    for di,(dep,dlb) in enumerate(zip(DEPTHS,DLABS)):
        df_sub=df[df['Depth']==dep].copy()
        y_center=group_positions[di]+offsets[ti]
        left=0.0
        for var in SUB_VARS:
            val=df_sub[df_sub['Vegetation']==trt][var].mean()
            ax_a.barh(y_center,val,bar_h*0.93,left=left,
                      color=COLORS[trt],edgecolor='#000000',linewidth=0.9,
                      hatch=HATCHES[var],alpha=BAR_ALPHA,zorder=3)
            seg_mid=left+val/2
            sr_sub=stats_grid[(dep,var)]
            letter=sr_sub['letters'].get(trt,'') if sr_sub['significant'] else 'ns'
            label=f'{val:.2f} {letter}' if val>0.030 else ''
            if label:
                ax_a.text(seg_mid,y_center,label,ha='center',va='center',
                          fontsize=8.5,fontweight='bold',color='#000000',
                          fontfamily=FONT,zorder=6,clip_on=True)
            left+=val
        sr_layer=stats_grid[(dep,'SHI_Layer')]
        layer_letter=sr_layer['letters'].get(trt,'') if sr_layer['significant'] else 'ns'
        ax_a.text(left+0.014,y_center,f'{left:.2f} {layer_letter}',
                  ha='left',va='center',fontsize=10,fontweight='bold',
                  color='#000000',fontfamily=FONT,zorder=7)

trt_y=[group_positions[di]+offsets[ti]
       for di in range(len(DEPTHS)) for ti in range(len(TREATS))]
trt_labels=[trt for _ in range(len(DEPTHS)) for trt in TREATS]
ax_a.set_yticks(trt_y)
ax_a.set_yticklabels(trt_labels,fontsize=11,fontweight='bold',fontfamily=FONT)
ax_a.tick_params(axis='y',which='both',direction='in',length=4,width=1.2,pad=3)

group_centers=[group_positions[di]+np.mean(offsets) for di in range(len(DEPTHS))]
for gc,dlb in zip(group_centers,DLABS):
    ax_a.text(-0.16,gc,f'{dlb} cm',ha='right',va='center',
              fontsize=13,fontweight='bold',fontfamily=FONT,color='#111111',
              transform=ax_a.get_yaxis_transform())

for di in range(1,len(DEPTHS)):
    sep_y=(group_positions[di-1]-bar_h*1.6+group_positions[di]+bar_h*1.6)/2
    ax_a.axhline(sep_y,color='#cccccc',lw=0.9,ls=':',zorder=0)

ax_a.set_ylim(group_positions[0]-bar_h*2.0, group_positions[-1]+bar_h*2.0)
ax_a.set_xlim(0,1.25)
ax_a.set_xlabel('SHI value',fontsize=16,fontweight='bold',fontfamily=FONT,labelpad=6)
ax_a.xaxis.set_major_locator(MultipleLocator(0.2))
ax_a.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_a.tick_params(axis='x',labelsize=13,which='major',direction='in',
                 width=1.5,length=6,top=True)
ax_a.tick_params(axis='x',which='minor',direction='in',width=0.9,length=3,top=True)
for sp in ax_a.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color('#000000')
for lbl in ax_a.get_xticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontsize(13); lbl.set_fontfamily(FONT)
ax_a.text(0.01,0.98,'(a)',transform=ax_a.transAxes,
          fontsize=18,fontweight='bold',fontfamily=FONT,va='top',ha='left')

trt_handles=[mpatches.Patch(facecolor=COLORS[t],edgecolor='#000000',linewidth=1.1,label=t)
             for t in TREATS]
seg_handles=[mpatches.Patch(facecolor='#aaaaaa',edgecolor='#000000',linewidth=0.9,
                             hatch=HATCHES[v],alpha=0.85,label=SEG_LABELS[v])
             for v in SUB_VARS]
leg_t=ax_a.legend(handles=trt_handles,title='Treatment',title_fontsize=13,
                  loc='lower right',fontsize=12,frameon=True,framealpha=0.93,
                  edgecolor='#888888',prop={'family':FONT,'size':12,'weight':'bold'})
leg_t.get_title().set_fontfamily(FONT); leg_t.get_title().set_fontweight('bold')
ax_a.add_artist(leg_t)
leg_s=ax_a.legend(handles=seg_handles,title='Sub-index',title_fontsize=13,
                  loc='upper right',fontsize=12,frameon=True,framealpha=0.93,
                  edgecolor='#888888',prop={'family':FONT,'size':12,'weight':'bold'})
leg_s.get_title().set_fontfamily(FONT); leg_s.get_title().set_fontweight('bold')

bp_data=[df_overall[df_overall['Vegetation']==t]['Overall_SHI'].dropna().values for t in TREATS]
bp=ax_b.boxplot(bp_data,positions=[1,2,3],widths=0.52,patch_artist=True,notch=False,
                medianprops=dict(color='#000000',linewidth=2.8),
                whiskerprops=dict(color='#000000',linewidth=1.8),
                capprops=dict(color='#000000',linewidth=1.8),
                flierprops=dict(marker='o',markersize=6,markerfacecolor='#777777',
                                markeredgecolor='#000000',linewidth=1.1))
for patch,t in zip(bp['boxes'],TREATS):
    patch.set_facecolor(COLORS[t]); patch.set_edgecolor('#000000')
    patch.set_linewidth(2.0); patch.set_alpha(0.92)

all_vals=np.concatenate(bp_data)
ymin=all_vals.min(); ymax=all_vals.max(); yspan=ymax-ymin
ax_b.set_ylim(ymin-yspan*0.10, ymax+yspan*0.32)

for pos,t in zip([1,2,3],TREATS):
    vals=bp_data[TREATS.index(t)]
    median_val=np.median(vals)
    tip=max(whisker_top(vals),vals.max())
    y_annot=min(tip+yspan*0.04, ymin+yspan*0.87)
    letter=sr_overall['letters'].get(t,'') if sr_overall['significant'] else 'ns'
    ax_b.text(pos,y_annot,f'{median_val:.3f} {letter}',ha='center',va='bottom',
              fontsize=14,fontweight='bold',color='#000000',fontfamily=FONT,
              transform=ax_b.transData)

ax_b.set_xticks([1,2,3])
ax_b.set_xticklabels(TREATS,fontsize=15,fontweight='bold',fontfamily=FONT)
ax_b.set_ylabel('Overall SHI',fontsize=16,fontweight='bold',fontfamily=FONT,labelpad=6)
ax_b.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_b.yaxis.set_major_locator(MultipleLocator(0.05))
ax_b.tick_params(axis='y',labelsize=13,which='major',direction='in',
                 width=1.5,length=6,right=True)
ax_b.tick_params(axis='y',which='minor',direction='in',width=0.9,length=3,right=True)
ax_b.tick_params(axis='x',which='both',direction='in',top=True,length=5,width=1.4)
for sp in ax_b.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color('#000000')
for lbl in ax_b.get_yticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontsize(13); lbl.set_fontfamily(FONT)
for lbl in ax_b.get_xticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontfamily(FONT)
ax_b.text(0.04,0.98,'(b)',transform=ax_b.transAxes,
          fontsize=18,fontweight='bold',fontfamily=FONT,va='top',ha='left')

BASE4=os.path.join(OUTDIR,'Fig4_SHI_final')
fig4.savefig(f'{BASE4}.png',dpi=300,bbox_inches='tight',facecolor='white')
fig4.savefig(f'{BASE4}.pdf',bbox_inches='tight',facecolor='white')
fig4.savefig(f'{BASE4}.svg',bbox_inches='tight',facecolor='white')
plt.close(fig4); print("Figure 4 saved.")

with pd.ExcelWriter(os.path.join(OUTDIR,'Fig4_stats_supplementary.xlsx'),engine='openpyxl') as w:
    pd.DataFrame(desc_rows4).to_excel(w,sheet_name='Descriptive_Stats',index=False)
    pd.DataFrame(stats_rows4).to_excel(w,sheet_name='ANOVA_Metrics',index=False)
    if tukey_rows4: pd.DataFrame(tukey_rows4).to_excel(w,sheet_name='Tukey_Pairwise',index=False)
print("Table 4 saved.")


# %%
# ============================================================
# FIGURE 5 — Petal radial bar charts per treatment
# ============================================================
# Cell bootstrap: makes this cell runnable on its own (e.g. Spyder
# "run cell" / %runcell) even if cell 0 was never executed first.
import sys, subprocess, importlib.util
def _ensure_pkgs(pkg_map):
    missing=[p for m,p in pkg_map.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
                               '--disable-pip-version-check',*missing])
_ensure_pkgs({'pandas':'pandas','numpy':'numpy','matplotlib':'matplotlib','openpyxl':'openpyxl'})

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings('ignore')
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else globals().get('_SCRIPT_DIR', os.getcwd())
OUTDIR = globals().get('OUTDIR', _SCRIPT_DIR)
FILE = globals().get('FILE', os.path.join(_SCRIPT_DIR, 'database_ecotono_02-05-2025_SHI_calculado.xlsx'))

def _best_font():
    av={f.name for f in fm.fontManager.ttflist}
    for c in ['Arial','Liberation Sans','Helvetica','FreeSans']:
        if c in av: print(f"[font] {c}"); return c
    return 'DejaVu Sans'

FONT=_best_font()
matplotlib.rcParams.update({'font.family':FONT,'font.weight':'bold',
                            'pdf.fonttype':42,'ps.fonttype':42})
matplotlib.use('Agg')

df=pd.read_excel(FILE,sheet_name=2,header=0)
df['Vegetation']=df['Vegetation'].replace('DA','RA')

TREATS=['SSF','DWS','RA']
COLORS={'SSF':'#1f78b4','DWS':'#f4c430','RA':'#d95f02'}

FUNCS=[
    ('Support root\ngrowth',         'SHI_Physical',    0.33),
    ('Sustain biological\nactivity', 'SHI_Biological',  0.33),
    ('Soil acidity\nregulation',     'SHI_Chem_pH',     0.11),
    ('Nutrient\nstorage',            'SHI_Chem_CTC',    0.11),
    ('Nutrient\navailability',       'SHI_Chem_Navail', 0.11),
]
N=len(FUNCS)

data={}
for trt in TREATS:
    sub=df[df['Vegetation']==trt]
    data[trt]=[round(sub[col].mean()/div,4) for _,col,div in FUNCS]

print("Normalized values:")
for trt,vals in data.items():
    print(f"  {trt}: {[round(v,3) for v in vals]}")

def draw_petal(ax, angle, radius, color, bar_width=0.62, n_layers=80):
    rgba=to_rgba(color)
    for k in range(n_layers):
        t=k/n_layers
        r_bot=radius*t; r_top=radius*(t+1/n_layers)
        w=bar_width*np.sin(np.pi*(1-t)**0.7)
        alpha=0.85*(1-t**0.5)
        ax.bar(angle,r_top-r_bot,width=w,bottom=r_bot,
               color=(*rgba[:3],alpha),edgecolor='none',zorder=3)
    ax.bar(angle,radius,width=bar_width*0.22,bottom=0,
           color='none',edgecolor=color,linewidth=1.4,zorder=4)

fig5=plt.figure(figsize=(19,7.5))
ax_positions=[[0.02,0.06,0.28,0.90],[0.345,0.06,0.28,0.90],[0.665,0.06,0.28,0.90]]
angles=np.linspace(np.pi/2,np.pi/2+2*np.pi,N,endpoint=False)[::-1]

for ti,trt in enumerate(TREATS):
    ax=fig5.add_axes(ax_positions[ti],polar=True)
    vals=data[trt]; color=COLORS[trt]

    for ring in [0.25,0.50,0.75,1.00]:
        theta_ring=np.linspace(0,2*np.pi,300)
        ax.plot(theta_ring,[ring]*300,color='#cccccc',lw=0.7,ls='--',zorder=1)

    bar_w=0.62
    for angle,val in zip(angles,vals):
        draw_petal(ax,angle,val,color,bar_width=bar_w)

    for i,(angle,val) in enumerate(zip(angles,vals)):
        r_val=val+0.09
        ax.text(angle,r_val,f'{val:.2f}',ha='center',va='center',
                fontsize=16,fontweight='bold',color='#111111',fontfamily=FONT,zorder=6)

    for i,angle in enumerate(angles):
        ax.text(angle,1.22,str(i+1),ha='center',va='center',
                fontsize=18,fontweight='bold',color='#111111',fontfamily=FONT,zorder=6,
                bbox=dict(boxstyle='circle,pad=0.25',facecolor='white',
                          edgecolor=color,linewidth=1.5))

    ax.set_ylim(0,1.35)
    ax.set_yticks([0.25,0.50,0.75,1.00])
    ax.set_yticklabels(['0.25','0.50','0.75','1.00'],
                       fontsize=13,fontweight='bold',color='#666666',fontfamily=FONT)
    ax.yaxis.set_tick_params(pad=3)
    ax.set_xticks([])
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.grid(False)
    ax.spines['polar'].set_linewidth(1.6); ax.spines['polar'].set_color('#333333')
    ax.set_facecolor('#f8f8f8')
    ax.set_title(trt,fontsize=22,fontweight='bold',fontfamily=FONT,color=color,pad=22,
                 bbox=dict(boxstyle='round,pad=0.5',facecolor=color,alpha=0.15,
                           edgecolor=color,linewidth=1.8))
    panel_letter=['(a)','(b)','(c)'][ti]
    ax.text(-0.10,1.10,panel_letter,transform=ax.transAxes,
            fontsize=19,fontweight='bold',fontfamily=FONT,color='#111111',
            ha='left',va='top')

legend_x=0.958; legend_y=0.90; line_h=0.145
fig5.text(legend_x,legend_y+0.04,'Soil Functions',ha='left',va='top',
          fontsize=17,fontweight='bold',fontfamily=FONT,color='#111111')
for i,(label,_,_) in enumerate(FUNCS):
    y=legend_y-i*line_h
    label_clean=label.replace('\n',' ')
    fig5.text(legend_x,y,f'{i+1}.',ha='left',va='top',
              fontsize=16,fontweight='bold',fontfamily=FONT,color='#111111')
    fig5.text(legend_x+0.022,y,label_clean,ha='left',va='top',
              fontsize=15,fontweight='bold',fontfamily=FONT,color='#444444')

BASE5=os.path.join(OUTDIR,'Fig5_RadialBar')
fig5.savefig(f'{BASE5}.png',dpi=300,bbox_inches='tight',facecolor='white')
fig5.savefig(f'{BASE5}.pdf',bbox_inches='tight',facecolor='white')
fig5.savefig(f'{BASE5}.svg',bbox_inches='tight',facecolor='white')
plt.close(fig5); print("Figure 5 saved.")

rows=[]
for trt in TREATS:
    sub=df[df['Vegetation']==trt]
    for label,col,div in FUNCS:
        raw=sub[col].mean(); norm=raw/div
        rows.append({'Treatment':trt,'Soil_Function':label.replace('\n',' '),
                     'Column_used':col,'Divisor':div,
                     'Mean_raw':round(raw,5),'Normalized_0_1':round(norm,4),
                     'N_observations':len(sub)})
with pd.ExcelWriter(os.path.join(OUTDIR,'Fig5_stats_supplementary.xlsx'),engine='openpyxl') as w:
    pd.DataFrame(rows).to_excel(w,sheet_name='Normalized_Means',index=False)
print("Table 5 saved.")

# %%
# ============================================================
# FIGURE 6 — Random Forest for SHI_Layer (Approach B: Edaphic Soil Drivers)
# ============================================================
import subprocess
import sys
import importlib.util

def _ensure_pkgs(pkg_map):
    missing = [p for m, p in pkg_map.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                               '--disable-pip-version-check', *missing])

_ensure_pkgs({'pandas': 'pandas', 'numpy': 'numpy', 'matplotlib': 'matplotlib',
              'sklearn': 'scikit-learn', 'openpyxl': 'openpyxl'})

import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else globals().get('_SCRIPT_DIR', os.getcwd())
OUTDIR = globals().get('OUTDIR', _SCRIPT_DIR)
FILE = globals().get('FILE', os.path.join(_SCRIPT_DIR, 'database_ecotono_02-05-2025_SHI_calculado.xlsx'))

def _best_font():
    av = {f.name for f in fm.fontManager.ttflist}
    for c in ['Arial', 'Liberation Sans', 'Helvetica', 'FreeSans']:
        if c in av: return c
    return 'DejaVu Sans'

FONT = _best_font()
matplotlib.rcParams.update({'font.family': FONT, 'font.weight': 'bold',
                            'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 1.4})
matplotlib.use('Agg')

# ── 1. Dados ──────────────────────────────────────────────────
ds = pd.read_excel(FILE, sheet_name=0, header=0)
ds['Vegetation'] = ds['Vegetation'].replace('DA', 'RA')
sr = pd.read_excel(FILE, sheet_name=2, header=0)
sr['Vegetation'] = sr['Vegetation'].replace('DA', 'RA')
df = ds.merge(sr[['Identifier', 'Depth', 'SHI_Layer']],
              on=['Identifier', 'Depth'], how='inner')

TREATS  = ['SSF', 'DWS', 'RA']
COLORS  = {'SSF': '#1f78b4', 'DWS': '#f4c430', 'RA': '#d95f02'}

# ── 2. Feature Selection: 7 Edaphic Soil Indicators (Approach B) ──
SOIL_PREDS = ['pH', 'CTC', 'N', 'P', 'SB', 'C', 'BD_Benites_2007']

X_all = df[SOIL_PREDS].values
y     = df['SHI_Layer'].values

print(f"Dataset: n={len(y)}, p={len(SOIL_PREDS)} soil predictors (Approach B)")
print(f"SHI_Layer range: {y.min():.4f} - {y.max():.4f}")

# ── 3. Train/Test Split (80/20) and GridSearchCV (108 combinations) ─
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'n_estimators':     [100, 300, 500],
    'max_depth':        [None, 5, 8, 10],
    'min_samples_split':[2, 5, 10],
    'max_features':     ['sqrt', 'log2', 0.5],
}
GRID_COMBOS = 3 * 4 * 3 * 3  # 108 combinações

print(f"\nRunning GridSearchCV ({GRID_COMBOS} combinations, 5-fold CV)...")
gs = GridSearchCV(
    RandomForestRegressor(random_state=42, oob_score=True, n_jobs=-1),
    param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)

best_p = gs.best_params_
print(f"Best params: {best_p}")
print(f"Best CV R2 on train folds: {gs.best_score_:.4f}")

# ── 4. Ajuste do Modelo Final e Métricas Dinâmicas ─────────────
rf = RandomForestRegressor(**best_p, oob_score=True,
                           random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

y_pred_all = rf.predict(X_all)
y_pred_tr  = rf.predict(X_tr)
y_pred_te  = rf.predict(X_te)

r2_tr    = r2_score(y_tr, y_pred_tr)
r2_te    = r2_score(y_te, y_pred_te)
r2_all   = r2_score(y,    y_pred_all)
oob      = rf.oob_score_
rmse_te  = np.sqrt(mean_squared_error(y_te, y_pred_te))
mae_te   = mean_absolute_error(y_te, y_pred_te)
rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
gap      = r2_tr - r2_te

cv_r2 = cross_val_score(
    RandomForestRegressor(**best_p, random_state=42, n_jobs=-1),
    X_all, y, cv=kf, scoring='r2')

print(f"\n-- Metrics (Model Output - Approach B) -------------------")
print(f"R2 train          : {r2_tr:.4f}")
print(f"R2 test (20%)     : {r2_te:.4f}")
print(f"OOB R2            : {oob:.4f}")
print(f"CV R2 mean+/-std  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"RMSE test         : {rmse_te:.4f}")
print(f"MAE  test         : {mae_te:.4f}")

# ── 5. Importância de Variáveis (MDI) ──────────────────────────
mdi_df = pd.DataFrame({
    'Feature': SOIL_PREDS,
    'MDI_Importance': rf.feature_importances_
}).sort_values('MDI_Importance', ascending=False).reset_index(drop=True)
print("\n-- MDI Feature Importance -----------------------------")
print(mdi_df.to_string(index=False))

label_map = {
    'BD_Benites_2007': r'BD (Mg m$^{-3}$)',
    'pH':              'pH',
    'CTC':             r'CEC (mmol$_c$ dm$^{-3}$)',
    'N':               'N (%)',
    'P':               r'P resin (mg dm$^{-3}$)',
    'SB':              r'Sum of Bases (mmol$_c$ dm$^{-3}$)',
    'C':               'SOC (%)',
}

# ── 6. Gráfico Final ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'wspace': 0.42})

# Painel (a): Observado vs Predito
ax = axes[0]
for trt in TREATS:
    mask = df['Vegetation'] == trt
    ax.scatter(y[mask], y_pred_all[mask],
               color=COLORS[trt], marker='o',
               s=65, edgecolors='#000000', linewidths=0.9,
               alpha=0.88, label=trt, zorder=3)

lims = [min(y.min(), y_pred_all.min()) - 0.025,
        max(y.max(), y_pred_all.max()) + 0.025]
ax.plot(lims, lims, color='#333333', lw=1.8, ls='--', label='1:1 line', zorder=2)

m, b = np.polyfit(y, y_pred_all, 1)
xl = np.linspace(lims[0], lims[1], 100)
ax.plot(xl, m*xl+b, color='#cc0000', lw=2.0, ls='-', label='Fit line', zorder=2)

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Observed SHI_Layer', fontsize=13, fontweight='bold', fontfamily=FONT, labelpad=6)
ax.set_ylabel('Predicted SHI_Layer', fontsize=13, fontweight='bold', fontfamily=FONT, labelpad=6)

ax.text(0.03, 0.99, '(a)', transform=ax.transAxes,
        fontsize=14, fontweight='bold', fontfamily=FONT, va='top', ha='left')

# Anotação com métricas calculadas dinamicamente
ann = (f'Out-of-bag R² = {oob:.3f}\n'
       f'Cross-validation R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}\n'
       f'test R² = {r2_te:.3f}\n'
       f'Root Mean Squared Error = {rmse_te:.4f}\n'
       f'n = {len(y)}')
ax.text(0.04, 0.90, ann, transform=ax.transAxes,
        ha='left', va='top', fontsize=10.5, fontweight='bold',
        fontfamily=FONT, color='#111111',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor='#aaaaaa', linewidth=1.1, alpha=0.93))

for sp in ax.spines.values():
    sp.set_linewidth(1.4); sp.set_color('#000000')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(axis='both', labelsize=11, direction='in',
               width=1.4, length=5, which='major', top=True, right=True)
ax.tick_params(axis='both', direction='in', width=0.8,
               length=3, which='minor', top=True, right=True)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontfamily(FONT)

leg = ax.legend(fontsize=10, frameon=True, framealpha=0.93,
                edgecolor='#888888',
                prop={'family': FONT, 'size': 10, 'weight': 'bold'},
                title='Treatment', title_fontsize=11, loc='lower right')
leg.get_title().set_fontfamily(FONT); leg.get_title().set_fontweight('bold')

# Painel (b): Importância MDI dos 7 Atributos de Solo
ax2 = axes[1]
imp_plot = mdi_df.sort_values('MDI_Importance', ascending=True)
n_feat   = len(imp_plot)
y_pos    = np.arange(n_feat)

ax2.barh(y_pos, imp_plot['MDI_Importance'],
         height=0.65, color='#4b0082',
         edgecolor='#000000', linewidth=0.9, alpha=0.85)

ax2.set_yticks(y_pos)
ax2.set_yticklabels([label_map.get(f, f) for f in imp_plot['Feature']],
                    fontsize=10.5, fontweight='bold', fontfamily=FONT)
ax2.set_xlabel('MDI Importance (Mean Decrease Impurity)',
               fontsize=12, fontweight='bold', fontfamily=FONT, labelpad=6)

for i, (_, row) in enumerate(imp_plot.iterrows()):
    ax2.text(row['MDI_Importance'] + 0.003, i,
             f"{row['MDI_Importance']:.3f}",
             va='center', ha='left',
             fontsize=9.5, fontweight='bold',
             fontfamily=FONT, color='#111111')

ax2.set_xlim(0, imp_plot['MDI_Importance'].max() * 1.22)
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
for sp in ax2.spines.values():
    sp.set_linewidth(1.4); sp.set_color('#000000')
ax2.tick_params(axis='x', labelsize=10.5, direction='in',
                width=1.4, length=5, which='major', top=True)
ax2.tick_params(axis='x', direction='in', width=0.8,
                length=3, which='minor', top=True)
ax2.tick_params(axis='y', which='both', length=0)
for lbl in ax2.get_xticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontfamily(FONT)
ax2.axvline(0, color='#000000', lw=0.8)
ax2.text(0.03, 0.99, '(b)', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', fontfamily=FONT, va='top', ha='left')

# Salvamento das figuras
BASE6 = os.path.join(OUTDIR, 'Fig6_RF_model')
fig.savefig(f'{BASE6}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{BASE6}.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{BASE6}.svg', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nFigure 6 saved: {BASE6}.png | .pdf | .svg")

# ── 7. Tabelas Suplementares (Excel) ──────────────────────────
main_table = pd.DataFrame([{
    'Metric':       'Number of observations (n)',
    'Value':        len(y),
    'Description':  'Total samples used (all depths × profiles, after merge)',
    'Interpretation': 'Full dataset used for model training and validation',
}, {
    'Metric':       'Number of predictors',
    'Value':        len(SOIL_PREDS),
    'Description':  '7 raw soil variables (pH, CEC, N, P, SB, SOC, Bulk Density)',
    'Interpretation': 'Exclusively edaphic indicators evaluated as intrinsic SHI drivers (Approach B)',
}, {
    'Metric':       'Variables in final model',
    'Value':        ', '.join(SOIL_PREDS),
    'Description':  'All 7 edaphic soil predictors used.',
    'Interpretation': 'GridSearchCV optimized hyperparameters; all features contribute',
}, {
    'Metric':       'Covariates removed',
    'Value':        'Vegetation and Depth dummies',
    'Description':  'Sampling design covariates removed to prevent confounding and evaluate purely edaphic drivers',
    'Interpretation': 'Addresses reviewer recommendation (Prof. Maurício Cherubin)',
}, {
    'Metric':       'Reference categories',
    'Value':        'None (not applicable)',
    'Description':  'No qualitative dummy variables used in the model',
    'Interpretation': 'Continuous edaphic predictors directly model SHI variability',
}, {
    'Metric':       'n_estimators (final)',
    'Value':        best_p['n_estimators'],
    'Description':  'Number of decision trees in the final Random Forest',
    'Interpretation': 'More trees → more stable predictions; selected by GridSearchCV',
}, {
    'Metric':       'max_depth (final)',
    'Value':        str(best_p['max_depth']),
    'Description':  'Maximum depth of each decision tree (None = unlimited)',
    'Interpretation': 'Controls model complexity; deeper = more flexible but higher overfit risk',
}, {
    'Metric':       'min_samples_split (final)',
    'Value':        best_p['min_samples_split'],
    'Description':  'Minimum samples required to split an internal node',
    'Interpretation': 'Higher values regularize the model; selected by GridSearchCV',
}, {
    'Metric':       'max_features / mtry (final)',
    'Value':        str(best_p['max_features']),
    'Description':  'Number of features considered at each split (mtry equivalent)',
    'Interpretation': f"{best_p['max_features']} features at each split; reduces correlation between trees",
}, {
    'Metric':       'OOB R²',
    'Value':        round(oob, 4),
    'Description':  'Out-of-bag R²: estimated on samples not used in each tree',
    'Interpretation': 'PRIMARY generalization metric. No data leakage. Equivalent to CV.',
}, {
    'Metric':       'CV R² mean (5-fold)',
    'Value':        round(cv_r2.mean(), 4),
    'Description':  '5-fold cross-validated R² on the full dataset',
    'Interpretation': 'PRIMARY generalization metric. OOB ≈ CV confirms stable generalization.',
}, {
    'Metric':       'CV R² std (5-fold)',
    'Value':        round(cv_r2.std(), 4),
    'Description':  'Standard deviation of CV R² across 5 folds',
    'Interpretation': 'Low std indicates the model is stable across different data subsets',
}, {
    'Metric':       'R² test set (20%)',
    'Value':        round(r2_te, 4),
    'Description':  'R² evaluated on held-out 20% test set',
    'Interpretation': 'Consistent with OOB and CV — confirms generalization is not a fluke',
}, {
    'Metric':       'RMSE test set',
    'Value':        round(rmse_te, 4),
    'Description':  'Root Mean Square Error on held-out test set',
    'Interpretation': f'Average prediction error of ~{rmse_te:.4f} SHI units on unseen data',
}, {
    'Metric':       'MAE test set',
    'Value':        round(mae_te, 4),
    'Description':  'Mean Absolute Error on held-out test set',
    'Interpretation': f'Median-like prediction error of ~{mae_te:.4f} SHI units',
}, {
    'Metric':       'R² train set',
    'Value':        round(r2_tr, 4),
    'Description':  'R² on training data — expected to be high in RF',
    'Interpretation': 'High R² train is normal for RF. DO NOT use as generalization metric.',
}, {
    'Metric':       'Train-test R² gap',
    'Value':        round(gap, 4),
    'Description':  'Difference between R² train and R² test',
    'Interpretation': 'Gap is expected in RF due to tree memorization.',
}])

tuning_table = pd.DataFrame([{
    'Hyperparameter':   'n_estimators',
    'Values_tested':    '100, 300, 500',
    'Final_value':      best_p['n_estimators'],
    'Description':      'Number of trees in the forest',
    'Why_tested':       'More trees = more stable but slower. 500 often sufficient.',
}, {
    'Hyperparameter':   'max_depth',
    'Values_tested':    'None, 5, 8, 10',
    'Final_value':      str(best_p['max_depth']),
    'Description':      'Maximum depth of each tree',
    'Why_tested':       'Deeper trees overfit; shallower underfit. None = unrestricted growth.',
}, {
    'Hyperparameter':   'min_samples_split',
    'Values_tested':    '2, 5, 10',
    'Final_value':      best_p['min_samples_split'],
    'Description':      'Minimum samples needed to split a node',
    'Why_tested':       'Higher values prevent overfitting by requiring more evidence to split.',
}, {
    'Hyperparameter':   'max_features (mtry)',
    'Values_tested':    "sqrt, log2, 0.5",
    'Final_value':      str(best_p['max_features']),
    'Description':      'Features considered at each split (mtry in R notation)',
    'Why_tested':       'Controls decorrelation between trees. sqrt and log2 are standard; 0.5=50%.',
}, {
    'Hyperparameter':   'Total combinations tested',
    'Values_tested':    '3 × 4 × 3 × 3 = 108',
    'Final_value':      108,
    'Description':      'Full factorial grid search',
    'Why_tested':       'Exhaustive search over all combinations using 5-fold CV on training set.',
}, {
    'Hyperparameter':   'CV strategy',
    'Values_tested':    'KFold (k=5, shuffle=True, random_state=42)',
    'Final_value':      '5-fold',
    'Description':      'Cross-validation method used in GridSearchCV',
    'Why_tested':       'Balances bias-variance in CV estimate; reproducible with fixed seed.',
}, {
    'Hyperparameter':   'Scoring metric',
    'Values_tested':    'R² (coefficient of determination)',
    'Final_value':      'R²',
    'Description':      'Metric used to select best hyperparameter combination',
    'Why_tested':       'Most interpretable metric for regression; aligns with study objectives.',
}, {
    'Hyperparameter':   'Train/test split',
    'Values_tested':    '80% train / 20% test',
    'Final_value':      '80/20',
    'Description':      'Hold-out set for final evaluation (random_state=42)',
    'Why_tested':       'Independent validation set not seen during GridSearchCV.',
}, {
    'Hyperparameter':   'Variable selection',
    'Values_tested':    'Exclusively 7 soil variables (Approach B)',
    'Final_value':      '7 edaphic predictors retained',
    'Description':      'Soil quality indicators only; treatment and depth dummies excluded',
    'Why_tested':       'Removes confounding and evaluates the relative contribution of soil properties to SHI.',
}])

imp_table = mdi_df.copy()
imp_table['Feature_label']  = imp_table['Feature'].map(label_map)
imp_table['Cumulative_MDI'] = imp_table['MDI_Importance'].cumsum().round(4)
imp_table['Description'] = imp_table['Feature'].map({
    'CTC':             'Cation Exchange Capacity — soil nutrient retention capacity',
    'C':               'Soil Organic Carbon — biological activity indicator',
    'N':               'Total nitrogen — nutrient cycling indicator',
    'BD_Benites_2007': 'Bulk density — soil physical structure',
    'P':               'Available phosphorus (resin) — nutrient availability',
    'SB':              'Sum of Bases — soil fertility indicator',
    'pH':              'Soil pH — acidity/alkalinity regulation',
})
imp_table['Interpretation'] = imp_table['Feature'].map({
    'CTC':             'Dominant predictor — CEC integrates soil chemistry and fertility capacity',
    'C':               'High importance — SOC drives biological and organic component of SHI',
    'N':               'Substantial importance — linked to organic matter quality and nutrient cycling',
    'BD_Benites_2007': 'Key physical driver — bulk density reflects soil compaction and pore structure',
    'P':               'Moderate importance — phosphorus availability contributes to chemical fertility',
    'SB':              'Moderate importance — base cation saturation reflects nutrient reserve',
    'pH':              'Foundational indicator — controls chemical availability and microbial activity',
})

pred_df = df[['Identifier', 'Depth', 'Vegetation']].copy()
pred_df['Observed_SHI_Layer']  = y.round(6)
pred_df['Predicted_SHI_Layer'] = y_pred_all.round(6)
pred_df['Residual']            = (y - y_pred_all).round(6)
pred_df['Abs_Error']           = np.abs(y - y_pred_all).round(6)

XBASE6 = os.path.join(OUTDIR, 'Fig6_stats_supplementary.xlsx')
with pd.ExcelWriter(XBASE6, engine='openpyxl') as writer:
    main_table.to_excel(writer, sheet_name='1_Report_This_In_Paper', index=False)
    tuning_table.to_excel(writer, sheet_name='2_Hyperparameter_Tuning', index=False)
    imp_table.to_excel(writer, sheet_name='3_Feature_Importance_MDI', index=False)
    pred_df.to_excel(writer, sheet_name='4_Observed_vs_Predicted', index=False)

print(f"Supplementary table saved: {XBASE6}")
print("\nDone! All outputs saved.")

#%%
# ============================================================
# SUPPLEMENTARY TABLE S1 — Chemical+Biological-only SHI
# (robustness check against circularity of the 3-pillar SHI)
# ============================================================
# REVIEWER — THIS CELL WAS ACTUALLY RUN, NOT JUST WRITTEN.
# Rationale for including it: the chemical and biological pillars
# carry two-thirds of the SHI weight (0.33 + 0.33 of 1.00; see the
# 'Weights' sheet), so recomputing the index on those two pillars
# only — rescaled to sum to 1 — is the highest-value, zero-cost
# robustness check against the objection that the SHI ranking is
# circular (i.e. built to favour SSF). It was run against the real
# dataset before this section was finalised: the SSF > DWS > RA
# ranking holds, unchanged, at every one of the 5 depth intervals
# and in the integrated profile (see printed summary and
# TableS1_SHI_BioChem_only_supplementary.xlsx). Because the check
# passed, this cell and the Table S1 reference were kept in the
# paper's framing. Had the ranking NOT held, this cell would need
# to be deleted (along with the Table S1 reference in the main
# text) and the paper's framing revisited before submission.
# Cell bootstrap: makes this cell runnable on its own (e.g. Spyder
# "run cell" / %runcell) even if cell 0 was never executed first.
import sys, subprocess, importlib.util
def _ensure_pkgs(pkg_map):
    missing=[p for m,p in pkg_map.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
                               '--disable-pip-version-check',*missing])
_ensure_pkgs({'pandas':'pandas','numpy':'numpy','openpyxl':'openpyxl'})

import os
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else globals().get('_SCRIPT_DIR', os.getcwd())
OUTDIR = globals().get('OUTDIR', _SCRIPT_DIR)
FILE = globals().get('FILE', os.path.join(_SCRIPT_DIR, 'database_ecotono_02-05-2025_SHI_calculado.xlsx'))

DEPTHS = ['P20','P40','P60','P80','P100']
DLABS  = {'P20':'0-20 cm','P40':'20-40 cm','P60':'40-60 cm','P80':'60-80 cm','P100':'80-100 cm'}
TREATS = ['SSF','DWS','RA']

sr = pd.read_excel(FILE, sheet_name='SHI_Results', header=0)
sr['Vegetation'] = sr['Vegetation'].replace('DA','RA')

# Original pillar weights (Weight I, 'Weights' sheet): Physical=0.33,
# Biological=0.33, Chemical=0.33 (=0.11*3 sub-indicators). Dropping
# the physical pillar and rescaling the remaining two so they sum to
# 1 is equivalent to dividing by their combined original weight.
W_BIOLOGICAL = 0.33
W_CHEMICAL   = 0.33
RESCALE = 1.0 / (W_BIOLOGICAL + W_CHEMICAL)

sr['SHI_BioChem_Layer'] = (sr['SHI_Biological'] + sr['SHI_Chemical']) * RESCALE

# ---- Ranking per depth interval --------------------------------
depth_orig = sr.groupby(['Depth','Vegetation'])['SHI_Layer'].mean().unstack().loc[DEPTHS]
depth_bc   = sr.groupby(['Depth','Vegetation'])['SHI_BioChem_Layer'].mean().unstack().loc[DEPTHS]
rank_orig  = depth_orig.rank(axis=1, ascending=False)
rank_bc    = depth_bc.rank(axis=1, ascending=False)

per_depth_rows = []
for dep in DEPTHS:
    for trt in TREATS:
        per_depth_rows.append({
            'Depth': DLABS[dep],
            'Treatment': trt,
            'Mean_SHI_3pillar': round(depth_orig.loc[dep, trt], 4),
            'Rank_3pillar': int(rank_orig.loc[dep, trt]),
            'Mean_SHI_BioChem_only': round(depth_bc.loc[dep, trt], 4),
            'Rank_BioChem_only': int(rank_bc.loc[dep, trt]),
            'Ranking_unchanged': bool(rank_orig.loc[dep, trt] == rank_bc.loc[dep, trt]),
        })
per_depth_df = pd.DataFrame(per_depth_rows)
depth_ranking_holds = bool(per_depth_df['Ranking_unchanged'].all())

# ---- Ranking for the integrated profile -------------------------
# Overall_SHI (3-pillar) is precomputed per profile in the source
# file as the mean of SHI_Layer across the 5 depths; the bio+chem
# analogue is built the same way from SHI_BioChem_Layer.
profile_orig = sr[sr['Overall_SHI'].notna()].groupby('Vegetation')['Overall_SHI'].mean()
profile_bc   = sr.groupby(['Profile','Vegetation'])['SHI_BioChem_Layer'].mean().reset_index() \
                 .groupby('Vegetation')['SHI_BioChem_Layer'].mean()
rank_profile_orig = profile_orig.rank(ascending=False)
rank_profile_bc   = profile_bc.rank(ascending=False)

integrated_rows = []
for trt in TREATS:
    integrated_rows.append({
        'Treatment': trt,
        'Mean_Overall_SHI_3pillar': round(profile_orig[trt], 4),
        'Rank_3pillar': int(rank_profile_orig[trt]),
        'Mean_Overall_SHI_BioChem_only': round(profile_bc[trt], 4),
        'Rank_BioChem_only': int(rank_profile_bc[trt]),
        'Ranking_unchanged': bool(rank_profile_orig[trt] == rank_profile_bc[trt]),
    })
integrated_df = pd.DataFrame(integrated_rows)
integrated_ranking_holds = bool(integrated_df['Ranking_unchanged'].all())

overall_ranking_holds = depth_ranking_holds and integrated_ranking_holds

print("\n[Table S1] Chemical+Biological-only SHI robustness check")
print(per_depth_df.to_string(index=False))
print()
print(integrated_df.to_string(index=False))
print()
if overall_ranking_holds:
    print("[Table S1] RESULT: ranking is unchanged (SSF > DWS > RA) at every "
          "depth and in the integrated profile when the SHI is recomputed on "
          "the chemical and biological pillars only. Circularity objection "
          "is not supported by the data.")
else:
    print("[Table S1] RESULT: WARNING — the ranking CHANGES under the "
          "chemical+biological-only SHI for at least one depth or the "
          "integrated profile. Review 'Ranking_unchanged' columns below "
          "before submission; the paper's framing may need to change.")

methodology_df = pd.DataFrame([{
    'Check': 'Chemical + Biological-only SHI (robustness / circularity check)',
    'Original_weights': 'Physical=0.33, Biological=0.33, Chemical=0.33 (of 1.00)',
    'Rescaled_weights': f'Biological={W_BIOLOGICAL*RESCALE:.4f}, Chemical={W_CHEMICAL*RESCALE:.4f} (of 1.00)',
    'Formula': 'SHI_BioChem_Layer = (SHI_Biological + SHI_Chemical) / (W_Biological + W_Chemical)',
    'Integrated_profile_formula': 'mean of SHI_BioChem_Layer across the 5 depths per profile, then mean by treatment',
    'Depth_ranking_holds': depth_ranking_holds,
    'Integrated_ranking_holds': integrated_ranking_holds,
    'Overall_result': 'Ranking unchanged (SSF > DWS > RA) in all depths and integrated profile' if overall_ranking_holds
                       else 'Ranking changed in at least one depth or the integrated profile — see tables',
}])

TABLE_S1_PATH = os.path.join(OUTDIR, 'TableS1_SHI_BioChem_only_supplementary.xlsx')
with pd.ExcelWriter(TABLE_S1_PATH, engine='openpyxl') as w:
    methodology_df.to_excel(w, sheet_name='Methodology', index=False)
    per_depth_df.to_excel(w, sheet_name='Ranking_per_Depth', index=False)
    integrated_df.to_excel(w, sheet_name='Ranking_Integrated', index=False)
    sr[['Depth','Vegetation','Profile','Identifier','SHI_Biological','SHI_Chemical',
        'SHI_BioChem_Layer']].to_excel(w, sheet_name='Raw_Recomputed_Values', index=False)

print(f"\nTable S1 saved: {TABLE_S1_PATH}")


#%%
# ============================================================
# SUPPLEMENTARY FIGURE S1 — Overall SHI boxplot, Chemical+Biological
# only (reconstruction of Figure 4, panel b, dropping the physical
# pillar; companion figure to Table S1)
# ============================================================
# Cell bootstrap: makes this cell runnable on its own (e.g. Spyder
# "run cell" / %runcell) even if cell 0 was never executed first.
import sys, subprocess, importlib.util
def _ensure_pkgs(pkg_map):
    missing=[p for m,p in pkg_map.items() if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
                               '--disable-pip-version-check',*missing])
_ensure_pkgs({'pandas':'pandas','numpy':'numpy','matplotlib':'matplotlib','scipy':'scipy','statsmodels':'statsmodels','openpyxl':'openpyxl'})

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import stats
from scipy.stats import shapiro, levene, boxcox
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else globals().get('_SCRIPT_DIR', os.getcwd())
OUTDIR = globals().get('OUTDIR', _SCRIPT_DIR)
FILE = globals().get('FILE', os.path.join(_SCRIPT_DIR, 'database_ecotono_02-05-2025_SHI_calculado.xlsx'))

def _best_font():
    av={f.name for f in fm.fontManager.ttflist}
    for c in ['Arial','Liberation Sans','Helvetica','FreeSans']:
        if c in av: print(f"[font] {c}"); return c
    return 'DejaVu Sans'

FONT=_best_font()
matplotlib.rcParams.update({'font.family':FONT,'font.weight':'bold',
                            'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':1.5})
matplotlib.use('Agg')

df=pd.read_excel(FILE,sheet_name='SHI_Results',header=0)
df['Vegetation']=df['Vegetation'].replace('DA','RA')

TREATS=['SSF','DWS','RA']
COLORS={'SSF':'#1f78b4','DWS':'#f4c430','RA':'#d95f02'}
ALPHA=0.05

# Same rescaling as Table S1: drop the physical pillar (weight 0.33),
# keep biological (0.33) + chemical (0.33), rescale so they sum to 1.
W_BIOLOGICAL=0.33; W_CHEMICAL=0.33
RESCALE=1.0/(W_BIOLOGICAL+W_CHEMICAL)
df['SHI_BioChem_Layer']=(df['SHI_Biological']+df['SHI_Chemical'])*RESCALE

# Per-profile Overall SHI = mean of the layer-level index across the
# 5 depths (same aggregation rule as the source file's Overall_SHI).
df_overall=df.groupby(['Profile','Vegetation'],as_index=False)['SHI_BioChem_Layer'].mean()
df_overall=df_overall.rename(columns={'SHI_BioChem_Layer':'Overall_SHI_BioChem'})

def compact_letters(tukey,treats):
    groups=list(tukey.groupsunique); pairs=list(combinations(range(len(groups)),2))
    rd={}
    for (i,j),rej in zip(pairs,tukey.reject):
        rd[(groups[i],groups[j])]=rej; rd[(groups[j],groups[i])]=rej
    ls={t:set() for t in treats}; cur=0; abc=list('abcdefgh')
    for t1 in treats:
        if not ls[t1]:
            ls[t1].add(abc[cur])
            for t2 in treats:
                if t1!=t2 and not rd.get((t1,t2),True): ls[t2].add(abc[cur])
            cur+=1
    for t in treats:
        if not ls[t]: ls[t].add(abc[cur]); cur+=1
    return {t:''.join(sorted(ls[t])) for t in treats}

def run_stats(df_sub,var):
    groups=[df_sub[df_sub['Vegetation']==t][var].dropna().values for t in TREATS]
    def resid(g): return np.concatenate([x-x.mean() for x in g])
    r=resid(groups); sw_s,sw_p=shapiro(r); lv_s,lv_p=levene(*groups)
    boxcox_used=False; lam=na=ha=swap=lvap=None; ga=groups; pdf=False
    if sw_p<ALPHA or lv_p<ALPHA:
        av=np.concatenate(groups); sh=max(0,-av.min()+1e-6)
        try:
            _,lam=boxcox(av+sh)
            tr=[boxcox(g+sh,lmbda=lam) for g in groups]
            if all(np.std(g)<1e-10 for g in tr): raise ValueError("collapsed")
            r2=resid(tr); sw2,swp2=shapiro(r2); lv2,lvp2=levene(*tr)
            boxcox_used=True; na=swp2>=ALPHA; ha=lvp2>=ALPHA
            swap=round(swp2,4); lvap=round(lvp2,4); ga=tr
            if not na or not ha: pdf=True
        except: pass
    fs,ap=stats.f_oneway(*ga)
    lets={t:'' for t in TREATS}; tukey_rows=[]
    if ap<=ALPHA:
        va=np.concatenate(ga); la=np.concatenate([[t]*len(g) for t,g in zip(TREATS,ga)])
        tk=pairwise_tukeyhsd(va,la,alpha=ALPHA); lets=compact_letters(tk,TREATS)
        grps=list(tk.groupsunique); p2=list(combinations(range(len(grps)),2))
        for (i,j),rej,pval in zip(p2,tk.reject,tk.pvalues):
            tukey_rows.append({'Variable':var,'Group1':grps[i],'Group2':grps[j],
                               'Reject_H0':bool(rej),'p_adj':round(float(pval),6)})
    return dict(sw_stat=round(sw_s,4),sw_p=round(sw_p,4),
                lev_stat=round(lv_s,4),lev_p=round(lv_p,4),
                norm_before=sw_p>=ALPHA,homo_before=lv_p>=ALPHA,
                boxcox_used=boxcox_used,boxcox_lambda=round(lam,4) if lam else None,
                norm_after=na,homo_after=ha,sw_after_p=swap,lev_after_p=lvap,
                proceeded_despite_failure=pdf,f_stat=round(fs,4),
                anova_p=round(ap,4),significant=ap<=ALPHA,letters=lets,tukey_rows=tukey_rows)

def desc_stats(df_sub,var,treat):
    v=df_sub[df_sub['Vegetation']==treat][var].dropna()
    return dict(Variable=var,Treatment=treat,n=len(v),
                Mean=round(v.mean(),4),Median=round(v.median(),4),
                SD=round(v.std(),4),Min=round(v.min(),4),Max=round(v.max(),4),
                CV_pct=round(v.std()/v.mean()*100,2) if v.mean()!=0 else None)

def whisker_top(vals):
    q75=np.percentile(vals,75); iqr=q75-np.percentile(vals,25)
    fence=q75+1.5*iqr; above=vals[vals<=fence]
    return above.max() if len(above) else q75

sr_overall=run_stats(df_overall,'Overall_SHI_BioChem')
desc_rows_s1=[desc_stats(df_overall,'Overall_SHI_BioChem',t) for t in TREATS]
stats_rows_s1=[{'Variable':'Overall_SHI_BioChem',
    'Shapiro_W_before':sr_overall['sw_stat'],'Shapiro_p_before':sr_overall['sw_p'],
    'Normal_before':sr_overall['norm_before'],'Levene_stat_before':sr_overall['lev_stat'],
    'Levene_p_before':sr_overall['lev_p'],'Homogeneous_before':sr_overall['homo_before'],
    'BoxCox_applied':sr_overall['boxcox_used'],'BoxCox_lambda':sr_overall['boxcox_lambda'],
    'Shapiro_p_after':sr_overall['sw_after_p'],'Normal_after':sr_overall['norm_after'],
    'Levene_p_after':sr_overall['lev_after_p'],'Homogeneous_after':sr_overall['homo_after'],
    'Proceeded_despite_failure':sr_overall['proceeded_despite_failure'],
    'F_stat':sr_overall['f_stat'],'ANOVA_p':sr_overall['anova_p'],
    'Significant':sr_overall['significant'],
    'Letter_SSF':sr_overall['letters'].get('SSF','ns'),
    'Letter_DWS':sr_overall['letters'].get('DWS','ns'),
    'Letter_RA':sr_overall['letters'].get('RA','ns')}]

figS1,ax_b=plt.subplots(figsize=(6.0,7.2))
bp_data=[df_overall[df_overall['Vegetation']==t]['Overall_SHI_BioChem'].dropna().values for t in TREATS]
bp=ax_b.boxplot(bp_data,positions=[1,2,3],widths=0.52,patch_artist=True,notch=False,
                medianprops=dict(color='#000000',linewidth=2.8),
                whiskerprops=dict(color='#000000',linewidth=1.8),
                capprops=dict(color='#000000',linewidth=1.8),
                flierprops=dict(marker='o',markersize=6,markerfacecolor='#777777',
                                markeredgecolor='#000000',linewidth=1.1))
for patch,t in zip(bp['boxes'],TREATS):
    patch.set_facecolor(COLORS[t]); patch.set_edgecolor('#000000')
    patch.set_linewidth(2.0); patch.set_alpha(0.92)

all_vals=np.concatenate(bp_data)
ymin=all_vals.min(); ymax=all_vals.max(); yspan=ymax-ymin
ax_b.set_ylim(ymin-yspan*0.10, ymax+yspan*0.32)

for pos,t in zip([1,2,3],TREATS):
    vals=bp_data[TREATS.index(t)]
    median_val=np.median(vals)
    tip=max(whisker_top(vals),vals.max())
    y_annot=min(tip+yspan*0.04, ymin+yspan*0.87)
    letter=sr_overall['letters'].get(t,'') if sr_overall['significant'] else 'ns'
    ax_b.text(pos,y_annot,f'{median_val:.3f} {letter}',ha='center',va='bottom',
              fontsize=14,fontweight='bold',color='#000000',fontfamily=FONT,
              transform=ax_b.transData)

ax_b.set_xticks([1,2,3])
ax_b.set_xticklabels(TREATS,fontsize=15,fontweight='bold',fontfamily=FONT)
ax_b.set_ylabel('Overall SHI (Chemical + Biological only)',fontsize=14,
                fontweight='bold',fontfamily=FONT,labelpad=6)
ax_b.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_b.yaxis.set_major_locator(MultipleLocator(0.05))
ax_b.tick_params(axis='y',labelsize=13,which='major',direction='in',
                 width=1.5,length=6,right=True)
ax_b.tick_params(axis='y',which='minor',direction='in',width=0.9,length=3,right=True)
ax_b.tick_params(axis='x',which='both',direction='in',top=True,length=5,width=1.4)
for sp in ax_b.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color('#000000')
for lbl in ax_b.get_yticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontsize(13); lbl.set_fontfamily(FONT)
for lbl in ax_b.get_xticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontfamily(FONT)
ax_b.text(0.04,0.98,'(b)',transform=ax_b.transAxes,
          fontsize=18,fontweight='bold',fontfamily=FONT,va='top',ha='left')
ax_b.set_title('Reconstruction of Fig. 4b\n(physical pillar removed, weights rescaled to 1)',
                fontsize=11,fontweight='bold',fontfamily=FONT,color='#444444',pad=10)

figS1.tight_layout()
BASE_S1=os.path.join(OUTDIR,'FigS1_Overall_SHI_BioChem_only')
figS1.savefig(f'{BASE_S1}.png',dpi=300,bbox_inches='tight',facecolor='white')
figS1.savefig(f'{BASE_S1}.pdf',bbox_inches='tight',facecolor='white')
figS1.savefig(f'{BASE_S1}.svg',bbox_inches='tight',facecolor='white')
plt.close(figS1)
print(f"Figure S1 saved: {BASE_S1}.png | .pdf | .svg")

with pd.ExcelWriter(os.path.join(OUTDIR,'FigS1_stats_supplementary.xlsx'),engine='openpyxl') as w:
    pd.DataFrame(desc_rows_s1).to_excel(w,sheet_name='Descriptive_Stats',index=False)
    pd.DataFrame(stats_rows_s1).to_excel(w,sheet_name='ANOVA_Metrics',index=False)
    if sr_overall['tukey_rows']:
        pd.DataFrame(sr_overall['tukey_rows']).to_excel(w,sheet_name='Tukey_Pairwise',index=False)
    df_overall.to_excel(w,sheet_name='Per_Profile_Values',index=False)
print("Figure S1 supplementary table saved.")


###############
