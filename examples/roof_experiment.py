"""Example experiment, not a built-in reconstruction tool or forced agent recipe.

Assumes a mostly gabled roof whose ridge approximately follows its long XY axis.
Fits independent cross sections, so a curved ridge or changing pitch is possible.
Reports assumptions and residuals instead of claiming general roof recognition.
Paste into the experiment editor with a roof-only point selection.
"""
import numpy as np
from scipy.optimize import least_squares

xy=points[:,:2];center=xy.mean(axis=0)
values,axes=np.linalg.eigh(np.cov((xy-center).T));along=axes[:,-1];across=axes[:,0]
q=np.column_stack([(xy-center)@along,(xy-center)@across,points[:,2]])
edges=np.linspace(q[:,0].min(),q[:,0].max(),13)
sections=[];vertices=[];ridges=[];eaves=[];residuals=[]
for section_index,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
    p=q[(q[:,0]>=lo)&(q[:,0]<=hi)]
    if len(p)<60: continue
    t,z=p[:,1],p[:,2];a,b=np.quantile(t,[.01,.99])
    # Continuous asymmetric broken-line cross section with a free ridge location.
    def height(params,x):
        ridge,peak,left,right=params
        return peak-np.where(x<ridge,left*(ridge-x),right*(x-ridge))
    fit=least_squares(lambda x:height(x,t)-z,[np.median(t),np.quantile(z,.95),.5,.5],
        bounds=([a,float(z.min()),0,0],[b,float(z.max())+2,5,5]),loss="soft_l1",f_scale=.015)
    ridge,peak,left,right=fit.x
    # Cover the actually observed end boundaries. The first/last strip fit is
    # an estimate at that boundary, not a claim of an exactly sampled ridge.
    station=lo if section_index==0 else hi if section_index==len(edges)-2 else (lo+hi)/2
    def world(cross,elevation): return [*(center+along*station+across*cross),float(elevation)]
    section=[world(a,height(fit.x,a)),world(ridge,peak),world(b,height(fit.x,b))]
    vertices.extend(section);ridges.append(section[1]);eaves.extend([section[0],section[2]])
    residuals.extend((height(fit.x,t)-z).tolist())
    sections.append({"station_m":station,"ridge_cross_m":ridge,"ridge_z_m":peak,
        "left_pitch_degrees":float(np.degrees(np.arctan(left))),"right_pitch_degrees":float(np.degrees(np.arctan(right))),
        "eave_left_z_m":section[0][2],"eave_right_z_m":section[2][2],"support":len(p)})
if len(sections)<2: raise ValueError("Zu wenig Stützung für zwei Dachquerschnitte.")
faces=[]
for i in range(len(sections)-1):
    a=i*3;b=(i+1)*3
    faces.extend([[a,b,b+1],[a,b+1,a+1],[a+1,b+1,b+2],[a+1,b+2,a+2]])
ridge_lines=[p for pair in zip(ridges[:-1],ridges[1:]) for p in pair]
result={"method":"robust asymmetric cross-section regression / variable ridge and pitch", "sections":sections,
    "rms_m":float(np.sqrt(np.mean(np.square(residuals)))),
    "assumptions":["Dachregion ohne Innenpunkte wählen", "First etwa parallel zur langen XY-PCA-Achse", "Jeder Schnitt hat einen First"],
    "uncertainty":"Walmdächer, Mehrfachfirste und starke Verdeckung benötigen ein anderes Verfahren. Zwischen Schnitten linear interpoliert; Randprofile aus den angrenzenden Punktstreifen geschätzt.",
    "vertices":vertices,"faces":faces,
    "overlays":[{"kind":"mesh","vertices":vertices,"faces":faces,"label":"Experimentelle Dachfläche"},
                {"kind":"lines","vertices":ridge_lines,"label":"Gemessener Firstverlauf"}]}
