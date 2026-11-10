"""Deterministic software point renderer with explicit camera metadata."""
from __future__ import annotations
import uuid
import numpy as np
from PIL import Image, ImageDraw
from .geometry import unit, vector

def render_view(scene, position=None, target=None, up=None, projection="orthographic",
                span=None, region=None, section=None, width=960, height=640, label="Agentenansicht"):
    if projection not in ("orthographic","perspective"): raise ValueError("Ungültige Projektion.")
    width,height=int(width),int(height)
    if not 128<=width<=1600 or not 128<=height<=1200: raise ValueError("Ungültige Bildgröße.")
    p,c,source_ids,total=scene.sample(region,180000,section=section)
    if not len(p): raise ValueError("Die gewählte Ansicht enthält keine Punkte.")
    if len(p)>180000:
        idx=np.linspace(0,len(p)-1,180000,dtype=int);p,c=p[idx],c[idx]
    bounds=np.array([p.min(axis=0),p.max(axis=0)]);center=bounds.mean(axis=0)
    extent=max(float(np.max(bounds[1]-bounds[0])),.1)
    target=vector(target if target is not None else center)
    position=vector(position if position is not None else target+extent*np.array([1.1,-1.4,1.]))
    forward=unit(target-position);up=unit(up if up is not None else [0,0,1])
    if abs(up@forward)>.999: up=np.array([0.,1.,0.])
    right=unit(np.cross(forward,up));up=unit(np.cross(right,forward))
    rotation=np.array([right,up,forward]);q=(p-position)@rotation.T
    span=float(span if span is not None else extent*1.45)
    if not .001<span<1e9: raise ValueError("Ungültiger Bildausschnitt.")
    fx=width/span
    if projection=="perspective":
        fx=width/(2*np.tan(np.radians(50)/2));scale=fx/np.maximum(q[:,2],1e-9)
    else: scale=np.full(len(p),fx)
    px=np.rint(width/2+q[:,0]*scale).astype(int)
    py=np.rint(height/2-q[:,1]*scale).astype(int)
    visible=(q[:,2]>0)&(px>=0)&(px<width)&(py>=0)&(py<height)
    idx=np.where(visible)[0];idx=idx[np.argsort(q[idx,2])[::-1]]
    pixels=np.empty((height,width,3),dtype=np.uint8);pixels[:]=[15,25,36]
    pixels[py[idx],px[idx]]=c[idx]
    img=Image.fromarray(pixels);draw=ImageDraw.Draw(img)
    draw.text((18,18),str(label)[:100],fill=(190,216,222))
    draw.text((18,height-28),f"{projection} | Z up | metres | {len(p):,} points",fill=(133,161,172))
    vid=uuid.uuid4().hex[:10];filename=f"view-{vid}.png";img.save(scene.folder/filename)
    metadata={"id":vid,"label":label,"file":filename,"position":position.tolist(),"target":target.tolist(),
              "up":up.tolist(),"rotation_world_to_camera":rotation.tolist(),"projection":projection,
              "span_m":span,"fov_degrees":50 if projection=="perspective" else None,
              "image_size":[width,height],"pixel_focal_scale":fx,"region":region,"section":section,
              "coordinate_system":"right-handed Z-up, metres, source coordinates","source_hash":scene.state["source_hash"],
              "point_count":len(p),"visible_points":int(visible.sum()),"region_point_count":total,"sampled":len(p)<total,
              "url":f"/api/scenes/{scene.state['id']}/artifacts/{filename}"}
    scene.state["views"].append(metadata)
    return metadata
