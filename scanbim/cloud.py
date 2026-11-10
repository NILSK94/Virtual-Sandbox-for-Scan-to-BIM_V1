"""Bounded PLY (ASCII/binary), PCD (ASCII/binary), XYZ readers."""
from __future__ import annotations
import hashlib
import io
import struct
from pathlib import Path
import numpy as np

MAX_POINTS = 1_000_000
MAX_BYTES = 64 * 1024 * 1024
PLY_TYPES = {"char":"i1", "uchar":"u1", "int8":"i1", "uint8":"u1",
             "short":"i2", "ushort":"u2", "int16":"i2", "uint16":"u2",
             "int":"i4", "uint":"u4", "int32":"i4", "uint32":"u4",
             "float":"f4", "double":"f8", "float32":"f4", "float64":"f8"}

def _columns(a, names):
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError("Die Punktdatei benötigt x, y und z.")
    get = (lambda k: a[k]) if a.dtype.names else (lambda k: a[:, names.index(k)])
    p = np.column_stack([get(k) for k in ("x","y","z")]).astype(float)
    c = None
    for keys in [("red","green","blue"),("r","g","b")]:
        if all(k in names for k in keys):
            c = np.column_stack([get(k) for k in keys]).astype(float)
            break
    if c is None and "rgb" in names:
        packed = np.asarray(get("rgb"), dtype=np.float32).view(np.uint32)
        c = np.column_stack([(packed >> 16)&255, (packed >> 8)&255, packed&255])
    return p, c

def _ply(raw):
    stream = io.BytesIO(raw)
    if stream.readline().strip() != b"ply":
        raise ValueError("Ungültiger PLY-Kopf.")
    fmt, count, props, vertex = None, None, [], False
    for _ in range(4096):
        line = stream.readline().decode("ascii").strip().split()
        if not line:
            continue
        if line[0] == "format": fmt = line[1]
        if line[0] == "element":
            if count is None and line[1] != "vertex":
                raise ValueError("PLY: vertex muss das erste Element sein.")
            vertex = line[1] == "vertex"
            if vertex: count = int(line[2])
        if line[0] == "property" and vertex:
            if line[1] == "list": raise ValueError("PLY-Vertex-Listen werden nicht unterstützt.")
            props.append((line[2], PLY_TYPES[line[1]]))
        if line[0] == "end_header": break
    else: raise ValueError("PLY-Kopf zu lang.")
    if count is None or not 3 <= count <= MAX_POINTS: raise ValueError("Ungültige Punktanzahl.")
    if fmt == "ascii":
        lines = [stream.readline() for _ in range(count)]
        a = np.loadtxt(io.BytesIO(b"".join(lines)), ndmin=2)
    elif fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        a = np.frombuffer(stream.read(), dtype=np.dtype([(n,endian+t) for n,t in props]), count=count)
    else: raise ValueError("Nicht unterstütztes PLY-Format.")
    if len(a) != count: raise ValueError("PLY-Punktanzahl stimmt nicht mit dem Kopf überein.")
    return _columns(a, [n for n,_ in props])

def _pcd(raw):
    stream, header = io.BytesIO(raw), {}
    for _ in range(128):
        line = stream.readline().decode("ascii").strip()
        if not line or line.startswith("#"): continue
        parts = line.split(); header[parts[0].upper()] = parts[1:]
        if parts[0].upper() == "DATA": break
    fields = header.get("FIELDS", header.get("FIELD", []))
    count = int(header.get("POINTS", [str(int(header.get("WIDTH", [0])[0])*int(header.get("HEIGHT", [1])[0]))])[0])
    if not 3 <= count <= MAX_POINTS: raise ValueError("Ungültige PCD-Punktanzahl.")
    counts = [int(x) for x in header.get("COUNT", ["1"]*len(fields))]
    if header["DATA"] == ["ascii"]:
        a = np.loadtxt(stream, ndmin=2)
        expanded = [k if j == 0 else f"{k}_{j}" for k,n in zip(fields,counts) for j in range(n)]
        p,c = _columns(a, expanded)
    elif header["DATA"] == ["binary"]:
        types = {"F":"f", "I":"i", "U":"u"}
        dtype = [(k, "<"+types[t]+s, (n,)) if n>1 else (k,"<"+types[t]+s)
                 for k,t,s,n in zip(fields,header["TYPE"],header["SIZE"],counts)]
        a = np.frombuffer(stream.read(), dtype=np.dtype(dtype), count=count)
        p,c = _columns(a,fields)
    else: raise ValueError("PCD binary_compressed: bitte als ASCII oder binary speichern.")
    if len(p) != count: raise ValueError("PCD-Punktanzahl stimmt nicht.")
    return p,c

def read_cloud(raw: bytes, filename: str, unit="m"):
    if len(raw) > MAX_BYTES: raise ValueError("Datei größer als 64 MiB.")
    if unit not in {"m","cm","mm"}: raise ValueError("Einheit muss m, cm oder mm sein.")
    ext = Path(filename).suffix.lower()
    if ext == ".ply": p,c = _ply(raw)
    elif ext == ".pcd": p,c = _pcd(raw)
    elif ext in (".xyz", ".txt"):
        a = np.loadtxt(io.BytesIO(raw), ndmin=2)
        if a.shape[1] < 3: raise ValueError("XYZ benötigt mindestens drei Spalten.")
        p,c = a[:,:3], (a[:,3:6] if a.shape[1]>=6 else None)
    else: raise ValueError("Unterstützt: PLY, PCD, XYZ. E57/LAS/LAZ folgen als Adapter.")
    if not 3 <= len(p) <= MAX_POINTS: raise ValueError("Erlaubt sind 3 bis 1.000.000 Punkte.")
    if not np.isfinite(p).all(): raise ValueError("NaN/Inf in Punktkoordinaten; Import abgelehnt.")
    p = np.asarray(p,float) * {"m":1,"cm":.01,"mm":.001}[unit]
    if np.max(np.abs(p)) > 1e9: raise ValueError("Koordinaten außerhalb des unterstützten Bereichs.")
    if c is None:
        t = (p[:,2]-p[:,2].min()) / max(np.ptp(p[:,2]),1e-9)
        c = np.column_stack([.30+.25*t, .60+.18*t, .67+.22*t])*255
    elif np.max(c) <= 1: c = c*255
    c = np.nan_to_num(np.clip(c,0,255)).astype(np.uint8)
    return p,c,hashlib.sha256(raw).hexdigest()

def demo_room(seed=17, step=.055):
    rng = np.random.default_rng(seed)
    surfaces, colors = [], []
    def surface(p, color):
        surfaces.append(p+rng.normal(0,.003,p.shape))
        colors.append(np.tile(color,(len(p),1)) + rng.normal(0,5,(len(p),3)))
    x,y,z = np.arange(0,6+step/2,step),np.arange(0,4.5+step/2,step),np.arange(0,3+step/2,step)
    xx,yy=np.meshgrid(x,y); surface(np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)], [102,128,133])
    surface(np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,3.)], [112,142,150])
    for b in (0.,4.5):
        xx,zz=np.meshgrid(x,z); mask=np.ones(xx.shape,bool)
        if b==0: mask &= ~((xx>1.2)&(xx<2.2)&(zz<2.15))
        else: mask &= ~((xx>3.1)&(xx<4.7)&(zz>.9)&(zz<2.2))
        surface(np.c_[xx[mask],np.full(mask.sum(),b),zz[mask]], [143,178,184] if b else [123,158,166])
    yy,zz=np.meshgrid(y,z)
    for b in (0.,6.): surface(np.c_[np.full(yy.size,b),yy.ravel(),zz.ravel()],[139,168,175])
    p,c=np.concatenate(surfaces),np.clip(np.concatenate(colors),0,255).astype(np.uint8)
    return p,c

def write_ply(path, points, colors):
    header=f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with Path(path).open("w",encoding="ascii") as f:
        f.write(header)
        np.savetxt(f,np.c_[points,colors],fmt="%.6f %.6f %.6f %d %d %d")
