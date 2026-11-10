"""Geometry algorithms operate only on numeric 3D data, never image pixels."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree,ConvexHull

def vector(x):
    x=np.asarray(x,float)
    if x.shape != (3,) or not np.isfinite(x).all(): raise ValueError("Endlicher 3D-Vektor erforderlich.")
    return x

def unit(x):
    x=vector(x); length=np.linalg.norm(x)
    if length < 1e-10: raise ValueError("Nullvektor ist keine Richtung.")
    return x/length

def region_points(points, region=None):
    if not region: return points
    lo,hi=vector(region["min"]),vector(region["max"])
    if np.any(lo>=hi): raise ValueError("Region: min muss kleiner als max sein.")
    return points[np.all((points>=lo)&(points<=hi),axis=1)]

def pca(points):
    if len(points)<3: raise ValueError("Mindestens drei Punkte erforderlich.")
    center=points.mean(axis=0); values,axes=np.linalg.eigh(np.cov((points-center).T))
    order=np.argsort(values)[::-1]; axes=axes[:,order]
    q=(points-center)@axes
    return {"center":center.tolist(),"axes":axes.T.tolist(),"eigenvalues":values[order].tolist(),
            "local_min":q.min(axis=0).tolist(),"local_max":q.max(axis=0).tolist()}

def normals(points, k=16, limit=15000):
    p=points[np.linspace(0,len(points)-1,min(len(points),limit),dtype=int)]
    if len(p)<4: raise ValueError("Zu wenige Punkte für Normalen.")
    _,idx=cKDTree(points).query(p,k=min(max(4,k),len(points)))
    neighbors=points[idx]; centered=neighbors-neighbors.mean(axis=1,keepdims=True)
    cov=np.einsum("nki,nkj->nij",centered,centered)
    vals,vecs=np.linalg.eigh(cov)
    return {"points":p.tolist(),"normals":vecs[:,:,0].tolist(),"curvature":(vals[:,0]/np.maximum(vals.sum(axis=1),1e-12)).tolist()}

def detect_planes(points, threshold=.025, iterations=180, max_planes=12, min_points=180, seed=17):
    if not .001<=threshold<=.5 or not 10<=iterations<=2000 or not 1<=max_planes<=30:
        raise ValueError("RANSAC-Parameter außerhalb der Grenzen.")
    rng=np.random.default_rng(seed)
    remaining=points.copy(); planes=[]
    center=points.mean(axis=0)
    for _ in range(max_planes):
        if len(remaining)<max(3,min_points): break
        sample=remaining[rng.choice(len(remaining),min(7000,len(remaining)),replace=False)]
        best,bn,bd=0,None,None
        for _ in range(iterations):
            a,b,c=sample[rng.choice(len(sample),3,replace=False)]
            n=np.cross(b-a,c-a); length=np.linalg.norm(n)
            if length<1e-8: continue
            n/=length; d=-n@a
            score=np.count_nonzero(np.abs(sample@n+d)<threshold)
            if score>best: best,bn,bd=score,n,d
        if bn is None or best/len(sample)*len(remaining)<min_points: break
        mask=np.abs(remaining@bn+bd)<threshold
        support=remaining[mask]
        origin=np.median(support,axis=0)
        _,_,vh=np.linalg.svd(support-origin,full_matrices=False); n=vh[-1]
        if np.dot(n,center-origin)<0: n=-n
        if abs(n[2])>.9 and n[2]<0: n=-n
        d=-np.median(support@n)
        mask=np.abs(remaining@n+d)<threshold
        # Previously removed intersection strips still belong to this measured
        # surface. Fit bounds against all source inliers, not only residuals.
        support=points[np.abs(points@n+d)<threshold]
        vertical=abs(n[2])<.2
        u=unit(np.cross([0,0,1],n)) if vertical else unit(vh[0])
        v=unit(np.cross(n,u))
        if vertical and v[2]<0: u=-u;v=-v
        if not vertical:
            projected=support@np.column_stack([u,v])
            if len(projected)>=4:
                hull=projected[ConvexHull(projected).vertices]
                edges=np.roll(hull,-1,axis=0)-hull
                directions=edges/np.maximum(np.linalg.norm(edges,axis=1)[:,None],1e-12)
                areas=[]
                for direction in directions:
                    rotation=np.array([direction,[-direction[1],direction[0]]]).T
                    areas.append(np.prod(np.ptp(projected@rotation,axis=0)))
                direction=directions[int(np.argmin(areas))]
                u=unit(u*direction[0]+v*direction[1]);v=unit(np.cross(n,u))
        q=support@np.column_stack([u,v])
        lo,hi=np.quantile(q,.001,axis=0),np.quantile(q,.999,axis=0)
        origin=u*lo[0]+v*lo[1]-n*d
        planes.append({"id":f"plane-{len(planes)+1}","origin":origin.tolist(),"normal":n.tolist(),
                       "u":u.tolist(),"v":v.tolist(),"width":float(hi[0]-lo[0]),"height":float(hi[1]-lo[1]),
                       "kind":"wall" if vertical else ("slab" if abs(n[2])>.9 else "inclined"),
                       "support":len(support),"rms_m":float(np.sqrt(np.mean((support@n+d)**2))),
                       "threshold_m":threshold,"seed":seed})
        remaining=remaining[~mask]
    return planes

def plane_coordinates(points,plane):
    delta=points-vector(plane["origin"])
    return delta@np.array([plane["u"],plane["v"],plane["normal"]]).T

def find_openings(points,plane,cell=.08):
    if plane["kind"]!="wall": return []
    if not .03<=cell<=.3: raise ValueError("Rasterweite zwischen 0,03 und 0,3 m erforderlich.")
    q=plane_coordinates(points,plane); q=q[np.abs(q[:,2])<plane.get("threshold_m",.03)]
    w,h=plane["width"],plane["height"]
    nx,ny=int(np.ceil(w/cell)),int(np.ceil(h/cell))
    if min(nx,ny)<5 or nx*ny>1_000_000: return []
    grid,ex,ey=np.histogram2d(q[:,0],q[:,1],bins=[nx,ny],range=[[0,w],[0,h]])
    occupied=ndimage.binary_closing(grid>0,structure=np.ones((3,3)),border_value=1)
    missing=~occupied
    missing[:2,:]=False;missing[-2:,:]=False;missing[:,-2:]=False
    labels,n=ndimage.label(missing)
    openings=[]
    for i in range(1,n+1):
        xx,yy=np.where(labels==i)
        if len(xx)<12: continue
        x0,x1=ex[xx.min()],ex[xx.max()+1];y0,y1=ey[yy.min()],ey[yy.max()+1]
        rect=(xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)
        if x1-x0<.4 or y1-y0<.5 or len(xx)/rect<.75: continue
        if y0<.22: y0=0.
        openings.append({"x":float(x0),"y":float(y0),"width":float(x1-x0),"height":float(y1-y0),
                         "semantic_hint":"door" if y0==0 else "window","status":"hypothesis",
                         "confidence":.62,"resolution_m":cell,
                         "uncertainty":"Punktlücke kann auch durch Verdeckung entstehen; Füllung nicht beobachtet."})
    return openings

def element_rectangles(element):
    """Visible planar patches split at all opening boundaries; shared by export/view/QA."""
    w,h=element["width"],element["height"]
    holes=[hole for hole in element.get("openings",[]) if "x" in hole]
    xs=sorted(set([0.,w]+[np.clip(x,0,w) for o in holes for x in (o["x"],o["x"]+o["width"])]))
    ys=sorted(set([0.,h]+[np.clip(y,0,h) for o in holes for y in (o["y"],o["y"]+o["height"])]))
    rects=[]
    for x0,x1 in zip(xs,xs[1:]):
        for y0,y1 in zip(ys,ys[1:]):
            x,y=(x0+x1)/2,(y0+y1)/2
            if not any(o["x"]<x<o["x"]+o["width"] and o["y"]<y<o["y"]+o["height"] for o in holes):
                rects.append((float(x0),float(x1),float(y0),float(y1)))
    return rects

def plane_distance(points,element):
    q=plane_coordinates(points,element); best=np.full(len(q),np.inf)
    for x0,x1,y0,y1 in element_rectangles(element):
        dx=np.maximum(np.maximum(x0-q[:,0],0),q[:,0]-x1)
        dy=np.maximum(np.maximum(y0-q[:,1],0),q[:,1]-y1)
        best=np.minimum(best,np.sqrt(dx*dx+dy*dy+q[:,2]*q[:,2]))
    return best

def triangle_distance(points,vertices,faces):
    """Exact triangle distance, pruning subtrees with conservative box bounds."""
    vertices=np.asarray(vertices);faces=np.asarray(faces,dtype=int)
    triangles=vertices[faces]
    lower,upper=triangles.min(axis=1),triangles.max(axis=1)
    centers=(lower+upper)*.5
    best=cKDTree(vertices[np.unique(faces)]).query(points)[0]

    def visit(face_ids, point_ids):
        low,high=lower[face_ids].min(axis=0),upper[face_ids].max(axis=0)
        delta=np.maximum(np.maximum(low-points[point_ids],points[point_ids]-high),0)
        near=point_ids[np.einsum("ij,ij->i",delta,delta)<=best[point_ids]**2+1e-18]
        if not len(near):return
        if len(face_ids)>12:
            axis=np.argmax(high-low);mid=len(face_ids)//2
            split=face_ids[np.argpartition(centers[face_ids,axis],mid)]
            visit(split[:mid],near)
            visit(split[mid:],near)
        else:
            sample=points[near]
            distances=best[near]
            for face_id in face_ids:
                distances=np.minimum(distances,triangle_distances(sample,triangles[face_id]))
            best[near]=distances

    visit(np.arange(len(faces)),np.arange(len(points)))
    return best


def triangle_distances(points,triangle):
    a,b,c=triangle; ab,ac=b-a,c-a
    n=np.cross(ab,ac); ns=np.dot(n,n)
    if ns<1e-24:return np.full(len(points),np.inf)
    delta=points-a; signed=delta@n
    projected=delta-signed[:,None]*n/ns
    d00,d01,d11=ab@ab,ab@ac,ac@ac; denom=d00*d11-d01*d01
    s=(d11*(projected@ab)-d01*(projected@ac))/denom
    t=(d00*(projected@ac)-d01*(projected@ab))/denom
    inside=(s>=0)&(t>=0)&(s+t<=1)
    distance=np.where(inside,np.abs(signed)/np.sqrt(ns),np.inf)
    for p0,p1 in [(a,b),(b,c),(c,a)]:
        edge=p1-p0; along=np.clip((points-p0)@edge/max(edge@edge,1e-16),0,1)
        distance=np.minimum(distance,np.linalg.norm(points-p0-along[:,None]*edge,axis=1))
    return distance


def mesh_probes(vertices,faces):
    triangles=vertices[faces]
    a,b,c=triangles.transpose(1,0,2)
    area=np.linalg.norm(np.cross(b-a,c-a),axis=1)
    rng=np.random.default_rng(41)
    if area.sum()==0:return vertices[:1]
    indices=rng.choice(len(triangles),size=min(12000,max(64,len(faces)*12)),p=area/area.sum())
    st=rng.random((len(indices),2))
    st[st.sum(axis=1)>1]=1-st[st.sum(axis=1)>1]
    return a[indices]+st[:,:1]*(b-a)[indices]+st[:,1:]*(c-a)[indices]

def validate(points,elements,tolerance=.03):
    if not elements: raise ValueError("Zunächst Modellgeometrie erzeugen.")
    if not .001<=tolerance<=1: raise ValueError("Ungültige Toleranz.")
    sample_ids=np.linspace(0,len(points)-1,min(len(points),80000),dtype=int)
    p=points[sample_ids]; distances=[]; report=[]; tree=cKDTree(points)
    for e in elements:
        if e["kind"]=="mesh":
            verts=np.asarray(e["vertices"]);faces=np.asarray(e["faces"],int)
            d=triangle_distance(p,verts,faces)
            probes=mesh_probes(verts,faces)
        else:
            d=plane_distance(p,e);probes=[]
            for x0,x1,y0,y1 in element_rectangles(e):
                xx,yy=np.meshgrid(np.linspace(x0,x1,min(60,max(2,int((x1-x0)/.15)))),
                                  np.linspace(y0,y1,min(60,max(2,int((y1-y0)/.15)))))
                probes.extend(np.asarray(e["origin"])+xx.ravel()[:,None]*e["u"]+yy.ravel()[:,None]*e["v"])
        distances.append(d)
        supported=d<max(.1,tolerance*3)
        nearest=tree.query(np.asarray(probes))[0] if len(probes) else np.array([np.inf])
        # Coverage is sampled surface support, not scanner visibility.
        report.append({"id":e["id"],"rms_m":float(np.sqrt(np.mean(d[supported]**2))) if supported.any() else None,
                       "p95_m":float(np.quantile(d[supported],.95)) if supported.any() else None,
                       "surface_coverage":float(np.mean(nearest<max(.09,tolerance*2))),
                       "supported_points":int(supported.sum()),"unobserved_fraction":float(np.mean(nearest>=max(.09,tolerance*2)))})
    distances=np.asarray(distances)
    assignment=np.argmin(distances,axis=0)
    d=np.min(distances,axis=0)
    # Attribute scan points to their closest finite surface before reporting
    # per-element residuals; neighbouring walls must not inflate each other.
    for index,item in enumerate(report):
        assigned=d[assignment==index]
        item["rms_m"]=float(np.sqrt(np.mean(assigned*assigned))) if len(assigned) else None
        item["p95_m"]=float(np.quantile(assigned,.95)) if len(assigned) else None
        item["supported_points"]=int(np.count_nonzero(assigned<=tolerance))
        item["assigned_points"]=len(assigned)
    return {"rms_m":float(np.sqrt(np.mean(d*d))),"p95_m":float(np.quantile(d,.95)),
            "within_tolerance":float(np.mean(d<=tolerance)),"tolerance_m":tolerance,"sample_count":len(p),
            "elements":report,"sample_ids":sample_ids.tolist(),"distances_m":d.tolist(),
            "coverage_note":"Geometrische Oberflächenstützung; keine Sichtbarkeits- oder Vollständigkeitsgarantie."}
