"""Trusted local segmentation; selections retain exact source point identities."""
import time
import uuid
import numpy as np
from scipy.spatial import cKDTree
from . import geometry
from .analysis_visuals import COLORS


def load_selection(scene, selection_id):
    if not isinstance(selection_id, str) or len(selection_id)!=32 or any(c not in '0123456789abcdef' for c in selection_id):
        raise ValueError('Ungueltige Punktauswahl-ID.')
    path=scene.folder/'point-selections'/(selection_id+'.npz')
    if not path.exists(): raise ValueError('Punktauswahl fehlt in dieser Szene.')
    with np.load(path, allow_pickle=False) as data:
        ids=data['source_indices'].copy()
        sampled=bool(data['sampled'])
    return ids, sampled


def save_selection(scene, ids, sampled):
    folder=scene.folder/'point-selections';folder.mkdir(exist_ok=True)
    selection_id=uuid.uuid4().hex
    np.savez_compressed(folder/(selection_id+'.npz'),source_indices=np.asarray(ids,dtype=np.int64),sampled=bool(sampled))
    return selection_id


def sample_selection(scene, region=None, limit=30000, selection_id=None):
    limit=int(limit)
    if not 1<=limit<=200000: raise ValueError('Analyse-Stichprobe: 1 bis 200000 Punkte.')
    if selection_id is None:
        points,colors,ids,total=scene.sample(region,limit=limit)
        return points,colors,ids,total,len(points)<total
    ids,previously_sampled=load_selection(scene,selection_id)
    if region:
        low,high=geometry.vector(region['min']),geometry.vector(region['max'])
        if np.any(low>=high): raise ValueError('Ungueltige Region.')
        p=scene.points[ids];ids=ids[np.all((p>=low)&(p<=high),axis=1)]
    total=len(ids)
    if total>limit: ids=ids[np.linspace(0,total-1,limit,dtype=int)]
    return scene.points[ids],scene.colors[ids],ids,total,previously_sampled or len(ids)<total


def dbscan(points, eps=.08, min_samples=8, *, stop=None, timeout_s=20):
    eps=float(eps);min_samples=int(min_samples)
    if not np.isfinite(eps) or not .001<=eps<=10 or not 2<=min_samples<=10000:
        raise ValueError('eps: 0.001 bis 10 Meter; min_samples: 2 bis 10000.')
    labels=np.full(len(points),-1,dtype=np.int32)
    if not len(points): return labels
    tree=cKDTree(points)
    counts=tree.query_ball_point(points,eps,return_length=True)
    if counts.sum()>5_000_000:
        raise ValueError('Zu viele Nachbarschaften. Kleinere Region, kleineres eps oder kleinere sample_limit waehlen.')
    core=counts>=min_samples;visited=np.zeros(len(points),bool)
    deadline=time.monotonic()+timeout_s;cluster=0
    for root in np.flatnonzero(core):
        if visited[root]: continue
        queue=[int(root)];visited[root]=True;labels[root]=cluster
        while queue:
            if time.monotonic()>deadline or (stop is not None and stop.is_set()):
                raise ValueError('Clustering angehalten oder Zeitlimit erreicht; kleinere Region waehlen.')
            index=queue.pop()
            neighbors=np.asarray(tree.query_ball_point(points[index],eps),dtype=int)
            labels[neighbors[labels[neighbors]<0]]=cluster
            expand=neighbors[core[neighbors]&~visited[neighbors]]
            visited[expand]=True;queue.extend(expand.tolist())
        cluster+=1
    return labels


def cluster_points(scene, region=None, selection_id=None, eps=.08, min_samples=8, min_cluster_points=30, sample_limit=30000):
    sample_limit=int(sample_limit)
    if not 1<=sample_limit<=50000: raise ValueError("Clustering-Stichprobe: 1 bis 50000 Punkte; groessere Szenen regional analysieren.")
    min_cluster_points=int(min_cluster_points)
    if min_cluster_points<1: raise ValueError('min_cluster_points muss positiv sein.')
    p,_,ids,total,sampled=sample_selection(scene,region,sample_limit,selection_id)
    labels=dbscan(p,eps,min_samples,stop=scene.stop)
    clusters=[];overlays=[]
    for label in range(int(labels.max())+1 if len(labels) else 0):
        mask=labels==label;part=p[mask]
        if len(part)<min_cluster_points: continue
        key=save_selection(scene,ids[mask],sampled)
        clusters.append({'selection_id':key,'point_count':len(part),
            'bounds':{'min':part.min(axis=0).tolist(),'max':part.max(axis=0).tolist()},
            'pca':geometry.pca(part) if len(part)>=3 else None})
        if len(overlays)<24:
            preview=part[np.linspace(0,len(part)-1,min(len(part),500),dtype=int)]
            overlays.append({'kind':'points','label':'Cluster '+str(label),'vertices':preview.tolist(),
                             'colors':[list(COLORS[label%len(COLORS)])]*len(preview),'source':'local-analysis'})
    scene.state['overlays']=overlays
    return {'method':'local_scipy_dbscan','eps_m':eps,'min_samples':min_samples,
        'sampling':{'analyzed':len(p),'region_total':total,'sampled':sampled},
        'noise_points':int(np.count_nonzero(labels<0)), 'clusters':clusters,
        'note':'Cluster sind geometrische Gruppen, keine Bauteilklassen. selection_id liefert genau diese Quellpunkte an detect_planes, pca und eigene Experimente. Bei Stichproben kleine Details regional nachuntersuchen.'}
