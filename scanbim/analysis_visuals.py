"""Bounded, measured analysis overlays shared by live views and recordings."""
import numpy as np

COLORS=((89,229,203),(111,168,255),(255,186,105),(195,145,255),(255,125,163),(177,232,117))


def analysis_overlays(scene, tool, result):
    overlays=[]
    def add(kind,vertices,color,label,**extra):
        vertices=np.asarray(vertices,float)
        overlays.append(dict(kind=kind,vertices=vertices.tolist(),colors=[list(color)]*len(vertices),
                             label=label,source='local-analysis',**extra))
    if tool=='detect_planes':
        for i,plane in enumerate(result.get('planes',[])[:12]):
            o,u,v=(np.asarray(plane[k]) for k in ('origin','u','v'))
            w,h=plane['width'],plane['height'];color=COLORS[i%len(COLORS)]
            corners=np.array([o,o+u*w,o+u*w+v*h,o+v*h])
            add('mesh',corners,color,plane['id'],faces=[[0,1,2],[0,2,3]])
            add('lines',corners[[0,1,1,2,2,3,3,0]],color,plane['id'])
    elif tool=='pca':
        center=np.asarray(result['center']);axes=np.asarray(result['axes'])
        lo,hi=np.asarray(result['local_min']),np.asarray(result['local_max'])
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])@axes+center
        edges=[(i,i^(1<<axis)) for i in range(8) for axis in range(3) if not i&(1<<axis)]
        add('lines',corners[np.array(edges).ravel()],COLORS[0],'PCA bounds')
        for i,axis in enumerate(axes):
            add('lines',[center,center+axis*max((hi[i]-lo[i])*.55,.03)],COLORS[i],'PCA axis')
    elif tool=='measure' and 'coordinates' in result:
        add('lines',result['coordinates'],COLORS[2],'Measured distance')
        add('points',result['coordinates'],COLORS[2],'Measured endpoints')
    elif tool=='get_points':
        points=np.asarray(result['points']);ids=np.linspace(0,len(points)-1,min(len(points),2000),dtype=int)
        if len(ids):
            overlays.append(dict(kind='points',vertices=points[ids].tolist(),
                colors=np.asarray(result['colors'])[ids].tolist(),label='Source sample',source='local-analysis'))
    elif tool=='validate' and scene.state.get('quality'):
        quality=scene.state['quality'];ids=np.asarray(quality['sample_ids'],int)
        subset=np.linspace(0,len(ids)-1,min(len(ids),10000),dtype=int)
        distance=np.asarray(quality['distances_m'])[subset]
        ratio=np.clip(distance/max(float(result.get('tolerance_m',.03)),1e-9),0,2)/2
        rgb=np.array([89,229,203])*(1-ratio[:,None])+np.array([255,110,124])*ratio[:,None]
        overlays.append(dict(kind='points',vertices=scene.points[ids[subset]].tolist(),
            colors=rgb.astype(int).tolist(),label='Scan / model residuals',source='local-analysis'))
    elif tool=='inspect_fit':
        worst=result.get('worst_points',[])
        if worst:
            # Fit records store explicit world coordinates, never image-derived positions.
            points=[p.get('xyz',p.get('point',p.get('coordinates'))) if isinstance(p,dict) else p for p in worst]
            points=[p for p in points if p is not None]
            if points:add('points',points,(255,110,124),'Fit residuals')
    else:return
    if overlays:
        previous=[o for o in scene.state['overlays'] if o.get('source')!='local-analysis'][-6:]
        scene.state['overlays']=previous+overlays
