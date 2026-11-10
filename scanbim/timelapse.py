"""Server-side reconstruction film with measured analysis and true mesh sections."""
import logging
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .geometry import element_rectangles, unit
from .analysis_visuals import COLORS

_log=logging.getLogger(__name__)
_run_name=re.compile(r'timelapse-[0-9a-f]{12}')
_size=(960,600)
_MAIN=(20,66,610,552)
_SECTION=(634,66,940,552)
_PHASE={'detect_planes':'PLANES','cluster_points':'CLUSTERS','pca':'GEOMETRY',
        'estimate_normals':'NORMALS','run_experiment':'ANALYSIS','call_custom_tool':'ANALYSIS',
        'inspect_fit':'COMPARE','validate':'COMPARE','check_ifc':'IFC',
        'request_view':'SCAN','agent_start':'SCAN','agent_end':'IFC',
        'update_reconstruction_plan':'PLAN','review_model':'REVIEW'}


def mesh_section(vertices,faces,origin,normal):
    """Exact triangle/plane intersections in world coordinates, including coplanar edges."""
    triangles=np.asarray(vertices,float)[np.asarray(faces,int)]
    if not len(triangles):return np.empty((0,2,3))
    distances=(triangles-np.asarray(origin))@np.asarray(normal)
    active=(distances.min(axis=1)<=1e-8)&(distances.max(axis=1)>=-1e-8)
    segments=[]
    for triangle,d in zip(triangles[active],distances[active]):
        hits=[]
        for i,j in ((0,1),(1,2),(2,0)):
            if abs(d[i])<1e-8:hits.append(triangle[i])
            if d[i]*d[j]<-1e-16:hits.append(triangle[i]+(triangle[j]-triangle[i])*(d[i]/(d[i]-d[j])))
        unique=[]
        for point in hits:
            if not any(np.linalg.norm(point-other)<1e-8 for other in unique):unique.append(point)
        if len(unique)==2:segments.append(unique)
        elif len(unique)==3:
            segments.extend([[unique[i],unique[(i+1)%3]] for i in range(3)])
    return np.asarray(segments).reshape(-1,2,3)


class Recording:
    def __init__(self,scene):
        self.path=scene.folder/('timelapse-'+uuid.uuid4().hex[:12]);self.path.mkdir()
        self.points,self.colors,_,_=scene.sample(limit=50000)
        self.bounds=np.asarray([scene.bounds['min'],scene.bounds['max']])
        self.center=self.bounds.mean(axis=0);self.extent=max(np.linalg.norm(np.ptp(self.bounds,axis=0)),.1)
        self.count=0;self.revision=None;self.mesh_revision=None;self.meshes=[]
        # A stable scale and camera during reconstruction make changes easy to compare.
        self.rotation=self._rotation(0)
        self._fit_projection()

    def _rotation(self,angle):
        a=np.radians(angle);direction=np.array([-1.1,1.4,-1.])
        forward=unit([np.cos(a)*direction[0]-np.sin(a)*direction[1],
                      np.sin(a)*direction[0]+np.cos(a)*direction[1],direction[2]])
        right=unit(np.cross(forward,[0,0,1]));up=np.cross(right,forward)
        return np.stack([right,-up,forward])

    def _fit_projection(self):
        corners=np.array([[x,y,z] for x in self.bounds[:,0] for y in self.bounds[:,1] for z in self.bounds[:,2]])
        q=(corners-self.center)@self.rotation.T
        self.scale=min((_MAIN[2]-_MAIN[0]-54)/max(np.ptp(q[:,0]),.1),
                       (_MAIN[3]-_MAIN[1]-64)/max(np.ptp(q[:,1]),.1))

    def _project(self,points):
        p=(np.asarray(points)-self.center)@self.rotation.T
        p[:,:2]=p[:,:2]*self.scale+np.array([(_MAIN[0]+_MAIN[2])/2,(_MAIN[1]+_MAIN[3])/2])
        return p

    def _meshes(self,scene):
        if self.mesh_revision==scene.state['model_revision']:return
        self.meshes=[]
        if scene.state['elements']:
            try:
                from .ifc_export import inspect_ifc
                _,data=inspect_ifc(scene.state,meshes=True)
                for i,element in enumerate(scene.state['elements']):
                    mesh=data['elements'].get(element['id'])
                    if mesh and mesh.get('faces'):
                        self.meshes.append((np.asarray(mesh['vertices']),np.asarray(mesh['faces'],int),COLORS[i%len(COLORS)]))
            except (ValueError,RuntimeError):
                _log.exception('IFC film geometry could not be read')
        self.mesh_revision=scene.state['model_revision']

    def _section(self,scene,sweep=None):
        view=scene.state['views'][-1] if scene.state['views'] else {}
        section=view.get('section')
        if section and sweep is None:
            return np.asarray(section['origin']),unit(section['normal']),float(section['thickness'])
        axis=int(np.argmax(self.bounds[1,:2]-self.bounds[0,:2]));normal=np.eye(3)[axis]
        origin=self.center.copy()
        if sweep is not None:origin[axis]=self.bounds[0,axis]+sweep*(self.bounds[1,axis]-self.bounds[0,axis])
        return origin,normal,max((self.bounds[1,axis]-self.bounds[0,axis])*.018,.025)

    @staticmethod
    def _points(image,positions,colors,box):
        xy=np.rint(positions[:,:2]).astype(int)
        valid=(xy[:,0]>box[0]+1)&(xy[:,0]<box[2]-1)&(xy[:,1]>box[1]+1)&(xy[:,1]<box[3]-1)
        ids=np.flatnonzero(valid)
        if positions.shape[1]>2:ids=ids[np.argsort(-positions[ids,2])]
        pixels=np.array(image);pixels[xy[ids,1],xy[ids,0]]=np.asarray(colors,dtype=np.uint8)[ids]
        image.paste(Image.fromarray(pixels))

    def _render(self,scene,tool,sweep=None,angle=0):
        self.rotation=self._rotation(angle)
        image=Image.new('RGB',_size,'#08111e');draw=ImageDraw.Draw(image)
        for y in range(600):
            t=y/600;draw.line((0,y,960,y),fill=(int(11-4*t),int(23-8*t),int(38-12*t)))
        draw.rounded_rectangle(_MAIN,radius=18,fill='#0d1d2b',outline='#203648')
        draw.rounded_rectangle(_SECTION,radius=18,fill='#0b1a28',outline='#203648')
        colors=np.clip(self.colors*.22+np.array([20,35,48]),0,255)
        self._points(image,self._project(self.points),colors,_MAIN)
        layer=Image.new('RGBA',_size);ld=ImageDraw.Draw(layer)
        polygons=[]
        for vertices,faces,color in self.meshes:
            p=self._project(vertices)
            for index in np.linspace(0,len(faces)-1,min(len(faces),1600),dtype=int):
                triangle=p[faces[index]]
                normal=np.cross(vertices[faces[index,1]]-vertices[faces[index,0]],vertices[faces[index,2]]-vertices[faces[index,0]])
                shade=.62+.32*abs(float(normal@unit([.4,-.6,1])))/max(np.linalg.norm(normal),1e-12)
                rgb=tuple(int(c*shade) for c in color)
                polygons.append((triangle,rgb))
        for triangle,color in sorted(polygons,key=lambda item:-item[0][:,2].mean()):
            ld.polygon([tuple(p[:2]) for p in triangle],fill=(*color,225),outline=(*color,240))
        for overlay in ([] if tool=='agent_end' else scene.state['overlays'][-30:]):
            vertices=np.asarray(overlay['vertices']);p=self._project(vertices)
            rgb=tuple(overlay.get('colors',[COLORS[2]])[0])
            if overlay['kind']=='lines':
                for pair in p[:len(p)//2*2].reshape(-1,2,3):ld.line([tuple(q[:2]) for q in pair],fill=(*rgb,220),width=2)
            elif overlay['kind']=='mesh':
                for face in overlay['faces']:ld.polygon([tuple(q[:2]) for q in p[face]],fill=(*rgb,45),outline=(*rgb,170))
            else:
                for index in range(0,len(p),max(1,len(p)//1000)):
                    q=p[index];color=tuple(overlay['colors'][index]) if overlay.get('colors') else rgb
                    ld.ellipse((q[0]-1.2,q[1]-1.2,q[0]+1.2,q[1]+1.2),fill=(*color,235))
        clipped=Image.new("RGBA",_size)
        clipped.paste(layer.crop(_MAIN),(_MAIN[0],_MAIN[1]))
        image=Image.alpha_composite(image.convert('RGBA'),clipped)
        origin,normal,thickness=self._section(scene,sweep)
        u=unit(np.cross([0,0,1],normal)) if abs(normal[2])<.95 else unit(np.cross([0,1,0],normal))
        v=unit(np.cross(normal,u))
        basis=np.array([u,v]);q=(self.points-origin)@basis.T
        lo,hi=q.min(axis=0),q.max(axis=0);center=(lo+hi)/2
        scale=min(256/max(hi[0]-lo[0],.1),350/max(hi[1]-lo[1],.1))
        def project_section(points):
            uv=(np.asarray(points)-origin)@basis.T
            uv=(uv-center)*[scale,-scale]+[787,305]
            return uv
        # Exact source points in a finite slab, with the mathematical IFC cut overlaid.
        mask=np.abs((self.points-origin)@normal)<=thickness/2
        image=image.convert('RGB')
        self._points(image,project_section(self.points[mask]),np.tile([124,165,194],(int(mask.sum()),1)),_SECTION)
        glow=Image.new('RGBA',_size);gd=ImageDraw.Draw(glow)
        for vertices,faces,color in self.meshes:
            for segment in mesh_section(vertices,faces,origin,normal):
                gd.line([tuple(p) for p in project_section(segment)],fill=(*color,255),width=3)
        image=Image.alpha_composite(image.convert('RGBA'),glow.filter(ImageFilter.GaussianBlur(4)))
        image=Image.alpha_composite(image,glow)
        draw=ImageDraw.Draw(image)
        # A visible section plane ties the inset to its measured location in the model.
        corners=np.array([origin+u*x+v*y for x,y in ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))])
        pp=self._project(corners)
        draw.line([tuple(p[:2]) for p in pp]+[tuple(pp[0,:2])],fill='#ffba69',width=2)
        draw.text((24,22),'SCAN  /  IFC',fill='#ecf6ff',font_size=20)
        phase='SECTION' if sweep is not None else _PHASE.get(tool,'BUILD' if self.meshes else 'ANALYSIS')
        draw.text((786,26),phase,fill='#59e5cb',font_size=13)
        draw.text((40,84),'RECONSTRUCTION',fill='#9bb5ca',font_size=11)
        draw.text((654,84),'SECTION',fill='#ffba69',font_size=11)
        draw.text((654,520),f'{thickness*1000:.0f} mm',fill='#68899f',font_size=11)
        draw.text((777,520),'SCAN',fill='#7ca5c2',font_size=10)
        draw.text((854,520),'IFC',fill='#59e5cb',font_size=10)
        draw.text((24,572),f'{len(scene.state["elements"]):02d} PARTS',fill='#9bb5ca',font_size=12)
        progress=1. if tool=='agent_end' else min(.97,scene.state['agent'].get('step',0)/max(scene.state['agent'].get('max_steps',60),1))
        draw.rounded_rectangle((140,576,936,579),radius=2,fill='#21374a')
        if progress>0:draw.rounded_rectangle((140,576,140+796*progress,579),radius=2,fill='#59e5cb')
        return image.convert('RGB')

    def _save(self,image):
        target=self.path/f'{self.count:04d}.png';temporary=target.with_suffix('.tmp')
        image.save(temporary,format='PNG');temporary.replace(target);self.count+=1

    def capture(self,scene,tool):
        revision=scene.state['model_revision'];final=tool=='agent_end'
        analysis_step=tool in {'update_reconstruction_plan','request_view','run_experiment','call_custom_tool',
            'record_decision','remember','inspect_fit','validate','review_model','check_ifc',
            'detect_planes','cluster_points','pca','estimate_normals','get_points','measure'}
        if not final and ((revision==self.revision and not analysis_step) or self.count>=219):return
        self._meshes(scene)
        self._save(self._render(scene,tool));self.revision=revision
        if final and self.meshes:
            # This showcase only moves the camera and cutting plane; it invents no reconstruction steps.
            for t in np.linspace(0,1,18):
                self._save(self._render(scene,tool,sweep=.08+.84*t,angle=18*np.sin(t*np.pi)))


def record_event(scene, tool):
    """A recording error must not interrupt the reconstruction."""
    try:
        if tool == "agent_start":
            scene.recording = Recording(scene)
        if scene.recording is not None:
            scene.recording.capture(scene, tool)
    except Exception:
        _log.exception("Zeitraffer-Aufnahme fehlgeschlagen")
    finally:
        if tool == "agent_end":
            scene.recording = None


def frame_paths(folder):
    return sorted(p for p in folder.glob("*.png")
                  if re.fullmatch(r"[0-9]{4}\.png", p.name) and p.is_file() and not p.is_symlink())


def recordings(scene):
    runs = []
    for folder in scene.folder.iterdir():
        if not _run_name.fullmatch(folder.name) or folder.is_symlink() or not folder.is_dir():
            continue
        frames = frame_paths(folder)
        if frames:
            runs.append({"id": folder.name, "frames": len(frames), "started_at": frames[0].stat().st_mtime})
    return sorted(runs, key=lambda run: run["started_at"], reverse=True)


def recording_path(scene, run):
    folder = scene.folder / run
    if not _run_name.fullmatch(run) or folder.is_symlink() or not folder.is_dir():
        raise ValueError("Aufnahme nicht gefunden.")
    return folder


def export_gif(folder):
    frames = frame_paths(folder)
    if not frames:
        raise ValueError("Die Aufnahme enthält noch keine Bilder.")
    # Preserve start and finish in long runs while keeping GIF memory bounded.
    indices = np.linspace(0, len(frames)-1, min(120, len(frames)), dtype=int)
    # One shared palette prevents independent frame quantisation from flickering.
    atlas=Image.new("RGB",(256,144*min(8,len(indices))))
    for row,index in enumerate(np.linspace(0,len(indices)-1,min(8,len(indices)),dtype=int)):
        with Image.open(frames[indices[index]]) as frame:
            thumbnail=frame.convert("RGB");thumbnail.thumbnail((256,144))
            atlas.paste(thumbnail,(0,row*144))
    palette=atlas.quantize(colors=256)
    images = []
    for index in indices:
        with Image.open(frames[index]) as frame:
            images.append(frame.convert("RGB").quantize(palette=palette,dither=Image.Dither.NONE))
    target = folder / "zeitraffer.gif"
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        durations = [300] * len(images)
        durations[0] = 700
        durations[-1] = 1800
        images[0].save(temporary, format="GIF", save_all=True, append_images=images[1:],
                       duration=durations, loop=0)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
