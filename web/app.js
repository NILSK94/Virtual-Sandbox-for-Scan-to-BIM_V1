import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

// Older open tabs can combine cached HTML with a newer module after an update.
const uiBuild='20261109-live-guidance-1';
const requiredControls=['settings-provider','timelapse-button','agent-guide-form'];
if(requiredControls.some(id=>!document.getElementById(id)) ||
   document.querySelector('meta[name="scanbim-ui-build"]')?.content!==uiBuild){
    const fresh=new URL(window.location.href);
    if(fresh.searchParams.get('ui')!==uiBuild){
        fresh.searchParams.set('ui',uiBuild);
        window.location.replace(fresh.href);
    }else{
        const message=document.createElement('p');
        message.textContent='Die Studio-Oberfläche ist veraltet. Bitte Start.bat neu starten und die Seite mit Strg+F5 laden.';
        message.setAttribute('role','alert');
        document.body.prepend(message);
    }
    throw new Error('Studio-Oberfläche wird aktualisiert.');
}

const $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=0)=>Number(n).toLocaleString('de-DE',{maximumFractionDigits:d,minimumFractionDigits:d});
let state=null,pointData=null,events=[],stream=null,activeTab='agent',selected=null,config={},replay=null,overlay=false,lastView=null,lastExperiment=null;
let cameraMotion=null,focusRegion=null,activitySeq=-1,lastFocus=null,resetDraft=null,requestedSection=null;
let measuring=false,measureIds=[],refreshing=false,refreshAgain=false,sceneLoadRevision=0;
const raycaster=new THREE.Raycaster();raycaster.params.Points.threshold=.065;
let toastTimer;
let ifcView=null,ifcKey=null,ifcRequest=null,replaySeq=null,scanEvidence=null,bimMode=false;
let activeImport=null,uploadRequest=null,importPollTimer=null;
const fileSize=n=>n>=1073741824?fmt(n/1073741824,1)+' GiB':fmt(n/1048576,1)+' MiB';
function toast(text,error=false){$('toast').textContent=text;$('toast').classList.remove('hidden');$('toast').classList.toggle('error',error);clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),error?10000:5000);}
async function api(path,options={}){let headers={'X-ScanBIM-Client':'studio',...(options.headers||{})};if(options.body&&!(options.body instanceof FormData))headers['Content-Type']='application/json';const r=await fetch(path,{...options,headers});if(!r.ok){let detail;try{detail=(await r.json()).detail;}catch{detail=r.statusText;}throw Error(typeof detail==='string'?detail:JSON.stringify(detail));}return r.json();}
async function action(tool,args={}){if(!state)throw Error('Zuerst eine Punktwolke laden.');const r=await api(`/api/scenes/${state.id}/action`,{method:'POST',body:JSON.stringify({tool,arguments:args})});await refresh();return r;}
function safely(fn){return async (...args)=>{try{await fn(...args);}catch(e){toast(e.message,true);}};}

class Viewport{
 constructor(id,type){this.holder=$(id);this.type=type;this.scene=new THREE.Scene();this.camera=new THREE.PerspectiveCamera(45,1,.01,10000);this.camera.up.set(0,0,1);this.camera.position.set(9,-10,8);this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.setClearColor(0,0);this.renderer.localClippingEnabled=true;this.holder.append(this.renderer.domElement);this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=false;this.controls.target.set(3,2.25,1);this.controls.minDistance=.02;this.controls.addEventListener('change',()=>this.sync());this.controls.addEventListener('start',()=>{cameraMotion=null;requestedSection=null;$('follow-camera').checked=false;renderFollow();clipping();});this.content=new THREE.Group();this.scene.add(this.content);this.scene.add(new THREE.HemisphereLight(0xcde7ed,0x344a50,2));const light=new THREE.DirectionalLight(0xffffff,2.3);light.position.set(3,-4,8);this.scene.add(light);this.grid=new THREE.GridHelper(30,30,0x3b5868,0x233947);this.grid.rotation.x=Math.PI/2;this.grid.position.z=-.06;this.grid.material.transparent=true;this.grid.material.opacity=.3;this.scene.add(this.grid);new ResizeObserver(()=>this.resize()).observe(this.holder);this.resize();this.renderer.domElement.addEventListener('pointerdown',e=>{this.startPointer=[e.clientX,e.clientY];});this.renderer.domElement.addEventListener('pointerup',e=>{if(this.startPointer&&Math.hypot(e.clientX-this.startPointer[0],e.clientY-this.startPointer[1])<5)this.pick(e);});}
 resize(){const w=this.holder.clientWidth,h=this.holder.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;if(this.camera.isOrthographicCamera){const span=this.orthoSpan||10;this.camera.left=-span/2;this.camera.right=span/2;this.camera.top=span/2*h/w;this.camera.bottom=-span/2*h/w;}this.camera.updateProjectionMatrix();}
 sync(){if(this.syncing)return;const other=this===scan?model:scan;if(!other)return;other.syncing=true;other.camera.position.copy(this.camera.position);other.camera.quaternion.copy(this.camera.quaternion);other.camera.up.copy(this.camera.up);other.controls.target.copy(this.controls.target);other.controls.update();other.syncing=false;}
 clear(){while(this.content.children.length){const obj=this.content.children[0];this.content.remove(obj);obj.traverse(n=>{n.geometry?.dispose();if(Array.isArray(n.material))n.material.forEach(m=>m.dispose());else n.material?.dispose();});}}
 setProjection(ortho){const old=this.camera;const w=this.holder.clientWidth||600,h=this.holder.clientHeight||400;this.orthoSpan=old.position.distanceTo(this.controls.target)*.9;const cam=ortho?new THREE.OrthographicCamera(-5,5,5,-5,.01,10000):new THREE.PerspectiveCamera(45,w/h,.01,10000);cam.position.copy(old.position);cam.quaternion.copy(old.quaternion);cam.up.copy(old.up);this.camera=cam;this.controls.object=cam;this.resize();this.controls.update();}
 pick(e){if(!state)return;const rect=this.renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),this.camera);const hits=raycaster.intersectObjects(this.content.children,true);if(measuring&&this===scan){const hit=hits.find(h=>h.object.userData.cloud);if(!hit)return;const source=pointData.source_indices[hit.object.userData.pointMap[hit.index]];measureIds.push(source);toast(measureIds.length===1?'Erster Quellpunkt gewählt. Zweiten Punkt anklicken.':'Messung wird berechnet.');if(measureIds.length===2)safely(async()=>{const result=await action('measure',{indices:measureIds});toast(`Abstand: ${fmt(result.distance_m,4)} m · ΔZ: ${fmt(result.delta_m[2],4)} m`);measuring=false;measureIds=[];$('measure-button').classList.remove('active');})();return;}const hit=hits.find(h=>h.object.userData.elementId);if(hit){selected=hit.object.userData.elementId;renderInspector();drawModel();}}
 render(){this.renderer.render(this.scene,this.camera);}
}
const scan=new Viewport('scan-view','scan'),model=new Viewport('model-view','model');
function animate(){requestAnimationFrame(animate);if(cameraMotion){const t=Math.min((performance.now()-cameraMotion.start)/700,1),k=t*t*(3-2*t);for(let i=0;i<2;i++){const v=[scan,model][i],a=cameraMotion.from[i],b=cameraMotion.to;v.syncing=true;v.camera.position.lerpVectors(a.position,b.position,k);v.controls.target.lerpVectors(a.target,b.target,k);v.camera.up.lerpVectors(a.up,b.up,k).normalize();v.orthoSpan=THREE.MathUtils.lerp(a.span,b.span,k);v.camera.lookAt(v.controls.target);v.resize();v.controls.update();v.syncing=false;}if(t===1)cameraMotion=null;}scan.render();model.render();}animate();
function worldOffset(){return new THREE.Vector3(...(state?.render_origin||[0,0,0]));}
function bounds(){return state?{lo:new THREE.Vector3(...state.bounds.min).sub(worldOffset()),hi:new THREE.Vector3(...state.bounds.max).sub(worldOffset())}:{lo:new THREE.Vector3(0,0,0),hi:new THREE.Vector3(6,4.5,3)};}
function cameraPreset(kind='iso'){cameraMotion=null;requestedSection=null;const{lo,hi}=bounds(),center=lo.clone().add(hi).multiplyScalar(.5),size=hi.clone().sub(lo).length();const direction={iso:[1,-1.4,1],top:[0,0,1],front:[0,-1,.03]}[kind];for(const v of[scan,model]){v.syncing=true;v.camera.up.set(...(kind==='top'?[0,1,0]:[0,0,1]));v.controls.target.copy(center);v.camera.position.copy(center).add(new THREE.Vector3(...direction).normalize().multiplyScalar(size*1.25));v.orthoSpan=size*1.15;v.camera.lookAt(center);v.controls.update();v.resize();v.syncing=false;}}
function applyCamera(view,smooth=true){if(!view)return;cameraMotion=null;const from=[scan,model].map(v=>({position:v.camera.position.clone(),target:v.controls.target.clone(),up:v.camera.up.clone(),span:v.orthoSpan||10}));for(const v of[scan,model]){v.syncing=true;v.setProjection(view.projection==='orthographic');if(view.fov_degrees&&v.camera.isPerspectiveCamera){v.camera.fov=view.fov_degrees;v.camera.updateProjectionMatrix();}v.syncing=false;}const to={position:new THREE.Vector3(...view.position).sub(worldOffset()),target:new THREE.Vector3(...view.target).sub(worldOffset()),up:new THREE.Vector3(...view.up),span:view.span_m};cameraMotion={from,to,start:performance.now()-(smooth?0:700)};requestedSection=view.section||null;$('slice-range').value=100;clipping();$('ortho-button').classList.toggle('active',view.projection==='orthographic');}
function focusBounds(region){if(!region)return;const lo=new THREE.Vector3(...region.min),hi=new THREE.Vector3(...region.max),target=lo.clone().add(hi).multiplyScalar(.5);const span=Math.max(hi.distanceTo(lo),.3);let direction=scan.camera.position.clone().sub(scan.controls.target).normalize();if(!direction.length())direction.set(1,-1,1).normalize();applyCamera({position:target.clone().addScaledVector(direction,span*1.3).toArray(),target:target.toArray(),up:[0,0,1],projection:scan.camera.isOrthographicCamera?'orthographic':'perspective',span_m:span*1.2});}
function elementBounds(e){if(!e)return null;let pts;if(e.kind==='mesh')pts=e.vertices;else{pts=[];for(const x of[0,e.width])for(const y of[0,e.height])for(const z of[0,(e.depth_direction??-1)*e.thickness])pts.push(e.origin.map((n,i)=>n+e.u[i]*x+e.v[i]*y+e.normal[i]*z));}const box=new THREE.Box3().setFromPoints(pts.map(p=>new THREE.Vector3(...p)));return{min:box.min.toArray(),max:box.max.toArray()};}
function renderFollow(){$('process-follow').textContent=$('follow-camera').checked?'Kameraführung aktiv · beim Drehen pausiert':'Freie Kamera · „Agentenkamera folgen“ zum Zurückkehren';}
function renderActivity(a){if(!a||replay)return;const stages={idle:'BEREIT',planning:'MODELL WERTET AUS',working:'WERKZEUG LÄUFT',done:'ERGEBNIS VORHANDEN',error:'FEHLER',partial:'ZWISCHENSTAND',completed:'ABGESCHLOSSEN',stopped:'ANGEHALTEN'};$('process-stage').textContent=stages[a.status]||a.status;$('process-stage').classList.toggle('busy',['planning','working'].includes(a.status));$('process-tool').textContent=toolLabels[a.tool]||a.tool||'Freie Untersuchung';$('process-message').textContent=a.message||'';const e=state?.elements.find(e=>e.id===a.element_id);focusRegion=a.region||elementBounds(e);$('process-focus').textContent=e?'Bauteil: '+e.name:focusRegion?'Ausgewählter Arbeitsbereich':'Gesamte Szene';const key=JSON.stringify([a.tool,a.region,a.element_id,!!e]);if(key!==lastFocus){lastFocus=key;if(e){selected=e.id;renderInspector();}if($('follow-camera').checked&&focusRegion&&a.tool!=='request_view')focusBounds(focusRegion);}activitySeq=a.seq;renderFollow();}
function renderAgentView(view){$('agent-view-button').disabled=!view;$('agent-view-image').classList.toggle('hidden',!view);if(!view){$('agent-view-image').removeAttribute('src');$('agent-view-label').textContent='Noch keine Ansicht angefordert';$('agent-view-meta').textContent='Ansichtsbild und Metadaten erscheinen hier.';return;}const img=$('agent-view-image');if(img.getAttribute('src')!==view.url)img.src=view.url;$('agent-view-label').textContent='Letzte angeforderte Ansicht · '+view.label;$('agent-view-meta').textContent=`${view.projection} · ${fmt(view.point_count)} Punkte${view.section?' · Schnitt':''}${view.region?' · Region':''}`;$('agent-view-button').title=JSON.stringify({position:view.position,target:view.target,up:view.up,section:view.section,source_hash:view.source_hash});}
function isCeiling(e){return e.kind!=='mesh'&&Math.abs(e.normal[2])>.9&&e.origin[2]>(state.bounds.min[2]+state.bounds.max[2])/2;}
function clipping(){if(!state)return;const value=Number($('slice-range').value);const z=state.bounds.min[2]+(state.bounds.max[2]-state.bounds.min[2])*value/100;let planes=value<100?[new THREE.Plane(new THREE.Vector3(0,0,-1),z-(state.render_origin?.[2]||0))]:[];if(requestedSection){const n=new THREE.Vector3(...requestedSection.normal).normalize(),o=new THREE.Vector3(...requestedSection.origin).sub(worldOffset()),d=n.dot(o),h=requestedSection.thickness/2;planes=[new THREE.Plane(n.clone(),-d+h),new THREE.Plane(n.clone().negate(),d+h)];}for(const v of[scan,model])v.renderer.clippingPlanes=planes;$('slice-value').value=value===100?'Aus':`${fmt(z,2)} m`;}
function addCloud(view){if(!pointData||!state)return;const positions=[],colors=[],map=[],qa=new Map();if($('heatmap').checked&&state.quality)state.quality.sample_ids.forEach((id,i)=>qa.set(id,state.quality.distances_m[i]));const qcolor=new THREE.Color();const ceiling=state.bounds.max[2]-.045;const renderOffset=state.render_origin||[0,0,0];for(let i=0;i<pointData.points.length/3;i++){let p=pointData.points.slice(i*3,i*3+3);if($('hide-ceiling').checked&&p[2]>ceiling)continue;positions.push(p[0]-renderOffset[0],p[1]-renderOffset[1],p[2]-renderOffset[2]);map.push(i);const distance=qa.get(pointData.source_indices[i]);if($('heatmap').checked&&distance!==undefined){const t=Math.min(distance/.06,1);qcolor.setRGB(.48+.44*t,.84-.38*t,.68-.23*t);colors.push(qcolor.r,qcolor.g,qcolor.b);}else colors.push(...pointData.colors.slice(i*3,i*3+3).map(x=>x/255));}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));const material=new THREE.PointsMaterial({size:2,vertexColors:true,sizeAttenuation:false,transparent:overlay,opacity:overlay ? 0.6 : 1});const cloud=new THREE.Points(geometry,material);cloud.userData={cloud:true,pointMap:map};view.content.add(cloud);}
function addRegion(view){const region=replay?replay.region:(focusRegion||state?.region);if(!region)return;const min=new THREE.Vector3(...region.min).sub(worldOffset()),max=new THREE.Vector3(...region.max).sub(worldOffset());const helper=new THREE.Box3Helper(new THREE.Box3(min,max),0xdfc17d);view.content.add(helper);}
function drawScan(){
    scan.clear();
    addCloud(scan);
    addRegion(scan);
    drawOverlays(scan);
    if(scanEvidence?.key!==ifcViewKey()||scanEvidence?.element!==selected)return;
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(
        scanEvidence.points.flatMap(v=>v.map((x,i)=>x-(state.render_origin?.[i]||0))),3));
    scan.content.add(new THREE.Points(geometry,new THREE.PointsMaterial({
        size:5,color:0xf2cb6d,sizeAttenuation:false,depthTest:false})));
}
async function toggleScanEvidence(){
    const key=ifcViewKey(),element=selected;
    if(scanEvidence?.key===key&&scanEvidence.element===element){
        scanEvidence=null;
    }else{
        const path='/api/scenes/'+state.id+'/ifc-evidence/'+encodeURIComponent(element)+(replay?'?seq='+replaySeq:'');
        const result=await api(path);
        if(key!==ifcViewKey()||element!==selected)return;
        scanEvidence={...result,key,element};
        toast(result.points.length+' stützende Prüfpunkte · '+fmt(result.tolerance_m*1000)+' mm Toleranz');
    }
    drawScan();
    renderInspector();
}
function ifcViewKey(){
    if(!state)return null;
    const revision=replay?replay.model_revision:state.model_revision;
    return [state.id,state.generation,revision??('history-'+replaySeq),bimMode].join('/');
}
function coordination(){return replay?replay.coordination:state?.coordination;}
function syncIfc(){
    const key=ifcViewKey();
    if(!key||ifcKey===key||ifcRequest?.key===key)return;
    ifcRequest?.controller.abort();
    const controller=new AbortController();
    ifcRequest={key,controller};
    const path='/api/scenes/'+state.id+'/ifc-model?bim='+bimMode+(replay?'&seq='+replaySeq:'');
    api(path,{signal:controller.signal}).then(report=>{
        if(key!==ifcViewKey()||controller.signal.aborted)return;
        if(!replay&&report.model_revision!==state.model_revision)return;
        ifcView=report;
        ifcKey=key;
        updateBimControls();
        $('ifc-schema').textContent=report.schema==='IFC4X3_ADD2'?'IFC 4.3':report.schema;
        drawModel();
        if(!document.querySelector('.inspector input:focus'))renderInspector();
        if(activeTab==='ifc')renderDock();
    }).catch(error=>{
        if(!controller.signal.aborted){
            $('ifc-empty-message').textContent=error.message;
            toast(error.message,true);
        }
    }).finally(()=>{if(ifcRequest?.controller===controller)ifcRequest=null;});
}
function drawModel(){
    model.clear();
    if(!state)return;
    if(overlay)addCloud(model);
    const elements=replay?.elements??state.elements;
    const current=ifcKey===ifcViewKey()?ifcView:null;
    const clashes=coordination()?.clashes||[];
    const colliding=new Set(clashes.flatMap(clash=>clash.elements));
    let shown=0;
    for(const e of elements){
        const shape=current?.elements[e.id];
        if(!shape||($('hide-ceiling').checked&&isCeiling(e)))continue;
        const assumed=e.parameter_status?.geometry==='assumed';
        const color=e.id===selected?0xe2c991:colliding.has(e.id)?0xd57970:assumed?0xb89773:e.semantic==='terrain'?0x7f9f70:shape.closed?0x87b5ae:0xa7c4d8;
        const material=new THREE.MeshStandardMaterial({color,roughness:.72,side:THREE.DoubleSide,
            transparent:overlay||assumed,opacity:overlay ? .46 : assumed ? .56 : 1});
        const geometry=new THREE.BufferGeometry();
        geometry.setAttribute('position',new THREE.Float32BufferAttribute(shape.vertices.flatMap(v=>v.map((x,i)=>x-(state.render_origin?.[i]||0))),3));
        geometry.setIndex(shape.faces.flat());
        geometry.computeVertexNormals();
        const mesh=new THREE.Mesh(geometry,material);
        mesh.userData.elementId=e.id;
        model.content.add(mesh);
        if(shape.faces.length<3000)addEdges(mesh);
        shown++;
    }
    for(const clash of clashes){
        if(bimMode&&!clash.elements.every(id=>current?.elements[id]))continue;
        const marker=new THREE.Mesh(new THREE.SphereGeometry(Math.max(.015,Math.min(clash.depth_m,.15)),12,8),
            new THREE.MeshBasicMaterial({color:0xfb8e80,depthTest:false}));
        marker.position.set(...clash.point).sub(worldOffset());
        model.content.add(marker);
    }
    if(!bimMode)drawOverlays(model);
    if(!bimMode)addRegion(model);
    $('model-empty').classList.toggle('hidden',shown>0);
    $('ifc-empty-message').textContent=current?.issues[0]||
        (current&&bimMode?'BIM-Modell erstellen, um Volumenkörper abzuleiten.':elements.length?'IFC-Körper werden geladen …':'Untersuchung starten oder ein Versuchsergebnis übernehmen.');
}
function addEdges(mesh){const edge=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:0xb4d5d3,transparent:true,opacity:.26}));mesh.add(edge);}
function drawOverlays(view){if(!$('show-attempts').checked)return;for(const o of(replay?.overlays??state?.overlays??[])){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(o.vertices.flatMap(v=>v.map((x,i)=>x-(state.render_origin?.[i]||0))),3));if(o.colors)geometry.setAttribute('color',new THREE.Float32BufferAttribute(o.colors.flat().map(x=>x/255),3));const colored=!!o.colors;let obj;if(o.kind==='mesh'){geometry.setIndex(o.faces.flat());geometry.computeVertexNormals();obj=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:colored?0xffffff:0xd3b981,vertexColors:colored,transparent:true,opacity:.55,side:THREE.DoubleSide}));}else if(o.kind==='lines')obj=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:colored?0xffffff:0xe2c389,vertexColors:colored}));else obj=new THREE.Points(geometry,new THREE.PointsMaterial({color:colored?0xffffff:0xe2c389,vertexColors:colored,size:4,sizeAttenuation:false}));view.content.add(obj);}}

async function loadScene(sid){const token=++sceneLoadRevision;if(stream)stream.close();const[nextState,nextPoints,nextEvents]=await Promise.all([api(`/api/scenes/${sid}`),api(`/api/scenes/${sid}/points`),api(`/api/scenes/${sid}/events`)]);if(token!==sceneLoadRevision)return;state=nextState;pointData=nextPoints;events=nextEvents;selected=null;replay=null;lastView=null;lastFocus=null;focusRegion=null;cameraMotion=null;activitySeq=-1;localStorage.setItem('scanbim-scene',sid);$('scene-select').value=sid;for(const viewport of[scan,model])viewport.grid.position.z=state.bounds.min[2]-(state.render_origin?.[2]||0)-.06;cameraPreset();render();stream=new EventSource(`/api/scenes/${sid}/stream`);stream.onmessage=event=>{const message=JSON.parse(event.data);renderActivity(message.activity);if(message.revision!==state.revision||message.generation!==state.generation)refresh().catch(e=>toast(e.message,true));};}
async function refresh(){if(!state)return;if(refreshing){refreshAgain=true;return;}refreshing=true;try{const sid=state.id,loadToken=sceneLoadRevision;const[newState,newEvents]=await Promise.all([api(`/api/scenes/${sid}`),api(`/api/scenes/${sid}/events?after=${events.length}`)]);if(state.id!==sid||loadToken!==sceneLoadRevision)return;if(state.generation!==newState.generation){events=await api(`/api/scenes/${sid}/events`);selected=null;replay=null;lastView=null;lastFocus=null;focusRegion=null;lastExperiment=null;cameraMotion=null;requestedSection=null;localStorage.removeItem('scanbim-experiment-code');}else events.push(...newEvents);state=newState;render();}finally{refreshing=false;if(refreshAgain){refreshAgain=false;refresh();}}}
async function updateScenes(){const scenes=await api('/api/scenes');$('scene-select').innerHTML='<option value="">Szene auswählen</option>'+scenes.map(s=>`<option value="${s.id}">${esc(s.name)}</option>`).join('');if(state)$('scene-select').value=state.id;return scenes;}
function render(){if(!state)return;const quality=state.quality,agent=state.agent;const elements=replay?.elements??state.elements;$('scene-name').textContent=state.name;$('source-label').textContent=state.name;$('source-detail').textContent=`${fmt(state.point_count)} Punkte · ${state.source_bytes?fileSize(state.source_bytes)+' · ':''}${state.storage==='disk-mapped'?'Festplattencache':state.source_hash.slice(0,8)}`;$('metric-points').textContent=fmt(state.point_count);$('display-count').textContent=`${fmt(pointData.displayed)} / ${fmt(state.point_count)} PUNKTE`;$('metric-elements').innerHTML=`${elements.length} <em>Elemente</em>`;$('metric-openings').textContent=`${elements.filter(e=>e.kind==='wall').length} Wände · ${elements.reduce((n,e)=>n+(e.openings?.length||0),0)} Öffnungen`;$('metric-rms').innerHTML=`${quality?fmt(quality.rms_m*1000,1):'—'} <em>mm RMS</em>`;$('metric-quality').textContent=quality?`${fmt(quality.within_tolerance*100,1)} % von ${fmt(quality.sample_count)} Prüfpunkten ≤ ${fmt(quality.tolerance_m*1000)} mm`:'Validierung ausstehend';const names={idle:'Bereit',running:'Untersucht',paused:'Pausiert',completed:'Abgeschlossen',error:'Fehler',stopped:'Angehalten',partial:'Zwischenstand'};$('metric-status').textContent=names[agent.status]||agent.status;$('metric-step').textContent=agent.step?`Schritt ${agent.step} · ${agent.mode==='demo'?'Demo-Prüflauf':'Visueller LLM'}`:'Jede Aktion wird protokolliert';$('agent-title').textContent=agent.mode==='vlm'?'Visueller Agent · freie Werkzeugwahl':agent.mode==='demo'?'Demo-Agent · reproduzierbarer Prüflauf':'Die nächste Frage an die Struktur.';$('agent-message').textContent=agent.message;renderTokenUsage(agent.token_usage);renderReconstructionPlan(state.reconstruction_plan);renderGuidance();const running=['running','paused'].includes(agent.status);$('start-button').disabled=running;$('reset-button').disabled=running;$('pause-button').classList.toggle('hidden',!running);$('stop-button').classList.toggle('hidden',!running);$('pause-button').textContent=agent.status==='paused'?'▶':'Ⅱ';$('pause-button').title=agent.status==='paused'?'Untersuchung fortsetzen':'Untersuchung pausieren';$('exp-count').textContent=state.experiments.length;$('action-count').textContent=events.length;$('replay-range').max=Math.max(events.length-1,0);if(!replay){$('replay-range').value=Math.max(events.length-1,0);$('replay-label').textContent='Live';}$('validation-legend').classList.toggle('hidden',!$('heatmap').checked||!quality);if(!replay)renderActivity(state.activity);updateBimControls();syncIfc();drawScan();drawModel();clipping();renderDock();if(!document.querySelector('.inspector input:focus'))renderInspector();const view=replay?replay.view:state.views.at(-1);renderAgentView(view);if(!replay&&view&&view.id!==lastView){lastView=view.id;if($('follow-camera').checked)applyCamera(view);}}
function setTab(tab){activeTab=tab;document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));renderDock();}
const toolLabels={user_guidance:'Nutzerhinweis',cluster_points:'Punkte clustern',update_reconstruction_plan:'Rekonstruktionsplan aktualisieren',model_usage:'Tokenverbrauch',read_agent_result:'Ergebnis gezielt nachlesen',inspect_fit:'Bauteil-Fit prüfen',list_tool_memory:'Werkzeuggedächtnis durchsuchen',recall_tool:'Werkzeug aus Gedächtnis laden',solidify_mesh:'Volumenkörper erzeugen',build_bim:'BIM-Körper vorbereiten',probe_thickness:'Rückseiten messen',model_terrain:'Gelände erzeugen',configure_ifc:'IFC-Struktur einrichten',set_ifc_metadata:'IFC-Daten zuordnen',join_elements:'Körper verschneiden',create_opening:'Ausschnitt erzeugen',list_methods:'Verfahrensbibliothek',evaluate_method:'Verfahren bewerten',apply_method:'Bibliotheksversuch',check_ifc:'IFC-Koordination prüfen',agent_action:'Nächste Aktion',reset:'Run zurücksetzen',define_tool:'Werkzeug entwickeln',call_custom_tool:'Eigenes Werkzeug',read_custom_tool:'Werkzeug lesen',record_decision:'Entscheidung',annotate_material:'Materialhypothese',review_model:'Gesamtmodell prüfen',request_view:'Kameraansicht',detect_planes:'Ebenen messen',model_plane:'Fläche modellieren',find_openings:'Öffnungen prüfen',validate:'Geometrie prüfen',export_ifc:'IFC exportieren',agent_start:'Start',agent_end:'Abschluss',run_experiment:'Eigener Versuch',remember:'Befund speichern',create_mesh:'Netz übernehmen',create_element:'Bauteil erzeugen',update_element:'Bauteil ändern',delete_element:'Bauteil löschen',set_region:'Region auswählen',import:'Punktwolke laden',measure:'Messung',pca:'PCA',estimate_normals:'Normalenschätzung',finish:'Befund abschließen'};
function eventDescription(e){const r=e.result||{};if(e.tool==='reset')return 'Ergebnisse gelöscht · Punktquelle erhalten';if(e.status==='error')return r.error||'Versuch fehlgeschlagen';if(e.tool==='agent_action')return e.args.rationale;if(e.tool==='agent_start')return e.args.mode==='demo'?'Deterministischer Referenzdurchlauf gestartet':'Visuelle Untersuchung mit freier Werkzeugwahl gestartet';if(e.tool==='agent_end')return r.summary;if(e.tool==='import')return `${fmt(r.points)} Punkte · Quellhash gespeichert`;if(e.tool==='request_view')return `${r.label} · ${r.projection} · ${fmt(r.point_count)} Punkte`;if(e.tool==='detect_planes')return `${r.planes?.length||0} Ebenenhypothesen aus 3D-Punkten`;if(e.tool==='model_plane')return `${r.name} · ${fmt(r.width,2)} × ${fmt(r.height,2)} m · Dicke angenommen`;if(e.tool==='find_openings')return `${r.openings?.length||0} Öffnungshypothesen · Semantik unbestätigt`;if(e.tool==='validate')return `${fmt(r.rms_m*1000,1)} mm RMS · ${fmt(r.within_tolerance*100,1)} % innerhalb der Toleranz`;if(e.tool==='record_decision')return r.decision+' · '+r.reason;if(e.tool==='probe_thickness')return r.candidates.length+' Dickenkandidaten aus der Punktwolke';if(e.tool==='solidify_mesh')return r.name+' · '+fmt(r.net_volume_m3,2)+' m³ · '+r.representation;if(e.tool==='build_bim')return r.built.length+' Körper erzeugt · '+r.still_open.length+' Messflächen offen';if(e.tool==='evaluate_method')return (r.passed?'Prüfung bestanden':'Kandidat zurückgestellt')+' · P95 '+fmt(r.candidate?.p95_m*1000,2)+' mm';if(e.tool==='check_ifc')return (r.clashes?.length||0)+' Überschneidungen · '+(r.issues?.length||0)+' IFC-Befunde';if(e.tool==='review_model')return r.accepted?'Prüfkriterien erfüllt':(r.issues||[]).concat(r.unresolved||[]).join(' · ');if(e.tool==='run_experiment'||e.tool==='call_custom_tool'||e.tool==='apply_method')return `${r.name} · ${r.ok?'Ergebnis verfügbar':r.error} · ${r.duration_s}s`;return r.summary||r.name||r.note||'Aktion mit Eingaben und Ergebnis protokolliert';}
function renderDock(){const content=$('dock-content');if(!state)return;if(activeTab==='agent'){const rows=events.slice().reverse();content.innerHTML=rows.map(e=>`<div class="activity-row ${e.status==='error'?'error':''}" data-seq="${e.seq}" tabindex="0" role="button" title="Gespeicherten Zustand anzeigen"><time>${new Date(e.time*1000).toLocaleTimeString('de-DE',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}</time><span class="step-icon">${e.status==='error'?'!':'✓'}</span><strong>${esc(toolLabels[e.tool]||e.tool)}</strong><p>${esc(eventDescription(e))}</p><span class="duration">${e.duration_ms?e.duration_ms+' ms':''}</span></div>`).join('')||'<div class="empty-note">Noch keine Aktionen.</div>';content.querySelectorAll('[data-seq]').forEach(row=>{const fn=()=>showReplay(Number(row.dataset.seq)-1);row.onclick=fn;row.onkeydown=e=>{if(e.key==='Enter')fn();};});}
if(activeTab==='experiments'){content.innerHTML='<div class="custom-tools">'+(state.custom_tools||[]).map(t=>`<details><summary>⌘ ${esc(t.name)} · Revision ${t.revision}</summary><p>${esc(t.description)}</p><p>${esc(t.parameter_help)}</p><pre>${esc(t.code)}</pre></details>`).join('')+'</div>'+ '<div class="experiment-actions"><button id="new-experiment" class="button secondary">＋ Neuer Versuch</button><span class="empty-note">Eigene Verfahren · isolierte Ausführung · reproduzierbare Ergebnisse</span></div>'+state.experiments.slice().reverse().map(e=>`<div class="experiment-row" data-exp="${e.id}"><span class="exp-icon">⌬</span><div><strong>${esc(e.name)}</strong><p>${esc(e.ok?'Erfolgreich':e.error||'Fehlgeschlagen')} · ${fmt(e.point_count)} Punkte · ${e.duration_s} s</p></div><span>↗</span></div>`).join('');$('new-experiment').onclick=()=>openExperiment();content.querySelectorAll('[data-exp]').forEach(row=>row.onclick=()=>openExperiment(state.experiments.find(e=>e.id===row.dataset.exp)));}
if(activeTab==='memory'){content.innerHTML=(state.decisions||[]).slice().reverse().map(d=>`<div class="memory-entry"><b>ENTSCHEIDUNG · ${fmt(d.confidence*100)} %</b><strong>${esc(d.decision)}</strong><p>${esc(d.reason)}</p><small>Nachweise: ${esc(d.evidence.join(' · '))}${d.supersedes?' · revidiert '+esc(d.supersedes):''}</small></div>`).join('')+(state.review?`<div class="memory-entry"><b>GESAMTMODELLPRÜFUNG</b>${esc(state.review.assessment)}<p>${esc([...state.review.issues,...state.review.unresolved].join(' · ')||'Gewählte Prüfkriterien erfüllt.')}</p></div>`:'')+[...state.observations.map(x=>({kind:'BEFUND',...x})),...state.questions.map(x=>({kind:'OFFENE FRAGE',...x})),...state.uncertainties.map(x=>({kind:'UNSICHERHEIT',...x}))].map(x=>`<div class="memory-entry"><b>${x.kind}</b>${esc(x.text)}</div>`).join('')||'<p class="empty-note">Befunde, offene Fragen und Unsicherheiten des Agenten erscheinen hier.</p>';}
if(activeTab==='ifc'){renderIfcDock(content);}
if(activeTab==='views'){content.innerHTML='<div class="views-grid">'+state.views.map(v=>`<button class="view-card" data-view="${v.id}"><img src="${v.url}" alt="${esc(v.label)}"><span>${esc(v.label)}</span></button>`).join('')+'</div>';if(!state.views.length)content.innerHTML='<p class="empty-note">Mit „Ansicht“ eine eigene Kameraposition aufnehmen oder den Agenten Ansichten anfordern lassen.</p>';content.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{applyCamera(state.views.find(v=>v.id===b.dataset.view));toast('Kamera und Projektion aus den Ansichtsmetadaten übernommen.');});}}
function renderIfcDock(content){
    const current=ifcKey===ifcViewKey()?ifcView:null;
    const all=replay?.elements??state.elements;
    const parts=bimMode?all.filter(part=>current?.elements[part.id]):all;
    const report=coordination();
    const issues=[...new Set([...(current?.issues||[]),...(report?.issues||[])])];
    content.innerHTML='<div class="experiment-actions"><button class="button secondary" id="ifc-check" '+(replay||!parts.length?'disabled':'')+'>IFC prüfen</button><span class="empty-note">'+
        esc(report?report.clashes.length+' Überschneidungen · '+issues.length+' Befunde':'Schnittvolumina und Bauteilbeziehungen prüfen')+'</span></div>'+
        issues.map(issue=>'<p class="ifc-issue">'+esc(issue)+'</p>').join('')+
        (report?.clashes||[]).map((clash,index)=>'<button class="ifc-row ifc-issue" data-clash="'+index+'">'+
            esc(clash.names.join(' / '))+' · '+fmt(clash.volume_m3*1000,3)+' l Überschneidung</button>').join('')+
        (bimMode&&current?.tree?renderAssetTree(current.tree):'')+
        parts.map(e=>{
            const shape=current?.elements[e.id];
            return '<button class="ifc-row" data-part="'+esc(e.id)+'"><strong>'+esc(e.name)+'</strong><span>'+
                esc(shape?shape.class+' · '+(shape.closed?'Volumen':'Fläche'):'IFC-Körper wird geladen')+'</span></button>';
        }).join('');
    $('ifc-check').onclick=safely(()=>action('check_ifc'));
    content.querySelectorAll('[data-part]').forEach(button=>button.onclick=()=>{
        selected=button.dataset.part;
        drawScan();
        focusBounds(elementBounds(parts.find(e=>e.id===selected)));
        drawModel();
        renderInspector();
    });
    content.querySelectorAll('[data-clash]').forEach(button=>button.onclick=()=>{
        const hit=report.clashes[Number(button.dataset.clash)];
        focusBounds({min:hit.point.map(x=>x-.3),max:hit.point.map(x=>x+.3)});
    });
}
function renderAssetTree(node){
    if(!node.children.length&&node.element_id)return '';
    const name=esc(node.name||node.class);
    const type=esc([node.class,node.type].filter(Boolean).join(' · '));
    return '<details open class="asset-node"><summary>'+name+' <small>'+type+'</small></summary>'+
        node.children.map(child=>child.element_id?
            '<div class="asset-child">'+esc(child.name)+' <small>'+esc(child.class)+'</small></div>':renderAssetTree(child)).join('')+'</details>';
}
function updateBimControls(){
    const current=ifcKey===ifcViewKey()?ifcView:null;
    $('export-button').disabled=!!replay||!current||!Object.keys(current.elements).length;
    $('build-bim').disabled=!!replay||!state||['running','paused'].includes(state.agent.status);
    if(!bimMode)return;
    if(!current){$('bim-status').textContent='BIM-Körper werden geladen …';return;}
    const bodies=Object.values(current.elements).filter(element=>element.closed).length;
    const draft=!replay&&state.completion?.bim&&state.completion.status==='complete'?'Geprüfter Abschluss':replay?'Verlaufsstand':'Arbeitsstand';
    $('bim-status').textContent=draft+' · '+bodies+' Volumenkörper · '+current.excluded_surfaces+' offene Bauteilmessflächen außerhalb des BIM';
}
function setViewerMode(mode){
    bimMode=mode==='bim';
    overlay=mode==='overlay';
    const workspace=document.querySelector('.viewer-workspace');
    workspace.classList.toggle('overlay-mode',mode!=='split');
    workspace.classList.toggle('bim-mode',bimMode);
    for(const [name,id] of [['split','split-button'],['overlay','overlay-button'],['bim','bim-button']]){
        $(id).classList.toggle('active',name===mode);
        $(id).setAttribute('aria-pressed',String(name===mode));
    }
    $('bim-controls').classList.toggle('hidden',!bimMode);
    $('model-view-title').textContent=bimMode?'BIM-VOLUMENKÖRPER':'IFC-ARBEITSMODELL';
    $('export-button').textContent=bimMode?'↓ BIM-IFC exportieren':'↓ IFC exportieren';
    if(bimMode)setTab('ifc');
    render();
    setTimeout(()=>{scan.resize();model.resize();},0);
}
function showReplay(index){
    const e=events[index];
    if(!e?.snapshot)return;
    replay=e.snapshot;
    replaySeq=e.seq;
    $('replay-range').value=index;
    $('replay-label').textContent='Aktion '+(index+1)+' / '+events.length;
    render();
    if(replay.view&&$('follow-camera').checked)applyCamera(replay.view);
    $('process-stage').textContent='VERLAUF';
    $('process-tool').textContent=toolLabels[e.tool]||e.tool;
    $('process-message').textContent=eventDescription(e);
}
function renderInspector(){const e=(replay?.elements??state?.elements??[]).find(e=>e.id===selected);document.querySelector('.workspace').classList.toggle('has-selection',!!e);if(!e){$('inspector-content').innerHTML='<div class="inspector-empty"><span>⌖</span><h3>Vom Bauteil zur Messung.</h3><p>Ein Element im Modell auswählen, um Parameter, Herkunft und Unsicherheiten einzusehen.</p></div>';return;}const shape=ifcKey===ifcViewKey()?ifcView?.elements[e.id]:null;const quality=replay?null:state.quality?.elements.find(q=>q.id===e.id);let fields=e.kind==='mesh'?`<p>${e.vertices.length} Punkte · ${e.faces.length} Dreiecke</p><pre>${esc(JSON.stringify(e.parameters||{},null,2))}</pre>`:['width','height','thickness'].map(k=>`<label>${{width:'Länge',height:'Höhe / Tiefe',thickness:'Dicke'}[k]} · m <span class="field-status ${e.parameter_status?.[k]==='assumed'?'assumed':''}">${{measured:'gemessen',assumed:'angenommen',manual:'manuell'}[e.parameter_status?.[k]]||'Experiment'}</span><input data-param="${k}" type="number" min="0.001" step="0.001" value="${Number(e[k]).toFixed(4)}" ${replay?'disabled':''}></label>`).join('');$('inspector-content').innerHTML=`<span class="element-tag">${esc(shape?.class||e.kind.toUpperCase())} ${e.locked?'· MANUELL GESPERRT':''}</span><h2>${esc(e.name)}</h2><span class="element-id">ID ${e.id}</span>${shape?`<div class="evidence-card"><h4>IFC-Objekt</h4><p>${esc(shape.global_id)}</p><p>${shape.closed?'Volumenkörper · '+fmt(shape.net_volume_m3,3)+' m³':'Oberfläche'}</p><p>${esc([shape.predefined_type,shape.type_name,shape.material].filter(Boolean).join(' · '))}</p><p>${e.cuts?.length||0} Verschnitte · ${esc({measured:'gemessen',derived:'abgeleitet',assumed:'angenommen'}[e.parameter_status?.geometry]||'gemessene Bezugsfläche')}</p>${e.solidification?`<p>${esc(shape.representation)} · Tiefe ${fmt(e.solidification.thickness_min_m,2)}–${fmt(e.solidification.thickness_max_m,2)} m · ${e.solidification.thickness_status==='assumed'?'angenommen':'aus Messflächen abgeleitet'}</p>${e.solidification.inferred_vertex_fraction?`<p>${fmt(e.solidification.inferred_vertex_fraction*100)} % der Rasterpunkte mit ergänzter Gegenfläche</p>`:''}`:''}</div>`:''}<label>Bezeichnung<input id="element-name" value="${esc(e.name)}" ${replay?'disabled':''}></label>${fields}<button id="save-element" class="button secondary" ${replay?'disabled':''}>Parameter speichern & sperren</button>${e.locked?'<button id="unlock-element" class="text-button">Für Agentenänderungen freigeben</button>':''}<div class="evidence-card"><h4>Messgrundlage</h4><p>${esc(e.evidence?.method||'Eigenes Experiment')}</p><p>${esc(e.evidence?.plane_id||e.evidence?.experiment_id||'')}</p><p>${e.evidence?.support?fmt(e.evidence.support)+' stützende Punkte':''}</p><p>Quelle ${esc(state.source_hash.slice(0,18))}…</p>${shape?'<button id="ifc-evidence-points" class="text-button">'+(scanEvidence?.key===ifcViewKey()&&scanEvidence.element===e.id?'Scanpunkte ausblenden':'Stützende Scanpunkte zeigen')+'</button>':''}${quality?`<p>Flächenstützung ${fmt(quality.surface_coverage*100,1)} %</p><p>RMS ${quality.rms_m===null?'—':fmt(quality.rms_m*1000,1)} mm</p>`:''}</div>${e.material?`<div class="evidence-card"><h4>Materialhypothese · ${esc(e.material.label)}</h4><p>${fmt(e.material.confidence*100)} % Konfidenz · RGB-Versuch ${esc(e.material.experiment_id)}</p><p>${esc(e.material.uncertainty)}</p></div>`:''}<div class="uncertainty">${esc(e.uncertainty||'Unsicherheit noch nicht bewertet.')}</div>${e.openings?.map(o=>`<div class="evidence-card"><h4>Öffnung · ${esc(o.semantic_hint)} · ${esc({measured:'gemessen',derived:'abgeleitet',hypothesis:'Hypothese'}[o.status]||'Hypothese')}</h4>${o.vertices?'Freier Ausschnitt':`<p>${fmt(o.width,2)} × ${fmt(o.height,2)} m · Raster ${fmt(o.resolution_m*1000)} mm</p>`}<p>${esc(o.uncertainty||o.status||'Hypothese')}</p></div>`).join('')||''}<button id="remove-element" class="text-button danger" ${replay?'disabled':''}>Bauteil aus dem Modell entfernen</button>`;if($('ifc-evidence-points'))$('ifc-evidence-points').onclick=safely(toggleScanEvidence);$('save-element').onclick=safely(async()=>{const changes={};if($('element-name').value!==e.name)changes.name=$('element-name').value;document.querySelectorAll('[data-param]').forEach(i=>{const key=i.dataset.param,value=Number(i.value);if(Math.abs(value-e[key])>.00005)changes[key]=value;});if(!Object.keys(changes).length){toast('Keine Parameteränderung vorhanden.');return;}await action('update_element',{element_id:e.id,changes});toast('Parameter gespeichert. Der Agent respektiert die manuelle Sperre.');});if($('unlock-element')&&!replay)$('unlock-element').onclick=safely(()=>action('update_element',{element_id:e.id,changes:{locked:false}}));$('remove-element').onclick=safely(async()=>{await action('delete_element',{element_id:e.id});selected=null;render();});}

const defaultCode=`# Eigene Untersuchung auf einer numerischen Arbeitsstichprobe.\n# ALLE Punkte: for block, colors, ids in iter_point_blocks(): ...\n# Kein bestimmtes Verfahren ist vorgeschrieben.\ncenter = points.mean(axis=0)\nvalues, axes = np.linalg.eigh(np.cov((points-center).T))\nresult = {\n    "point_count": len(points),\n    "center_m": center,\n    "eigenvalues": values,\n    "extent_m": np.ptp(points, axis=0),\n    "overlays": [{"kind": "points", "vertices": [center],\n                  "label": "Schwerpunkt der Versuchsregion"}]\n}\n`;
async function openExperiment(exp){if(!state){toast('Zuerst eine Punktwolke laden.',true);return;}lastExperiment=exp||null;const replayIndex=exp?events.findIndex(item=>item.result?.id===exp.id):-1;$('replay-experiment').classList.toggle('hidden',replayIndex<0);$('replay-experiment').onclick=()=>{$('experiment-dialog').close();showReplay(replayIndex);};$('experiment-name').value=exp?.name||'Geometrische Untersuchung';$('experiment-code').value=exp?.code||localStorage.getItem('scanbim-experiment-code')||defaultCode;$('experiment-output').textContent=exp?JSON.stringify(exp,null,2):'Noch kein Versuch ausgeführt.';$('adopt-mesh').classList.toggle('hidden',!(exp?.ok&&exp.result?.vertices&&exp.result?.faces));$('experiment-region').textContent=state.region?'Ausgewählte Untersuchungsregion':'Gesamte Punktwolke';$('sandbox-status').textContent=config.sandbox?.message||'Docker-Status unbekannt';$('experiment-dialog').showModal();}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));document.querySelectorAll('.close-dialog').forEach(b=>b.onclick=()=>b.closest('dialog').close());
$('demo-button').onclick=$('empty-demo').onclick=safely(async()=>{const s=await api('/api/demo',{method:'POST'});await updateScenes();await loadScene(s.id);toast('Referenzraum geladen. Der Demo-Prüflauf kann gestartet werden.');});
function showImport(job){activeImport=job.id;$('import-progress').classList.remove('hidden');$('import-progress-name').textContent=job.name||'Punktwolke importieren';const percent=Math.round((job.progress||0)*100);$('import-progress-fill').style.width=percent+'%';$('import-progress-percent').textContent=percent+' %';let detail=job.phase||'Vorbereiten';if(job.points_read)detail+=' · '+fmt(job.points_read)+' / '+fmt(job.point_count)+' Punkte';else if(job.bytes_read)detail+=' · '+fileSize(job.bytes_read)+' / '+fileSize(job.source_bytes);$('import-progress-phase').textContent=detail;}
async function pollImport(jid){if(activeImport!==jid)return;try{const job=await api('/api/imports/'+jid);showImport(job);if(job.status==='completed'){activeImport=null;$('import-progress').classList.add('hidden');await updateScenes();await loadScene(job.scene_id);toast(fmt(state.point_count)+' Punkte vollständig importiert · '+fmt(pointData.displayed)+' in der Vorschau.');return;}if(['error','cancelled'].includes(job.status)){activeImport=null;$('import-progress').classList.add('hidden');toast(job.error||job.phase,job.status==='error');return;}importPollTimer=setTimeout(()=>pollImport(jid),650);}catch(e){activeImport=null;$('import-progress').classList.add('hidden');toast(e.message,true);}}
$('upload-button').onclick=()=>{if(activeImport){toast('Ein Import läuft bereits. Er kann unten abgebrochen werden.');return;}$('import-unit-label').textContent=$('input-unit').selectedOptions[0].textContent;$('import-dialog').showModal();};
$('browse-local-cloud').onclick=safely(async()=>{const button=$('browse-local-cloud');button.disabled=true;try{const selected=await api('/api/imports/picker',{method:'POST'});if(selected.path)$('local-cloud-path').value=selected.path;}finally{button.disabled=false;}});
$('start-local-import').onclick=safely(async()=>{const path=$('local-cloud-path').value.trim();if(!path)throw Error('Bitte eine lokale Punktdatei auswählen oder ihren Pfad eintragen.');const job=await api('/api/imports/local',{method:'POST',body:JSON.stringify({path,unit:$('input-unit').value})});$('import-dialog').close();showImport(job);pollImport(job.id);});
$('browser-upload-button').onclick=()=>$('file-input').click();
$('file-input').onchange=safely(async()=>{const file=$('file-input').files[0];if(!file)return;if(activeImport)throw Error('Ein Import läuft bereits.');const job=await api('/api/imports',{method:'POST',body:JSON.stringify({name:file.name,size:file.size,unit:$('input-unit').value})});$('import-dialog').close();showImport(job);const request=new XMLHttpRequest();uploadRequest=request;request.open('PUT','/api/imports/'+job.id+'/source');request.setRequestHeader('X-ScanBIM-Client','studio');request.setRequestHeader('Content-Type','application/octet-stream');request.upload.onprogress=e=>{if(!e.lengthComputable)return;$('import-progress-fill').style.width=(100*e.loaded/e.total)+'%';$('import-progress-percent').textContent=Math.round(100*e.loaded/e.total)+' %';$('import-progress-phase').textContent='Übertragen · '+fileSize(e.loaded)+' / '+fileSize(e.total);};request.onload=()=>{uploadRequest=null;if(request.status>=200&&request.status<300)pollImport(job.id);else{activeImport=null;$('import-progress').classList.add('hidden');try{toast(JSON.parse(request.responseText).detail,true);}catch{toast('Übertragung fehlgeschlagen.',true);}}};request.onerror=()=>{uploadRequest=null;activeImport=null;$('import-progress').classList.add('hidden');toast('Übertragung unterbrochen. Für große Dateien „Lokal öffnen“ verwenden.',true);};request.send(file);$('file-input').value='';});
$('cancel-import').onclick=safely(async()=>{if(!activeImport)return;const jid=activeImport;await api('/api/imports/'+jid+'/cancel',{method:'POST'});if(uploadRequest){uploadRequest.abort();uploadRequest=null;}pollImport(jid);});

$('scene-select').onchange=safely(()=>{if($('scene-select').value)return loadScene($('scene-select').value);});
$('start-button').onclick=safely(async()=>{if(!state)throw Error('Zuerst den Referenzraum oder eine Punktwolke laden.');const mode=$('agent-mode').value;if(mode==='vlm'){await refreshConfiguration();if(!config.vlm_configured){await openSettings();return;}}if(replay){replay=null;render();}$('follow-camera').checked=true;renderFollow();await api(`/api/scenes/${state.id}/agent/start`,{method:'POST',body:JSON.stringify({mode,max_steps:mode==='demo'?35:100,bim:mode==='vlm'})});await refresh();});
$('pause-button').onclick=safely(async()=>{const command=state.agent.status==='paused'?'resume':'pause';await api(`/api/scenes/${state.id}/agent/${command}`,{method:'POST'});await refresh();});
function renderGuidance(){
    const direct=state?.user_guidance||[];
    const hints=direct.length?direct:(state?.reconstruction_plan?.items||[]).filter(i=>/^user[_-]/.test(i.id)).map(i=>({text:i.next_action||i.label}));
    $('agent-guide-history').innerHTML=hints.slice(-3).map(h=>`<p class="guide-note">${esc(h.text)}</p>`).join('');
    $('agent-guide-send').disabled=!state;
}
$('agent-guide-form').onsubmit=async event=>{
    event.preventDefault();
    if(!state){toast('Zuerst eine Szene laden.',true);return;}
    const text=$('agent-guide-text').value.trim();if(!text)return;
    const sid=state.id;const button=$('agent-guide-send');button.disabled=true;
    const status=$('agent-guide-status');status.textContent='Hinweis wird übergeben …';
    try{
        const response=await fetch(`/api/scenes/${sid}/agent/guide`,{method:'POST',
            headers:{'Content-Type':'application/json','X-ScanBIM-Client':'studio'},body:JSON.stringify({text})});
        const result=await response.json();
        if(!response.ok){
            // A running older Python process can already accept plan updates and observations.
            // This compatibility path requires only reloading the page, not restarting its agent.
            if(response.status!==404 && !(response.status===400 && String(result.detail).includes('Unbekannter Steuerbefehl')))
                throw Error(result.detail||'Hinweis konnte nicht gespeichert werden.');
            const item={id:'user-guide-'+Date.now(),label:'Nutzerhinweis: '+text.slice(0,90),
                status:'improving',next_action:text.slice(0,1000),evidence:[text]};
            await api(`/api/scenes/${sid}/action`,{method:'POST',body:JSON.stringify({tool:'update_reconstruction_plan',arguments:{items:[item]}})});
            await api(`/api/scenes/${sid}/action`,{method:'POST',body:JSON.stringify({tool:'remember',arguments:{observation:'NUTZERHINWEIS: '+text}})});
            await api(`/api/scenes/${sid}/action`,{method:'POST',body:JSON.stringify({tool:'record_decision',arguments:{decision:('Aktueller Nutzerauftrag: '+text).slice(0,2000),reason:text,evidence:['Expliziter Nutzerhinweis über Agenten anleiten; geometrische Angaben müssen anhand der Punktwolke geprüft werden.'],confidence:1}})});
        }
        $('agent-guide-text').value='';
        status.textContent='Gespeichert. Der Agent berücksichtigt den Hinweis bei der nächsten Modellentscheidung; einen pausierten Lauf mit Fortsetzen weiterführen.';
        await refresh();
    }catch(error){status.textContent=error.message;toast(error.message,true);}
    finally{button.disabled=false;}
};
$('stop-button').onclick=safely(async()=>{await api(`/api/scenes/${state.id}/agent/stop`,{method:'POST'});toast('Stop angefordert. Erkenntnisse und Zwischenstand bleiben erhalten.');});
$('export-button').onclick=safely(async()=>{const r=await action('export_ifc',{expected_model_revision:state.model_revision,bim_only:bimMode});window.location.assign(r.url);toast((state.completion?.status==='complete'?'Geprüfter IFC-Stand':'Aktueller IFC-Arbeitsstand')+' exportiert.');});
$('model-download').onclick=safely(async()=>{if(!state)throw Error('Keine Szene geladen.');window.location.assign(`/api/scenes/${state.id}/model`);});
$('journal-button').onclick=safely(async()=>{if(!state)throw Error('Keine Szene geladen.');window.location.assign(`/api/scenes/${state.id}/journal`);});
$('timelapse-button').onclick=safely(async()=>{
    if(!state)throw Error('Zuerst eine Szene öffnen.');
    const base=`/api/scenes/${state.id}/recordings`;
    const runs=await api(base);
    const choices=$('timelapse-run');
    choices.replaceChildren(...runs.filter(run=>run.frames).map(run=>new Option(
        `${new Date(run.started_at*1000).toLocaleString()} · ${run.frames} Bilder`,run.id)));
    $('timelapse-export').disabled=!choices.options.length;
    $('timelapse-message').textContent=choices.options.length
        ? 'Wähle einen Lauf und speichere den Zeitraffer als GIF.'
        : 'Für diese Szene gibt es noch keine Aufnahmen. Neue Agentenläufe werden automatisch aufgezeichnet.';
    $('timelapse-dialog').showModal();
    $('timelapse-export').onclick=safely(async()=>{
        const button=$('timelapse-export');
        button.disabled=true;
        $('timelapse-message').textContent='GIF wird erstellt …';
        try {
            const {url}=await api(`${base}/${encodeURIComponent(choices.value)}/gif`,{method:'POST'});
            $('timelapse-message').textContent='GIF gespeichert. Download gestartet.';
            window.location.assign(url);
        } catch(error) {
            $('timelapse-message').textContent=error.message;
        } finally {
            button.disabled=false;
        }
    });
});
$('help-button').onclick=()=>$('help-dialog').showModal();
$('split-button').onclick=()=>setViewerMode('split');
$('overlay-button').onclick=()=>setViewerMode('overlay');
$('bim-button').onclick=()=>setViewerMode('bim');
$('build-bim').onclick=safely(async()=>{
    if(!state)throw Error('Zuerst eine Punktwolke laden.');
    await refreshConfiguration();
    if(!config.vlm_configured&&!state.elements.length){await openSettings();return;}
    await api('/api/scenes/'+state.id+'/agent/start',{method:'POST',body:JSON.stringify({mode:'vlm',max_steps:140,bim:true})});
    await refresh();
});
document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>{$('follow-camera').checked=false;renderFollow();cameraPreset(b.dataset.camera);});$('fit-button').onclick=()=>{$('follow-camera').checked=false;renderFollow();cameraPreset();};$('ortho-button').onclick=()=>{const ortho=!scan.camera.isOrthographicCamera;for(const v of[scan,model])v.setProjection(ortho);$('ortho-button').classList.toggle('active',ortho);};
$('hide-ceiling').onchange=()=>{drawScan();drawModel();};$('heatmap').onchange=safely(async()=>{if(!state)throw Error('Zuerst Punktdaten laden.');if($('heatmap').checked&&!state.quality)await action('validate');render();});$('slice-range').oninput=()=>{requestedSection=null;clipping();};
$('capture-button').onclick=safely(async()=>{const cam=scan.camera;let args={position:cam.position.clone().add(worldOffset()).toArray(),target:scan.controls.target.clone().add(worldOffset()).toArray(),up:cam.up.toArray(),projection:cam.isOrthographicCamera?'orthographic':'perspective',span:scan.orthoSpan||10,label:'Manuelle Untersuchung'};const value=Number($('slice-range').value);if(state&&value<100){const z=state.bounds.min[2]+(state.bounds.max[2]-state.bounds.min[2])*value/100;args.section={origin:[0,0,z],normal:[0,0,1],thickness:.12};args.label='Horizontaler Schnitt';}await action('request_view',args);setTab('views');toast('Ansicht mit Kamera- und Koordinatenmetadaten gespeichert.');});
$('measure-button').onclick=()=>{measuring=!measuring;measureIds=[];$('measure-button').classList.toggle('active',measuring);if(measuring)toast('Zwei Punkte in der Punktwolke anklicken. Messung nutzt die Quellkoordinaten.');};
$('region-button').onclick=()=>{if(!state){toast('Zuerst eine Punktwolke laden.',true);return;}const r=state.region||state.bounds;$('region-fields').innerHTML=['x','y','z'].flatMap((axis,i)=>['min','max'].map(k=>`<label>${k==='min'?'Minimum':'Maximum'} ${axis.toUpperCase()} · m<input type="number" step="0.01" data-bound="${k}" data-axis="${i}" value="${r[k][i].toFixed(3)}"></label>`)).join('');$('region-dialog').showModal();};
$('save-region').onclick=safely(async()=>{let region={min:[],max:[]};document.querySelectorAll('[data-bound]').forEach(i=>region[i.dataset.bound][Number(i.dataset.axis)]=Number(i.value));await action('set_region',{region});$('region-dialog').close();});$('clear-region').onclick=safely(async()=>{await action('set_region',{region:null});$('region-dialog').close();});
$('show-attempts').onchange=()=>{drawScan();drawModel();};
$('replay-range').oninput=()=>showReplay(Number($('replay-range').value));$('live-button').onclick=()=>{replay=null;render();};
$('roof-example').onclick=safely(async()=>{$('experiment-code').value=(await api('/api/examples/roof')).code;$('experiment-name').value='Dach · First, Traufe und variable Neigung';});
$('run-experiment').onclick=safely(async()=>{const b=$('run-experiment');b.disabled=true;$('experiment-output').textContent='Isolierter Versuch läuft …';localStorage.setItem('scanbim-experiment-code',$('experiment-code').value);try{lastExperiment=await action('run_experiment',{name:$('experiment-name').value,code:$('experiment-code').value,timeout_s:20});$('experiment-output').textContent=JSON.stringify(lastExperiment,null,2);$('adopt-mesh').classList.toggle('hidden',!(lastExperiment.ok&&lastExperiment.result?.vertices&&lastExperiment.result?.faces));setTab('experiments');}catch(e){$('experiment-output').textContent=e.message;throw e;}finally{b.disabled=false;}});
$('adopt-mesh').onclick=safely(async()=>{if(!lastExperiment?.ok)return;const r=lastExperiment.result;await action('create_mesh',{experiment_id:lastExperiment.id,name:lastExperiment.name,semantic:'roof',vertices:r.vertices,faces:r.faces,parameters:{sections:r.sections||[],rms_m:r.rms_m},uncertainty:r.uncertainty||'Experimentelle Geometrie, Semantik angenommen.'});$('experiment-dialog').close();toast('Experimentelle Netzgeometrie mit Herkunftsnachweis übernommen.');});
$('experiment-code').addEventListener('keydown',e=>{if(e.key==='Tab'){e.preventDefault();const t=e.target,start=t.selectionStart;t.setRangeText('    ',start,t.selectionEnd,'end');}});
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&measuring){measuring=false;measureIds=[];$('measure-button').classList.remove('active');}});
safely(async()=>{await refreshConfiguration();const scenes=await updateScenes();const remembered=localStorage.getItem('scanbim-scene');const initial=scenes.find(s=>s.id===remembered)||scenes[0];if(initial)await loadScene(initial.id);})();

function settingsResult(message,kind=''){const box=$('settings-result');box.textContent=message;box.className='settings-result'+(kind?' '+kind:'');}
function settingsBusy(value){
    for(const id of['settings-save','settings-test','settings-load-models',
        'settings-provider','settings-model','settings-api-key'])$(id).disabled=value;
    document.querySelectorAll('[data-model-test]').forEach(button=>button.disabled=value);
}
async function refreshConfiguration(){config=await api('/api/config');$('agent-mode').querySelector('[value="vlm"]').textContent=config.vlm_configured?'Visueller LLM · freie Experimente':'Visueller LLM · Zugang einrichten';$('agent-settings-button').textContent=config.vlm_configured?'KI-Zugang':'KI einrichten';}
let localSettings = true;
function showSettings(status) {
    localSettings = status.provider === 'ollama';
    $('settings-provider').value = status.provider;
    $('settings-test').textContent = 'Text testen';
    const key = $('settings-api-key');
    key.value = '';
    key.hidden = localSettings;
    key.labels[0].hidden = localSettings;
    $('settings-key-status').hidden = localSettings;
    $('settings-key-status').textContent = status.has_api_key ? 'API-Schlüssel gespeichert.' : 'Noch kein API-Schlüssel gespeichert.';
    key.placeholder = status.has_api_key ? 'Leer lassen zum Beibehalten' : 'API-Schlüssel einfügen';
    $('settings-model').value = status.model || '';
    $('settings-model-list').replaceChildren();
    $('settings-title').textContent = localSettings ? 'Lokales Ollama-Modell'
        : status.provider === 'claude' ? 'Claude-Zugang' : 'OpenAI-Zugang';
    $('settings-dialog').querySelector('div.dialog-heading + p').textContent = localSettings
        ? 'Ollama läuft auf diesem Rechner. Hier kannst du ein installiertes Modell auswählen.'
        : 'API-Schlüssel und Modell werden in deiner lokalen Einstellungsdatei gespeichert.';
    $('settings-dialog').querySelector('.settings-privacy').textContent = localSettings
        ? 'Verbindung über 127.0.0.1:11434. Installierte Modell-ID eintragen oder über Verfügbare Modelle laden auswählen.'
        : 'Funktionstests senden Beispieltext und kleine Farbbilder und können API-Kosten verursachen.';
}
async function openSettings() {
    const status = await api('/api/settings');
    showSettings(status);
    settingsResult(status.error || 'Modell auswählen und bei Bedarf vor dem Speichern prüfen.', status.error ? 'error' : '');
    $('settings-dialog').showModal();
    $(localSettings || status.has_api_key ? 'settings-model' : 'settings-api-key').focus();
}
$('settings-button').onclick=safely(openSettings);$('agent-settings-button').onclick=safely(openSettings);
$('settings-dialog').addEventListener('close',()=>{$('settings-api-key').value='';});
$('settings-api-key').addEventListener('input',()=>settingsResult('Neue Eingabe noch nicht gespeichert.'));
$('settings-model').addEventListener('input',()=>settingsResult('Neue Eingabe noch nicht gespeichert.'));
const settingsDraft=()=>({api_key:$('settings-api-key').value,model:$('settings-model').value.trim(),
    provider:$('settings-provider').value});
$('settings-provider').onchange=async()=>{
    const provider=$('settings-provider').value;
    settingsBusy(true);
    try {
        showSettings(await api('/api/settings?provider='+encodeURIComponent(provider)));
        settingsResult('Anbieter ausgewählt. Mit „Lokal speichern“ für neue Runs aktivieren.');
    } catch(error) {
        settingsResult(error.message,'error');
    } finally {
        settingsBusy(false);
    }
};
$('settings-form').onsubmit=async event=>{event.preventDefault();settingsBusy(true);settingsResult('Einstellungen werden gespeichert …');try{const saved=await api('/api/settings',{method:'POST',body:JSON.stringify(settingsDraft())});showSettings(saved);await refreshConfiguration();settingsResult('Gespeichert. Die Einstellung gilt ab der nächsten Untersuchung.','success');}catch(error){settingsResult(error.message,'error');}finally{settingsBusy(false);}};
document.querySelectorAll('[data-model-test]').forEach(button=>{
    button.onclick=async()=>{
        settingsBusy(true);
        settingsResult('Modellfunktion wird getestet …');
        try {
            const result=await api('/api/settings/function-test',{
                method:'POST',body:JSON.stringify({...settingsDraft(),kind:button.dataset.modelTest})});
            settingsResult(result.message,result.ok?'success':'error');
        } catch(error) {
            settingsResult(error.message,'error');
        } finally {
            settingsBusy(false);
        }
    };
});
$('settings-load-models').onclick=async()=>{settingsBusy(true);settingsResult('Verfügbare Modelle werden geladen …');try{const result=await api('/api/settings/models',{method:'POST',body:JSON.stringify(settingsDraft())});$('settings-model-list').innerHTML=(result.models||[]).map(model=>'<option value="'+esc(model)+'"></option>').join('');settingsResult(result.message,result.ok?'success':'error');if(result.ok)$('settings-model').focus();}catch(error){settingsResult(error.message,'error');}finally{settingsBusy(false);}};

$('follow-camera').onchange=()=>{renderFollow();if($('follow-camera').checked){const view=replay?.view||state?.views.at(-1);if(view)applyCamera(view);else if(focusRegion)focusBounds(focusRegion);}else cameraMotion=null;};
$('agent-view-button').onclick=()=>{const view=replay?.view||state?.views.at(-1);if(view){$('follow-camera').checked=true;applyCamera(view);renderFollow();}};
$('reset-button').onclick=safely(async()=>{if(!state)return;const sid=state.id;const info=await api(`/api/scenes/${sid}/reset-preview`);if(state.id!==sid)return;resetDraft={sid,...info};$('reset-title').textContent='Diesen Run zurücksetzen?';$('reset-counts').textContent=`${info.counts.elements} Bauteile · ${info.counts.views} Ansichten · ${info.counts.experiments} Versuche · ${info.counts.custom_tools} eigene Werkzeuge · ${info.events} Protokolleinträge`;$('reset-result').textContent='Das Löschen kann nicht rückgängig gemacht werden.';$('reset-confirm').disabled=false;$('reset-dialog').showModal();});
$('reset-confirm').onclick=async()=>{if(!resetDraft)return;const button=$('reset-confirm');button.disabled=true;try{const r=await api(`/api/scenes/${resetDraft.sid}/reset`,{method:'POST',body:JSON.stringify({confirmed:true,expected_revision:resetDraft.revision})});$('reset-dialog').close();localStorage.removeItem('scanbim-experiment-code');lastExperiment=null;measuring=false;measureIds=[];$('measure-button').classList.remove('active');$('heatmap').checked=false;$('slice-range').value=100;await loadScene(resetDraft.sid);toast(r.warning||'Run gelöscht. Die Punktwolke ist bereit für einen neuen Lauf.',!!r.warning);}catch(e){$('reset-result').textContent=e.message;}finally{button.disabled=false;}};

function renderTokenUsage(usage){
    const box=$('agent-token-usage');
    if(!box)return;
    box.hidden=!usage;
    if(!usage)return;
    box.textContent=`Tokens: ${fmt(usage.input_tokens||0)} Eingabe · ${fmt(usage.output_tokens||0)} Ausgabe`+
        (usage.cached_input_tokens?` · ${fmt(usage.cached_input_tokens)} Cache`:'');
    box.title='Vom Anbieter gemeldete Tokens der erfassten Antworten. Fehlgeschlagene Anfragen ohne Verbrauchsangabe sind nicht enthalten. Bei Claude werden gelesene Cache-Tokens separat gemeldet.';
}

function renderReconstructionPlan(plan){
 const box=$('reconstruction-plan');if(!box)return;
 box.hidden=!plan?.items?.length;if(box.hidden)return;
 const labels={unexamined:'Offen',measuring:'Messen',modeling:'Modellieren',checking:'Vergleichen',improving:'Verbessern',done:'Geprueft',blocked:'Offene Frage'};
 box.innerHTML='<h4>Rekonstruktionsplan</h4><p>'+esc(plan.overview||'')+'</p>'+plan.items.map(item=>
 '<details><summary>'+esc(labels[item.status]||item.status)+' · '+esc(item.label)+'</summary><p>'+esc(item.next_action||item.method||'')+'</p>'+
 (item.uncertainty?'<p>'+esc(item.uncertainty)+'</p>':'')+'</details>').join('');
}
