"""Shared tool boundary for human actions, open-ended VLM and reference demo."""
from __future__ import annotations
import copy
import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np
from scipy import ndimage
from . import geometry as g
from .ifc_export import bim_state, export_ifc, inspect_ifc
from .ifc_tools import add_opening, add_terrain, change_join, configure_model, parse_mesh, set_metadata
from .methods import apply_method, evaluate_method, list_methods
from .model import clean
from .views import render_view
from .review import review_model,model_signature
from .solids import build_available_solids, open_structural_surfaces, probe_thickness, solidify
from . import tool_memory
from .fit import inspect_fit
from . import segmentation
from .analysis_visuals import analysis_overlays
from .token_context import read_result

CATALOG={
 "build_and_check_part":"Einen gemessenen Teil direkt als IFC-Bauteil bauen UND gegen die Punktwolke pruefen. args:{name,semantic,experiment_id?:string,code?:string,selection_id?:string,region?:{min,max},element_id?:string,uncertainty?:string,tolerance?:.03,baseline_id?:string,timeout_s?:20}. Eigener Code liefert result={vertices,faces}; alternativ erfolgreichen Versuch angeben. Geschlossener Koerper erforderlich (terrain darf Oberflaeche sein). element_id revidiert dasselbe Bauteil. Liefert IFC-Export und lokale Fit-Rueckmeldung, keine automatische Qualitaetsfreigabe.",
 "update_reconstruction_plan":"Eigenen Bauwerksplan erstellen/fortschreiben. args:{overview?:string,items?:[{id:string,label:string,status:unexamined|measuring|modeling|checking|improving|done|blocked,method?:string,next_action?:string,evidence?:[string],element_ids?:[string],uncertainty?:string}]}. Stabile IDs aktualisieren einzelne Aufgaben; andere bleiben erhalten. Keine vorgegebenen Bauteilklassen.",
 "tool_help":"Argumente und Beschreibung einzelner Werkzeuge nachlesen. args:{names:[string]}. Maximal 6 Namen.",
 "read_agent_result":"Lokales Vollergebnis lesen. args:{result_id?:string,pointer?:string,offset?:0,limit?:8}. Ohne ID: Archivliste. JSON Pointer z.B. /elements/0 oder /result/code; max.100 Listeneintraege oder 4000 Stringzeichen. Messwerte ungerundet.",
 "inspect_scene":"Szenengedächtnis, Punktanzahl, Grenzen und bisherige Befunde. args: {}",
 "get_points":"LOKALE numerische Punktstichprobe mit RGB, Farbherkunft und globalen Quellindizes. args:{region?:{min:[x,y,z],max:[x,y,z]},selection_id?:string,limit?:1..10000}. Fuer kompakte Proben; grosse Berechnungen lokal mit run_experiment ausfuehren.",
 "set_region":"Region für Auswahl und spätere Analysen. args: {region:{min,max}|null}",
 "request_view":"Kamera/Schnitt als PNG. args:{position?,target?,up?,projection?:orthographic|perspective,span?:m,region?:{min,max},section?:{origin,normal,thickness},label?,image_detail?:low|high}. Standard high erhaelt feine Details; low fuer einfache Uebersichten. Kamera-Vektoren [x,y,z].",
 "pca":"LOKALE PCA und orientierte Grenzen. args:{region?:{min,max},selection_id?:string,sample_limit?:200000}",
 "estimate_normals":"LOKALE Nachbarschaftsnormalen und Kruemmung aus PCA; liefert Normalen-Overlay zur Orientierung. args:{region?:{min,max},k?:16}. Bis 3000 Normalen, Analyse auf regionaler Punktstichprobe. Fuer gekruemmte oder geneigte Flaechen und eigene Segmentierung nutzbar.",
 "detect_planes":"LOKALE RANSAC-Ebenenanalyse (ohne API/Docker): Normalen, Grenzen, RMS und gespeicherte Inlier-Punktauswahl je Ebene. args:{region?:{min,max},selection_id?:string,sample_limit?:200000,threshold?:.025,iterations?:180,max_planes?:12,min_points?:180,seed?:17}. Getrennte koplanare Bauteile vorher clustern oder Inlier danach clustern; Ebenengrenzen sind Hypothesen.",
 "cluster_points":"LOKALES DBSCAN mit SciPy (ohne API/Docker). Punkte geometrisch gruppieren; kompakte Grenzen/PCA und selection_id je Cluster fuer weitere Analyse/eigenen Code. args:{region?:{min,max},selection_id?:string,eps?:.08,min_samples?:8,min_cluster_points?:30,sample_limit?:30000}. eps in Metern an Punktabstand anpassen. Cluster sind keine semantischen Klassen; du waehlst Region/Parameter und interpretierst das Ergebnis.",
 "measure":"Exakte Punktdatenmessung zwischen Quellpunktindizes oder Winkel zweier Ebenen. args: {indices:[i,j]} oder {plane_ids:[id,id]}",
 "model_plane":"Planare Hypothese als parametrisches Bauteil übernehmen. args: {plane_id:string,thickness?:.18,name?:string,depth_direction?:-1|1}. Dicke wird als angenommen markiert.",
 "find_openings":"Optionale Analyse leerer Wandrasterfelder. args: {element_id:string,cell?:.08}. Liefert unsichere Öffnungshypothesen.",
 "create_opening":"Beliebigen IFC-Ausschnitt am vorhandenen Körper erzeugen. args: {element_id,experiment_id,status?:measured|derived|hypothesis,semantic_hint?,uncertainty?}. Experiment liefert ein geschlossenes Netz aus vertices und faces.",
 "run_experiment":"EIGENEN Python-Code ausführen. points/colors: maximal 200000 echte Punkte als Arbeitsstichprobe. source_indices: globale Quellindizes. Für ALLE Punktdaten iter_point_blocks(block_size=131072) verwenden; liefert (points,colors,source_indices) je Block in der gewählten Region. data_info enthält vollständige Punktanzahl und Stichprobengröße. Mit selection_id werden nur die gespeicherten Cluster-/Ebenenpunkte bereitgestellt (auch iter_point_blocks). NumPy/SciPy verfügbar. result als JSON-Objekt setzen; optionale overlays=[{kind:points|lines|mesh,vertices,faces?,label?}]. args:{name,code,region?:{min,max},selection_id?:string,timeout_s?:20}. Verfahren frei wählen, Fehler korrigieren.",
 "read_experiment":"Gespeicherten Versuch einschließlich Quellcode, Ergebnis oder Fehler erneut lesen, auch nach Neustart. args: {experiment_id:string,include_code?:false}",
 "define_tool":"Eigenes wiederverwendbares Python-Werkzeug speichern oder revidieren. Keine feste Analysefolge. args:{name,description,code,parameter_help?:string,tool_id?:string}. Code erhält parameters als frei gestaltetes JSON-Objekt sowie points, colors, data_info, iter_point_blocks. result setzen. Revisionen bleiben im Protokoll.",
 "call_custom_tool":"Selbst definiertes Werkzeug im isolierten Versuchsraum ausführen. args:{tool_id,parameters?:{},region?:{min,max},selection_id?:string,timeout_s?:20}. Liefert Experiment mit Messnachweis zur Geometrieübernahme.",
 "list_methods":"Projektübergreifende Verfahrensbibliothek und Status anzeigen. args: {}",
 "list_tool_memory":"Gespeicherte eigene Werkzeuge mit Versionen, Ausführungsfehlern und Messwerten über alle Szenen durchsuchen. args:{query?:string,limit?:1..30}. Ausführungserfolg beweist keine geometrische Qualität.",
 "recall_tool":"Werkzeugversion aus dem dauerhaften Gedächtnis in diese Szene laden. args:{memory_id:string,version?:string}. Rückgabe enthält Code, lokale tool_id als id und Parameterbeschreibung. Vor Verwendung Eignung für diesen Scan prüfen.",
 "evaluate_method":"Erfolgreiches Netzverfahren numerisch bewerten und als Version lernen. args: {experiment_id,baseline_experiment_id?,method_id?,name?,scope?,tolerance?:.03}. Für neue Verfahren scope angeben; für neue Versionen gegen die geprüfte Version testen. Zwei verschiedene Quellscans sind zur Freigabe erforderlich.",
 "apply_method":"Bibliotheksverfahren isoliert auf die gewählte Region anwenden. args: {method_id,version?,region?,timeout_s?}. Ohne version wird die letzte geprüfte Version verwendet.",
 "read_custom_tool":"Eigenen Werkzeugcode und Parameterbeschreibung lesen. args:{tool_id:string}.",
 "record_decision":"Kurze öffentliche Entscheidung mit Begründung und Messnachweisen speichern. args:{decision,reason,evidence:[string],confidence:0..1,supersedes?:id}. Keine internen Gedankengänge.",
 "annotate_material":"Materialhypothese zu einem Bauteil speichern. args:{element_id,label,experiment_id,confidence:0..1,uncertainty:string}. RGB-basiert nur mit echten Quellfarben; Farbe allein beweist kein Material.",
 "create_element":"Eigene planare Geometrie aus erfolgreichem Experiment übernehmen. args: {experiment_id:string,element:{name,kind:wall|slab,origin,u,v,normal,width,height,thickness,parameter_status,uncertainty}}. Rahmen muss orthonormal/rechtshändig sein; Maße müssen aus Experiment stammen.",
 "create_mesh":"Freies IFC-Netz aus einem Experiment übernehmen oder mit element_id unter gleicher ID revidieren. args: {experiment_id,name?,semantic?,geometry_role?:auto|surface|solid,geometry_status?:measured|derived|assumed,anchor_id?,element_id?,vertices?,faces?,parameters?,uncertainty?}. Ohne vertices/faces direkt aus experiment.result. Hypothesen brauchen anchor_id und Begründung.",
 "model_terrain":"Gelände aus unteren Punkten im Raster bilden. args: {region?,cell?:.25,max_slope?:.8,name?}. Bei Bedarf eigenes Verfahren mit semantic:terrain übernehmen.",
 "configure_ifc":"Projektstruktur. args: {asset?,schema?,site_name?,asset_name?,asset_type?,levels?:[{id,name,elevation}],parts?:[{id,name,type,usage?,parent?}],map?}. IFC 4.3 Brückenteile z.B. DECK, ABUTMENT, PIER. Keine unbelegten Teile erzeugen.",
 "set_ifc_metadata":"IFC-Bauteil beschreiben. args: {element_id,changes:{class?,predefined_type?,type_name?,material?,layers?:[{name,thickness_m}],level?,part?,fills?:'hostId:1',bounds?:[element_id]}}. Material, Schichten und Füllung brauchen Nachweis. null entfernt eine Angabe.",
 "solidify_mesh":"Messfläche zu geprüftem IFC-Volumenkörper überführen. args: {element_id,opposite_id?,mode?:paired|paired_extend|paired_profile|planar|offset,direction?:[x,y,z],thickness_m?,min_depth_m?,max_depth_m?,cell?,extension_m?,tolerance?,name?,reason?}. Paar aus demselben Bauteil zusammenführen. Einseitig Richtung, Dicke und Begründung angeben. Gelände bleibt Oberfläche.",
 "probe_thickness":"Scan hinter einer Messfläche nach räumlich gestützten parallelen Rückseiten durchsuchen. args: {element_id,direction?,max_depth_m?:2,bin_m?:.04}. Liefert Kandidaten für Dicke und Materialrichtung.",
 "build_bim":"Bezeichnete Gegenflächen und dokumentierte Dicken automatisch zu BIM-Körpern verarbeiten. Liefert offene Bauteile zur gezielten Nacharbeit. args: {}",
 "join_elements":"Zwei Bauteile volumetrisch verschneiden (subtract) oder einen geprüften Stoß verknüpfen (contact). args: {element_id,target_id,mode?:subtract|contact,remove?:false}. Ziel ist das durchlaufende Bauteil.",
 "check_ifc":"IFC-Arbeitsstand neu einlesen und Volumenkörper, Schema und Schnittvolumina prüfen. args: {tolerance?:.002}. Liefert Fehlerstellen zur Nacharbeit.",
 "update_element":"Parameter oder Orientierung revidieren. args: {element_id,changes:{width?,height?,thickness?,depth_direction?:-1|1,name?,origin?,u?,v?,normal?,locked?},evidence?:{experiment_id?}}. Lage, Maße und Orientierung brauchen einen erfolgreichen Versuch. u,v,normal müssen einen orthonormalen rechtshändigen Rahmen bilden.",
 "delete_element":"Bauteil löschen. args: {element_id:string}. Manuell gesperrte Elemente geschützt.",
 "remember":"Befund, offene Frage und Unsicherheit speichern. args: {observation?:string,question?:string,uncertainty?:string}",
 "validate":"LOKALER Gesamt-/Regionalvergleich. region:null prueft die ganze Szene unabhaengig von der aktiven Region. Punkt-zu-endlicher-Fläche und Modell-zu-Punkt-Stützung. args: {region?:{min,max}|null,tolerance?:.03}",
 "inspect_fit":"Bauteil lokal gegen Scanpunkte prüfen. Liefert RMS, Median, P95, Flächenstützung und Koordinaten der größten Abweichungen. args:{element_id,region?:{min,max},padding?:.25,tolerance?:.03,baseline_id?:string,ifc_geometry?:false}. ifc_geometry:true prueft die tatsaechlich exportierten IFC-Dreiecke. Für Vorher-Nachher-Vergleich baseline_id nutzen. Dabei bleiben Region, Punkte und Toleranz identisch. Keine Geometrieänderung.",
 "export_ifc":"Aktuellen IFC-Arbeitsstand als IFC-Datei erzeugen. args: {expected_model_revision?,bim_only?:false}. BIM enthält Volumenkörper und Gelände.",
 "review_model":"Gesamtmodell auf Nachweise, identische Doppelgeometrie, entartete Dreiecke und aktuelle globale Validierung prüfen. Akzeptanzkriterien selbst passend zu Daten und Aufgabe wählen und begründen. args:{assessment:string,unresolved:[string],max_rms_m:number,min_coverage:0..1}. Keine Raum-, Dach- oder Ebenenfolge vorgeschrieben.",
 "finish":"Befund beenden. args:{summary:string,status:complete|partial}. complete erfordert eine bestandene aktuelle review_model-Prüfung und erzeugt IFC. partial benennt ausdrücklich offene Arbeiten. Abschluss bestätigt keine unsichtbaren Bauteile."
}

class ToolRegistry:
    def call(self,scene,name,args=None,actor="human"):
        if name not in CATALOG: raise ValueError("Unbekanntes Werkzeug: "+name)
        if not isinstance(args or {},dict): raise ValueError("Tool-Argumente müssen ein Objekt sein.")
        args=copy.deepcopy(args or {});start=time.monotonic()
        previous=scene.activity_snapshot()
        message=previous.get("message",name) if actor=="agent" else CATALOG[name].split(". ")[0]
        scene.set_activity(status="working",tool=name,message=message,
                           region=args.get("region",scene.state["region"]),element_id=args.get("element_id"),actor=actor)
        # One atomic tool mutation at a time. Stop/pause endpoints do not take this lock.
        with scene.lock:
            try:
                result=clean(self._execute(scene,name,args,actor))
                analysis_overlays(scene,name,result)
                status="error" if isinstance(result,dict) and result.get("ok") is False else "ok"
                scene.set_activity(status="error" if status=="error" else "done",tool=name,message=message,
                    region=args.get("region",scene.state["region"]),element_id=(result or {}).get("id") if name in ("create_mesh","create_element","model_plane","model_terrain","solidify_mesh") else args.get("element_id"),actor=actor)
                scene.event(name,args,result,status,time.monotonic()-start,actor)
                return result
            except Exception as error:
                scene.set_activity(status="error",tool=name,message=str(error),actor=actor)
                scene.event(name,args,{"error":str(error)},"error",time.monotonic()-start,actor)
                raise

    def _execute(self,s,name,a,actor):
        region=a.get("region",s.state["region"])
        if name=="build_and_check_part":
            if a.get("code") and a.get("experiment_id"):raise ValueError("Code ODER Experiment-ID angeben.")
            if a.get("code"):
                exp=self.call(s,"run_experiment",{"name":a["name"],"code":a["code"],"region":region,"timeout_s":a.get("timeout_s",20),**({"selection_id":a["selection_id"]} if a.get("selection_id") else {})},actor)
                if not exp.get("ok"):return {"ok":False,"experiment":exp,"stage":"measurement"}
                eid=exp["id"]
            else:eid=a.get("experiment_id")
            if not eid:raise ValueError("Eigenen Geometriecode oder erfolgreichen Versuch angeben.")
            mesh={"experiment_id":eid,"name":a["name"],"semantic":a["semantic"],
                  "geometry_role":"surface" if a["semantic"]=="terrain" else "solid",
                  "geometry_status":"derived","uncertainty":a.get("uncertainty","")}
            if a.get("element_id"):mesh["element_id"]=a["element_id"]
            element=self.call(s,"create_mesh",mesh,actor)
            result={"element_id":element["id"],"experiment_id":eid,"quality_accepted":False}
            try:
                result["ifc"]=self.call(s,"export_ifc",{},actor)
                result["fit"]=self.call(s,"inspect_fit",{"element_id":element["id"],"region":region,
                    "tolerance":a.get("tolerance",.03),"baseline_id":a.get("baseline_id"),"ifc_geometry":True},actor)
                result["ok"]=True
                result["next_action"]="Fit und Abweichungsstellen bewerten; bei Bedarf dasselbe element_id verbessern. ok bedeutet Ausfuehrungserfolg, nicht ausreichende Rekonstruktionsqualitaet."
            except (ValueError,RuntimeError) as error:
                result.update(ok=False,error=str(error),stage="export_or_comparison",geometry_retained=True)
            return result
        if name=="update_reconstruction_plan":
            plan=copy.deepcopy(s.state.get("reconstruction_plan",{"overview":"","items":[]}))
            if "overview" in a:
                if not isinstance(a["overview"],str) or len(a["overview"])>2000:raise ValueError("Uebersicht maximal 2000 Zeichen.")
                plan["overview"]=a["overview"]
            items=a.get("items",[])
            if not isinstance(items,list) or len(items)>100:raise ValueError("Maximal 100 Aufgaben je Aktualisierung.")
            for item in items:
                if not isinstance(item,dict):raise ValueError("Planaufgabe muss ein Objekt sein.")
                if not set(item)<=set("id label status method next_action evidence element_ids uncertainty".split()):raise ValueError("Unbekanntes Planfeld.")
                for field in ("id","label","status"):
                    if not isinstance(item.get(field),str) or not item[field].strip():raise ValueError("id, label und status erforderlich.")
                if item["status"] not in {"unexamined","measuring","modeling","checking","improving","done","blocked"}:raise ValueError("Ungueltiger Aufgabenstatus.")
                for field in ("method","next_action","uncertainty"):
                    if field in item and (not isinstance(item[field],str) or len(item[field])>1000):raise ValueError("Plantext maximal 1000 Zeichen.")
                for field in ("evidence","element_ids"):
                    if field in item and (not isinstance(item[field],list) or not all(isinstance(v,str) for v in item[field])):raise ValueError("Nachweise/Element-IDs als Textliste.")
                if any(eid not in {e["id"] for e in s.state["elements"]} for eid in item.get("element_ids",[])):raise ValueError("Verknuepftes Element fehlt.")
                previous=next((p for p in plan["items"] if p["id"]==item["id"]),None)
                if previous is None:plan["items"].append(copy.deepcopy(item))
                else:previous.update(copy.deepcopy(item))
            s.state["reconstruction_plan"]=plan
            return {"saved":True,"tasks":len(plan["items"]),"updated":[i["id"] for i in items]}
        if name=="tool_help":
            names=a.get("names",[])
            if not isinstance(names,list) or not 1<=len(names)<=6 or any(n not in CATALOG for n in names):
                raise ValueError("1 bis 6 gueltige Werkzeugnamen angeben.")
            return {n:CATALOG[n] for n in names}
        if name=="read_agent_result":
            return read_result(s,a)
        if name=="inspect_scene":
            state=s.public();state.pop("overlays",None)
            if actor=="agent":
                state["fit_reports"]=[{k:v for k,v in report.items() if k!="worst_points"}
                                      for report in state["fit_reports"][-8:]]
            if actor=="agent":
                for e in state["elements"]:
                    if e["kind"]=="mesh":
                        e["mesh_size"]=[len(e.pop("vertices")),len(e.pop("faces"))]
            if state["quality"]: state["quality"]={k:v for k,v in state["quality"].items() if k not in ("sample_ids","distances_m")}
            for experiment in state["experiments"]:
                experiment.pop("code",None)
                result=experiment.get("result")
                if isinstance(result,dict):
                    result.pop("overlays",None)
                    if actor=="agent" and isinstance(result.get("vertices"),list) and isinstance(result.get("faces"),list):
                        result["mesh_size"]=[len(result.pop("vertices")),len(result.pop("faces"))]
            for tool in state["custom_tools"]:tool.pop("code",None)
            return state
        if name=="get_points":
            limit=int(a.get("limit",3000))
            if not 1<=limit<=10000: raise ValueError("Stichprobe: 1 bis 10.000 Punkte.")
            p,c,ids,total,sampled=segmentation.sample_selection(s,region,limit,a.get("selection_id"))
            return {"points":p,"colors":c,"color_source":s.public()["color_source"],"source_indices":ids,"total":total,"unit":"m","source_hash":s.state["source_hash"],"sampled":sampled}
        if name=="set_region":
            if a.get("region"):
                low,high=g.vector(a["region"]["min"]),g.vector(a["region"]["max"])
                if np.any(low>=high):raise ValueError("Ungültige Region.")
            s.state["region"]=a.get("region");return {"region":s.state["region"]}
        if name=="request_view":
            detail=a.pop("image_detail","high")
            if detail not in ("low","high"):raise ValueError("Bilddetail low oder high waehlen.")
            a.setdefault("region",region)
            view=render_view(s,**a)
            view["model_image_detail"]=detail
            return view
        if name=="cluster_points":
            return segmentation.cluster_points(s,**dict(a,region=region))
        if name=="pca":
            p,_,_,total,sampled=segmentation.sample_selection(s,region,a.get("sample_limit",200000),a.get("selection_id"));r=g.pca(p)
            r["sampling"]={"analyzed":len(p),"region_total":total,"sampled":sampled};return r
        if name=="estimate_normals":
            p,_,_,total=s.sample(region);r=g.normals(p,int(a.get("k",16)),limit=3000)
            r["sampling"]={"analyzed":len(p),"region_total":total,"sampled":len(p)<total}
            s.state["overlays"]=[{"kind":"lines","label":"Normalen","vertices":np.stack([r["points"],np.asarray(r["points"])+np.asarray(r["normals"])*.12],axis=1).reshape(-1,3).tolist()}]
            return r
        if name=="detect_planes":
            params={k:v for k,v in a.items() if k not in {"region","selection_id"}}
            sample_limit=params.pop("sample_limit",200000)
            p,_,ids,total,sampled=segmentation.sample_selection(s,region,sample_limit,a.get("selection_id"))
            if len(p)<3:raise ValueError("Mindestens drei Punkte fuer Ebenenanalyse erforderlich.")
            planes=g.detect_planes(p,**params)
            # Never silently invalidate references held by existing model elements.
            offset=len(s.state["planes"])
            for i,plane in enumerate(planes):
                plane["id"]=f"plane-{offset+i+1}"
                plane["sampling"]={"analyzed":len(p),"region_total":total,"sampled":sampled,"region":region}
                inliers=np.abs((p-np.asarray(plane["origin"]))@np.asarray(plane["normal"]))<plane["threshold_m"]
                plane["selection_id"]=segmentation.save_selection(s,ids[inliers],sampled)
            s.state["planes"].extend(planes);return {"planes":planes}
        if name=="measure":
            if "indices" in a:
                ids=np.asarray(a["indices"],int)
                if ids.shape!=(2,) or ids.min()<0 or ids.max()>=len(s.points): raise ValueError("Zwei gültige Quellpunktindizes erforderlich.")
                p=s.points[ids];delta=p[1]-p[0]
                return {"indices":ids,"coordinates":p,"distance_m":np.linalg.norm(delta),"delta_m":delta,"source_hash":s.state["source_hash"]}
            planes=[self.plane(s,i) for i in a["plane_ids"]]
            if len(planes)!=2: raise ValueError("Zwei Ebenen erforderlich.")
            return {"angle_degrees":float(np.degrees(np.arccos(np.clip(abs(np.dot(planes[0]["normal"],planes[1]["normal"])),0,1)))),"plane_ids":a["plane_ids"]}
        if name=="model_plane":
            p=self.plane(s,a["plane_id"])
            if any(e.get("evidence",{}).get("plane_id")==p["id"] for e in s.state["elements"]): raise ValueError("Ebene ist bereits modelliert.")
            e={k:copy.deepcopy(p[k]) for k in ("origin","u","v","normal","width","height")}
            e.update(id=uuid.uuid4().hex[:10],kind="wall" if p["kind"]=="wall" else "slab",
                name=a.get("name",("Wand" if p["kind"]=="wall" else "Fläche")+f" {len(s.state['elements'])+1:02}"),
                thickness=float(a.get("thickness",.18)),depth_direction=a.get("depth_direction",-1),openings=[],locked=False,
                evidence={"plane_id":p["id"],"support":p["support"],"source_hash":s.state["source_hash"],"method":"RANSAC + SVD","sampling":p.get("sampling")},
                parameter_status={"origin":"measured","width":"measured","height":"measured","thickness":"assumed"},
                uncertainty="Dicke und Materialaufbau nicht gemessen; aus einseitiger Oberfläche nicht bestimmbar.")
            self.check_element(e);s.state["elements"].append(e);s.invalidate_quality();return e
        if name=="find_openings":
            e=self.element(s,a["element_id"],actor)
            if any("vertices" in o for o in e.get("openings",[])):
                raise ValueError("Vorhandene freie Ausschnitte werden nicht durch die Rastersuche ersetzt.")
            if any(other.get("ifc",{}).get("fills","").startswith(e["id"]+":") for other in s.state["elements"]):
                raise ValueError("Vor einer neuen Öffnungssuche die vorhandenen Füllungszuordnungen lösen.")
            p,_,_,total=s.sample(region)
            e["openings"]=g.find_openings(p,e,float(a.get("cell",.08)));s.invalidate_quality()
            return {"element_id":e["id"],"openings":e["openings"],"sampling":{"analyzed":len(p),"region_total":total,"sampled":len(p)<total}}
        if name=="create_opening":
            return add_opening(s,self.element(s,a["element_id"],actor),self.experiment(s,a["experiment_id"]),a)
        if name=="run_experiment":
            code=a["code"]
            if not isinstance(code,str) or not 1<=len(code.encode())<=64000:
                raise ValueError("Programm maximal 64 KB.")
            tool={"name":str(a.get("name","Eigenes Verfahren"))[:120],
                  "description":"Freier Versuch. Geltungsbereich vor Wiederverwendung prüfen.",
                  "parameter_help":"","code":code,
                  "memory_id":hashlib.sha256(code.encode()).hexdigest()[:12]}
            extra={"selection_id":a["selection_id"]} if a.get("selection_id") else {}
            return tool_memory.run_tool(s,tool,region,int(a.get("timeout_s",20)),a.get("parameters",{}),**extra)
        if name=="list_methods":
            return list_methods(s)
        if name=="list_tool_memory":
            return tool_memory.list_tool_memory(s,a.get("query",""),a.get("limit",12))
        if name=="recall_tool":
            return tool_memory.recall_tool(s,a["memory_id"],a.get("version"))
        if name=="evaluate_method":
            candidate=self.experiment(s,a["experiment_id"])
            baseline=self.experiment(s,a["baseline_experiment_id"]) if a.get("baseline_experiment_id") else None
            return evaluate_method(s,candidate,baseline,a)
        if name=="apply_method":
            return apply_method(s,a,region)
        if name=="read_experiment":
            experiment=next((e for e in s.state["experiments"] if e["id"]==a["experiment_id"]),None)
            if experiment is None: raise ValueError("Experiment nicht gefunden.")
            return {k:v for k,v in experiment.items() if k not in {"code","image_id","limits"} or a.get("include_code",False)}
        if name=="define_tool":
            code=a["code"]
            if not isinstance(code,str) or not 1<=len(code.encode())<=64000:raise ValueError("Werkzeugcode: 1 bis 64.000 Byte.")
            tool=next((t for t in s.state["custom_tools"] if t["id"]==a.get("tool_id")),None)
            if a.get("tool_id") and tool is None:raise ValueError("Eigenes Werkzeug nicht gefunden.")
            is_new=tool is None
            if is_new:
                if len(s.state["custom_tools"])>=40:raise ValueError("Maximal 40 eigene Werkzeuge je Szene.")
                tool={"id":uuid.uuid4().hex[:10],"revision":0}
            updated={**tool,"name":str(a["name"])[:120],"description":str(a["description"])[:2000],
                     "code":code,"parameter_help":str(a.get("parameter_help",""))[:2000],
                     "revision":tool["revision"]+1}
            updated.update(tool_memory.remember_tool(s,updated))
            tool.update(updated)
            if is_new:s.state["custom_tools"].append(tool)
            return tool
        if name in ("call_custom_tool","read_custom_tool"):
            tool=next((t for t in s.state["custom_tools"] if t["id"]==a["tool_id"]),None)
            if tool is None:raise ValueError("Eigenes Werkzeug nicht gefunden.")
            if name=="read_custom_tool":return tool
            extra={"selection_id":a["selection_id"]} if a.get("selection_id") else {}
            result=tool_memory.run_tool(s,tool,region,int(a.get("timeout_s",20)),a.get("parameters",{}),**extra)
            result["custom_tool"]={"id":tool["id"],"revision":tool["revision"]}
            return result
        if name=="record_decision":
            confidence=float(a["confidence"])
            if not np.isfinite(confidence) or not 0<=confidence<=1:raise ValueError("Konfidenz: 0 bis 1.")
            if not isinstance(a["evidence"],list) or not all(isinstance(v,str) for v in a["evidence"]):raise ValueError("Nachweise als Textliste angeben.")
            if a.get("supersedes") and not any(d["id"]==a["supersedes"] for d in s.state["decisions"]):raise ValueError("Vorherige Entscheidung nicht gefunden.")
            decision={"id":uuid.uuid4().hex[:10],"decision":str(a["decision"])[:2000],"reason":str(a["reason"])[:4000],
                      "evidence":a["evidence"],"confidence":confidence,"supersedes":a.get("supersedes"),"time":time.time()}
            s.state["decisions"].append(decision);return decision
        if name=="annotate_material":
            e=self.element(s,a["element_id"],actor);exp=self.experiment(s,a["experiment_id"])
            if s.public()["color_source"]!="source_rgb":raise ValueError("Echte Quellfarben sind nicht nachgewiesen. Für Materialerkennung neu mit RGB importieren.")
            confidence=float(a["confidence"])
            if not np.isfinite(confidence) or not 0<=confidence<=1:raise ValueError("Konfidenz: 0 bis 1.")
            e["material"]={"label":str(a["label"])[:160],"status":"hypothesis","confidence":confidence,
                "experiment_id":exp["id"],"uncertainty":str(a["uncertainty"])[:2000],"basis":"source_rgb"}
            s.state["review"]=None;s.state["completion"]=None;return e["material"]
        if name in ("create_element","create_mesh"):
            exp=self.experiment(s,a["experiment_id"])
            if name=="create_mesh":
                old=self.element(s,a["element_id"],actor) if a.get("element_id") else None
                if old is not None and old["kind"]!="mesh":
                    raise ValueError("Nur ein bestehendes Netz kann als Netz revidiert werden.")
                e=copy.deepcopy(old) if old is not None else {}
                result=exp.get("result")
                if not isinstance(result,dict):result={}
                e.update(parse_mesh(a.get("vertices",result.get("vertices")),a.get("faces",result.get("faces")),
                                    a.get("geometry_role",e.get("geometry_role","auto"))))
                status=a.get("geometry_status",e.get("parameter_status",{}).get("geometry","measured"))
                anchor=a.get("anchor_id",e.get("evidence",{}).get("anchor_id"))
                uncertainty=str(a.get("uncertainty",e.get("uncertainty",""))).strip()
                if status not in ("measured","derived","assumed"):
                    raise ValueError("Geometriestatus muss measured, derived oder assumed sein.")
                if anchor and not any(v["id"]==anchor and v["id"]!=e.get("id") and
                                      v.get("parameter_status",{}).get("geometry")!="assumed" for v in s.state["elements"]):
                    raise ValueError("Anker muss ein anderes, durch Messung gestütztes Bauteil sein.")
                if status=="assumed" and not (anchor and uncertainty):
                    raise ValueError("Angenommene Geometrie braucht einen Anker und eine Begründung.")
                e.update(kind="mesh",name=str(a.get("name",e.get("name","Freies Netz")))[:120],
                         semantic=str(a.get("semantic",e.get("semantic","other")))[:120],
                         parameters=a.get("parameters",e.get("parameters",{})),uncertainty=uncertainty[:2000],
                         parameter_status={"geometry":status,"semantic":"assumed"})
                e["evidence"]={"experiment_id":exp["id"],"source_hash":s.state["source_hash"],"region":exp.get("region")}
                if anchor:e["evidence"]["anchor_id"]=anchor
                if old is not None:
                    if (e.get("cuts") or any(old["id"] in v.get("cuts",[]) for v in s.state["elements"])) and e["geometry_role"]!="solid":
                        raise ValueError("Ein verbundenes Netz muss als solid revidiert werden.")
                    old.clear();old.update(e);s.invalidate_quality()
                    return old
            else:
                e=copy.deepcopy(a["element"]);e["openings"]=[]
                self.check_element(e)
                e.setdefault("parameter_status",{"geometry":"measured","thickness":"assumed"})
                e["evidence"]={"experiment_id":exp["id"],"source_hash":s.state["source_hash"],"region":exp.get("region")}
            e.update(id=uuid.uuid4().hex[:10],locked=False)
            s.state["elements"].append(e);s.invalidate_quality();return e
        if name=="model_terrain":
            return add_terrain(s,a,region)
        if name=="solidify_mesh":
            element=self.element(s,a["element_id"],actor)
            other=self.element(s,a["opposite_id"],actor) if a.get("opposite_id") else None
            return solidify(s,element,other,a)
        if name=="build_bim":
            return build_available_solids(s,actor)
        if name=="probe_thickness":
            return probe_thickness(s,self.element(s,a["element_id"],"human"),a)
        if name=="configure_ifc":
            return configure_model(s,a)
        if name=="set_ifc_metadata":
            e=self.element(s,a["element_id"],actor)
            return set_metadata(s,e,a["changes"])
        if name=="join_elements":
            e=self.element(s,a["element_id"],actor)
            target=self.element(s,a["target_id"],"human")
            return change_join(s,e,target,a.get("remove",False),a.get("mode","subtract"))
        if name=="check_ifc":
            tolerance=float(a.get("tolerance",.002))
            if not .0001<=tolerance<=.02:raise ValueError("IFC-Prüftoleranz: 0,1 bis 20 mm.")
            _,report=inspect_ifc(s.state,check=True,tolerance=tolerance)
            s.state["coordination"]=report
            return report
        if name=="update_element":
            e=self.element(s,a["element_id"],actor);changes=a["changes"]
            allowed={"width","height","thickness","depth_direction","name","origin","u","v","normal","locked"}
            if not set(changes)<=allowed: raise ValueError("Nicht unterstützte Parameteränderung.")
            if e["kind"]=="mesh" and set(changes)-{"name","locked"}:raise ValueError("Netzgeometrie mit create_mesh revidieren.")
            if actor!="human" and "locked" in changes: raise ValueError("Nur Anwender dürfen Sperren ändern.")
            if actor!="human" and set(changes)&{"width","height","origin","u","v","normal"}:
                self.experiment(s,a.get("evidence",{}).get("experiment_id"))
            revised=copy.deepcopy(e);revised.update(changes)
            if revised["kind"]!="mesh": self.check_element(revised)
            if "thickness" in changes and e.get("ifc",{}).get("layers"):
                raise ValueError("Vor einer Dickenänderung die vorhandenen Materialschichten lösen.")
            if actor=="human":
                revised["locked"]=changes.get("locked",True)
                for key in changes:
                    if key!="locked": revised.setdefault("parameter_status",{})[key]="manual"
            else:
                revised.setdefault("evidence",{})["revision"]=a.get("evidence",{})
            e.clear();e.update(revised);s.invalidate_quality();return e
        if name=="delete_element":
            e=self.element(s,a["element_id"],actor)
            if any(v is not e and (e["id"] in v.get("cuts",[]) or e["id"] in v.get("contacts",[]) or e["id"] in v.get("ifc",{}).get("bounds",[]) or
                   v.get("ifc",{}).get("fills","").startswith(e["id"]+":") or v.get("evidence",{}).get("anchor_id")==e["id"])
                   for v in s.state["elements"]):
                raise ValueError("Zuerst Verschnitte, Raumgrenzen, Füllungszuordnungen oder abgeleitete Geometrien lösen.")
            s.state["elements"].remove(e);s.invalidate_quality();return {"deleted":e["id"]}
        if name=="remember":
            for key,dest in [("observation","observations"),("question","questions"),("uncertainty","uncertainties")]:
                if a.get(key): s.state[dest].append({"text":str(a[key])[:4000],"time":time.time(),"actor":actor})
            return {"saved":True}
        if name=="validate":
            p,_,ids,total=s.sample(region,80000)
            if not len(p):raise ValueError("Prüfregion enthält keine Punkte.")
            elements=s.state["elements"]
            evaluated_ids={e["id"] for e in elements if e.get("cuts") or any("vertices" in o for o in e.get("openings",[]))}
            if evaluated_ids:
                _,evaluated=inspect_ifc(s.state,meshes=True)
                elements=[{**e,**evaluated["elements"][e["id"]],"kind":"mesh"} if e["id"] in evaluated_ids and
                          e["id"] in evaluated["elements"] else e for e in elements]
            quality=g.validate(p,elements,float(a.get("tolerance",.03)))
            quality["sample_ids"]=ids[np.asarray(quality["sample_ids"],int)].tolist()
            quality["source_point_count"]=len(s.points);quality["region_point_count"]=total
            quality["sampled"]=len(p)<total;s.state["quality"]=quality
            quality["region"]=region;quality["model_signature"]=model_signature(s.state["elements"])
            s.state["review"]=None;s.state["completion"]=None
            return {k:v for k,v in quality.items() if k not in ("sample_ids","distances_m")}
        if name=="inspect_fit":
            return inspect_fit(s,**a)
        if name=="export_ifc":
            if not s.state["elements"]: raise ValueError("Kein Modell zum Exportieren.")
            if a.get("expected_model_revision",s.state["model_revision"])!=s.state["model_revision"]:
                raise ValueError("Der Arbeitsstand wurde gerade geändert. Export nach Aktualisierung erneut starten.")
            bim=bool(a.get("bim_only"))
            if bim and not bim_state(s.state)["elements"]:
                raise ValueError("Für den BIM-Export fehlen geprüfte Körper oder Gelände.")
            filename="bim.ifc" if bim else "reconstruction.ifc"
            r=export_ifc(s,s.folder/filename,bim_only=bim)
            revision=s.state["model_revision"]
            s.state["exported_bim_revision" if bim else "exported_ifc_revision"]=revision
            r["url"]=f"/api/scenes/{s.state['id']}/artifacts/{filename}?revision={revision}"
            r["model_revision"]=revision
            return r
        if name=="review_model":return review_model(s,**a)
        if name=="finish":
            status=a.get("status","partial")
            if status not in ("complete","partial"):raise ValueError("Abschlussstatus complete oder partial erforderlich.")
            review=s.state["review"]
            if status=="complete" and (not review or not review["accepted"] or review["model_signature"]!=model_signature(s.state["elements"])):
                raise ValueError("Zuerst das Gesamtmodell prüfen und offene Mängel beheben oder mit status partial abschließen.")
            bim=s.state["agent"].get("bim",False) or bool(a.get("bim"))
            if status=="complete" and bim and open_structural_surfaces(s.state):
                raise ValueError("Für den BIM-Abschluss offene Bauteilmessflächen zu Körpern überführen.")
            result={"finished":True,"status":status,"summary":str(a["summary"])[:4000]}
            if not result["summary"].strip():raise ValueError("Abschlussbefund fehlt.")
            if s.state["elements"]:result["ifc"]=self._execute(s,"export_ifc",{},actor)
            if bim and bim_state(s.state)["elements"]:
                result["bim"]=self._execute(s,"export_ifc",{"bim_only":True},actor)
            s.state["completion"]=result;return result
        raise ValueError("Nicht implementiert.")

    @staticmethod
    def plane(s,pid):
        return next((p for p in s.state["planes"] if p["id"]==pid),None) or (_ for _ in ()).throw(ValueError("Ebene nicht gefunden."))

    @staticmethod
    def element(s,eid,actor):
        e=next((e for e in s.state["elements"] if e["id"]==eid),None)
        if e is None: raise ValueError("Bauteil nicht gefunden.")
        if actor!="human" and e.get("locked"): raise ValueError("Bauteil ist durch manuellen Eingriff gesperrt.")
        return e

    @staticmethod
    def experiment(s,eid):
        exp=next((e for e in s.state["experiments"] if e["id"]==eid and e.get("ok")),None)
        if exp is None: raise ValueError("Erfolgreiches Experiment als Messnachweis erforderlich.")
        return exp

    @staticmethod
    def check_element(e):
        if e["kind"] not in ("wall","slab"): raise ValueError("Planarer Bauteiltyp wall oder slab erforderlich.")
        g.vector(e["origin"])
        frame=np.array([g.vector(e[k]) for k in ("u","v","normal")])
        if not np.allclose(frame@frame.T,np.eye(3),atol=.001) or np.linalg.det(frame)<.999:
            raise ValueError("Bauteilrahmen muss orthonormal und rechtshändig sein (u × v = normal).")
        for key in ("width","height","thickness"):
            value=float(e[key])
            if not np.isfinite(value) or not .001<=value<=10000: raise ValueError("Ungültiger Wert für "+key)
        if e.get("depth_direction",-1) not in (-1,1):raise ValueError("Materialseite muss -1 oder 1 sein.")
