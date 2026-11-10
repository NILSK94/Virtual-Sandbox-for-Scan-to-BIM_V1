"""Explicit user steering, persisted separately from model-authored findings."""
import time
import uuid


def add_guidance(scene, text):
    if not isinstance(text,str) or not text.strip() or len(text)>4000:
        raise ValueError('Hinweis: 1 bis 4000 Zeichen eingeben.')
    with scene.lock:
        entry={'id':uuid.uuid4().hex[:12],'text':text.strip(),'time':time.time(),'actor':'human'}
        scene.state.setdefault('user_guidance',[]).append(entry)
        scene.event('user_guidance',{'text':entry['text']},{'saved':True,'id':entry['id']},actor='human')
        return {'saved':True,'id':entry['id'],'message':'Hinweis gespeichert. Er gilt ab der nächsten Modellentscheidung.'}
