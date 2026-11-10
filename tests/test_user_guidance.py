"""Live steering stays separate from agent findings and survives restart."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scanbim.app import create_app, GuideAgent
from scanbim.model import SceneStore
from scanbim.guidance import add_guidance
from scanbim.token_context import scene_context
from scanbim.reset import reset_scene

class UserGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.app=create_app(self.root)
        self.scene=self.app.state.store.create(np.array([[0.,0,0],[1,1,1],[2,2,2]]),np.full((3,3),150,dtype=np.uint8),'Bridge','hash')

    def test_endpoint_accepts_guidance_without_resuming_or_changing_model(self):
        s=self.scene;s.state['agent']['status']='paused';revision=s.state['model_revision']
        route=next(r for r in self.app.routes if getattr(r,'path',None)=='/api/scenes/{sid}/agent/guide')
        # Exercise the registered endpoint directly; no socket/event-loop setup is needed.
        result=route.endpoint(s.state['id'],GuideAgent(text='Kappen zuerst bearbeiten'))
        self.assertTrue(result['saved'])
        self.assertEqual(s.state['agent']['status'],'paused')
        self.assertEqual(s.state['model_revision'],revision)
        self.assertEqual(s.events[-1]['actor'],'human')
        self.assertIn('Kappen zuerst bearbeiten',scene_context(s,40))

    def test_guidance_survives_reload_and_is_cleared_by_new_run_reset(self):
        add_guidance(self.scene,'Geländer-Endpunkte messen')
        sid=self.scene.state['id'];loaded=SceneStore(self.root).get(sid)
        self.assertEqual(loaded.state['user_guidance'][0]['text'],'Geländer-Endpunkte messen')
        reset_scene(loaded,loaded.state['revision'],confirmed=True)
        self.assertEqual(loaded.state['user_guidance'],[])

    def test_bad_input_is_rejected_without_saving(self):
        for text in ('','  ','x'*4001):
            with self.assertRaises(ValueError):add_guidance(self.scene,text)
        self.assertFalse(self.scene.state.get('user_guidance'))

if __name__=='__main__':unittest.main()
