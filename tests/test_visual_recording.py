"""Visual result and exact IFC-section recording integration tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry
from scanbim.timelapse import mesh_section, recordings, export_gif, frame_paths

class VisualRecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        x,z=np.meshgrid(np.linspace(0,4,28),np.linspace(0,3,24))
        points=np.c_[x.ravel(),np.zeros(x.size),z.ravel()]
        self.scene=SceneStore(Path(self.tmp.name)).create(points,np.full(points.shape,160,dtype=np.uint8),'Wall','hash')
        self.tools=ToolRegistry()

    def test_plane_and_pca_overlays_retain_measured_world_coordinates(self):
        result=self.tools.call(self.scene,'detect_planes',{'max_planes':1})
        self.assertEqual(len(result['planes']),1)
        overlays=self.scene.state['overlays']
        self.assertTrue(any(o['kind']=='mesh' for o in overlays))
        self.assertTrue(all(len(o['colors'])==len(o['vertices']) for o in overlays))
        self.tools.call(self.scene,'pca',{})
        self.assertEqual(len(self.scene.state['overlays']),4)
        self.assertTrue(all(o['kind']=='lines' for o in self.scene.state['overlays']))

    def test_exact_cut_interpolates_triangle_edges(self):
        segments=mesh_section([[0,0,0],[2,0,0],[0,2,0]],[[0,1,2]],[.5,0,0],[1,0,0])
        self.assertEqual(segments.shape,(1,2,3))
        np.testing.assert_allclose(segments[:,:,0],.5)
        np.testing.assert_allclose(np.sort(segments[0,:,1]),[0,1.5])
        self.assertEqual(len(mesh_section([[0,0,0],[2,0,0],[0,2,0]],[[0,1,2]],[3,0,0],[1,0,0])),0)

    def test_ifc_recording_captures_analysis_build_and_final_section_sweep(self):
        s=self.scene;s.event('agent_start',{}, {},actor='system')
        result=self.tools.call(s,'detect_planes',{'max_planes':1})
        self.tools.call(s,'model_plane',{'plane_id':result['planes'][0]['id'],'thickness':.2})
        s.event('agent_end',{}, {},actor='system')
        folder=s.folder/recordings(s)[0]['id'];frames=frame_paths(folder)
        self.assertGreaterEqual(len(frames),21)
        with Image.open(frames[-1]) as final,Image.open(frames[-10]) as earlier:
            self.assertEqual(final.size,(960,600))
            self.assertFalse(np.array_equal(np.array(final),np.array(earlier)))
        with Image.open(export_gif(folder)) as film:
            self.assertGreater(film.n_frames,10)
            self.assertEqual(film.size,(960,600))

if __name__=='__main__':unittest.main()
