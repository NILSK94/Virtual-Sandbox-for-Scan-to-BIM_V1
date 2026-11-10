"""Numerical integration tests for local segmentation and exact source selections."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry, CATALOG
from scanbim.segmentation import dbscan, load_selection, sample_selection
from scanbim import token_context, experiments

class LocalSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        x,y=np.meshgrid(np.linspace(0,1,16),np.linspace(0,1,16))
        first=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        self.points=np.vstack([first,first+[3,0,0],[[10,10,10]]])
        self.scene=SceneStore(Path(self.tmp.name)).create(self.points,np.full(self.points.shape,120,dtype=np.uint8),'Islands','hash')
        self.tools=ToolRegistry()

    def clusters(self):
        return self.tools.call(self.scene,'cluster_points',{'eps':.11,'min_samples':4,'min_cluster_points':20})

    def test_clusters_feed_plane_fit_and_pca_with_exact_source_ids(self):
        result=self.clusters()
        self.assertEqual(len(result['clusters']),2)
        self.assertEqual(result['noise_points'],1)
        key=result['clusters'][0]['selection_id']
        ids,sampled=load_selection(self.scene,key)
        self.assertEqual(len(ids),256);self.assertFalse(sampled)
        self.assertLess(self.scene.points[ids,0].max(),1.01)
        fit=self.tools.call(self.scene,'detect_planes',{'selection_id':key,'min_points':100,'max_planes':1})['planes'][0]
        self.assertAlmostEqual(fit['width'],1.,places=4)
        self.assertLess(fit['rms_m'],1e-10)
        inliers,_=load_selection(self.scene,fit['selection_id'])
        np.testing.assert_array_equal(inliers,ids)
        measured=self.tools.call(self.scene,'pca',{'selection_id':key})
        self.assertLess(measured['center'][0],1.)
        # Selection remains reusable after the original in-memory result is gone.
        p,_,recovered,total,sampled=sample_selection(self.scene,selection_id=key)
        np.testing.assert_array_equal(p,self.points[recovered]);self.assertEqual(total,256)

    def test_sampling_warning_survives_downstream_selection(self):
        result=self.tools.call(self.scene,'cluster_points',{'eps':.3,'min_samples':2,'sample_limit':100,'min_cluster_points':5})
        self.assertTrue(result['sampling']['sampled'])
        self.assertTrue(sample_selection(self.scene,selection_id=result['clusters'][0]['selection_id'])[-1])

    def test_untrusted_selection_id_and_bad_parameters_are_rejected(self):
        for key in ('../outside','x'*32):
            with self.assertRaises(ValueError):load_selection(self.scene,key)
        for eps in (0,float('nan'),-1):
            with self.assertRaises(ValueError):dbscan(self.points,eps=eps)

    def test_local_analysis_tools_are_documented_in_full(self):
        index=token_context.tool_index(CATALOG)
        self.assertEqual(index,CATALOG)
        self.assertIsNot(index,CATALOG)
        for name in ('cluster_points','detect_planes','pca'):
            self.assertIn('selection_id',index[name])
            self.assertIn('args:',index[name])

    def test_experiment_receives_only_cluster_points_and_does_not_mount_full_cloud(self):
        key=self.clusters()['clusters'][0]['selection_id']
        scene=self.scene
        # Mimic an imported, disk-backed cloud. A selection must prevent full dataset mounts.
        scene.cloud_metadata={'color_source':'source'}
        class Process:
            def __init__(self,cmd,**kw):
                import io
                self.stdout=io.BytesIO(b'{"ok":true,"result":{"measured":true}}');self.stderr=io.BytesIO()
                self.cmd=cmd;self.returncode=0;commands.append(cmd)
            def poll(self):return 0
        commands=[]
        with patch.object(experiments,'docker_status',return_value={'available':True,'image_id':'test'}), \
             patch.object(experiments.subprocess,'Popen',Process), \
             patch.object(experiments.subprocess,'run'):
            result=experiments.run_experiment(scene,'result={}',selection_id=key)
        self.assertTrue(result['ok'])
        self.assertEqual(result['data_info']['selection_id'],key)
        self.assertEqual(result['point_count'],256)
        self.assertFalse(any('target=/dataset/' in argument for argument in commands[0]))
        with np.load(scene.folder/('experiment-'+result['id'])/'cloud.npz') as cloud:
            self.assertLess(cloud['points'][:,0].max(),1.01)
            np.testing.assert_array_equal(cloud['points'],self.points[cloud['source_indices']])

if __name__=='__main__':unittest.main()
