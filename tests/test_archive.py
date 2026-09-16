"""No hardware or real participant data. Synthetic interface regression tests only.
Run: python -m unittest discover -s tests -v
"""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp)
    sys.modules[name]=m;sp.loader.exec_module(m);return m

class StructureTest(unittest.TestCase):
    def test_manifest_counts(self):
        h=load('hgr_launcher_test',ROOT/'tools/hgr.py')
        expected={'smoke':2,'table5':75,'backbone':135,'labels':150,'five_class':75,'lr_quick':6,'lr':75}
        for name,n in expected.items():self.assertEqual(len(h.tasks(name)[1]),n)
    def test_sanitized_notebook(self):
        n=json.loads((ROOT/'acquisition/infineon/notebook_example_sanitized.ipynb').read_text())
        self.assertTrue(all(not c.get('outputs') for c in n['cells']))
    def test_geometry_prior_upload_identity_is_documented(self):
        self.assertTrue((ROOT/'legacy/geometry/processing_geometry_GPT.py').is_file())
        self.assertTrue((ROOT/'recovered_prior_uploads/Make_Merged_data_v4.py').is_file())

class ModelInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch,numpy,pandas
        except ImportError as e:raise unittest.SkipTest('Analysis dependencies unavailable: '+str(e))
        cls.torch=torch;torch.set_num_threads(1)
        cls.np=numpy;cls.pd=pandas
        cls.m=load('fusion_engine_test',ROOT/'experiments/fusion_protocol/run_fusion_protocol_suite.py')
        cls.r=load('revision_engine_test',ROOT/'experiments/revision/run_fusion_protocol_suite.py')
    def test_all_fusion_backbone_forward_shapes(self):
        t=self.torch
        with t.no_grad():
            for back in ['tcn','gru','transformer']:
                for method in ['radar_only','early_fusion','late_fusion','attention_fusion','motion_domain_proposed']:
                    if method in ['late_fusion','attention_fusion']:dims={'radar':12,'imu':12}
                    else:dims={'single':{'radar_only':12,'early_fusion':24,'motion_domain_proposed':11}[method]}
                    a=self.m.build_model_for_experiment(method,back,dims,4,64,.15,40).eval()
                    b=self.r.build_model_for_experiment(method,back,dims,4,64,.15,40).eval()
                    b.load_state_dict(a.state_dict())
                    xs=(t.randn(2,40,12),t.randn(2,40,12)) if 'radar' in dims else (t.randn(2,40,dims['single']),)
                    ya=a(*xs);yb=b(*xs)
                    self.assertEqual(tuple(ya.shape),(2,4));self.assertTrue(t.isfinite(ya).all())
                    self.assertTrue(t.equal(ya,yb))
    def test_feature_shapes_and_corrected_priority(self):
        n=self.np;p=self.pd
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/'synthetic.csv'
            frame=p.DataFrame({'range':n.ones(40),'range_corr':n.ones(40)*9,
              'doppler':n.linspace(-2,2,40),'horizontal_angle':n.linspace(-10,10,40),'vertical_angle':n.zeros(40),
              'q.w':n.ones(40),'q.i':n.zeros(40),'q.j':n.zeros(40),'q.k':n.zeros(40)})
            frame.to_csv(f,index=False)
            radar,imu,motion=self.m.read_csv_features(str(f),40)
            self.assertEqual(radar.shape,(40,12));self.assertEqual(imu.shape,(40,12));self.assertEqual(motion.shape,(40,11))
            self.assertTrue(n.all(radar[:,0]==9))
            self.assertTrue(n.isfinite(motion).all())
    def test_tcn_parameter_count(self):
        x=self.m.build_model_for_experiment('motion_domain_proposed','tcn',{'single':11},4,64,.15,40)
        self.assertEqual(self.m.count_params(x),51460)
if __name__=='__main__':unittest.main()
