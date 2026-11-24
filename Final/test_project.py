import unittest
import numpy as np
from analysis_core import *
class ProjectTests(unittest.TestCase):
    def test_data_and_filters(self):
        df=load_data()
        for label,n in [('All',569),('Malignant',212),('Benign',357)]:
            subset=filter_data(df,label)
            self.assertEqual(len(subset),n)
            if label!='All': self.assertEqual(set(subset.Diagnosis),{label})
            for x in FEATURES:
                for y in FEATURES:
                    self.assertEqual(len(subset[[x,y]]),n)
    def test_metrics(self):
        df=load_data(); a=evaluate(df); b=evaluate(df)
        self.assertEqual(a[0],b[0]); np.testing.assert_array_equal(a[5],b[5]); np.testing.assert_array_equal(a[1],[[62,2],[1,106]])
        self.assertEqual(a[3:5],(398,171))
        self.assertAlmostEqual(a[0]['Accuracy'],168/171)
    def test_streamlit(self):
        try: from streamlit.testing.v1 import AppTest
        except ImportError: self.skipTest('Streamlit not installed')
        from pathlib import Path
        app=AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs),3)
        self.assertEqual(len(app.selectbox),3)
        for label,n in [('Malignant',212),('Benign',357),('All',569)]:
            app.selectbox[0].select('mean texture'); app.selectbox[1].select('worst radius'); app.selectbox[2].select(label); app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any(f'Selected observations: {n}' in x.value for x in app.markdown))
        self.assertTrue(any(x.label=='Accuracy' and x.value=='98.25%' for x in app.metric))
if __name__=='__main__': unittest.main(verbosity=2)
