import unittest
from campaign import candidates, create_manifest
class MatrixTests(unittest.TestCase):
 def test_requested_topologies_have_exact_budget(self):
  rows=candidates()
  self.assertEqual(sum(r['model']=='8b' for r in rows),16)
  self.assertEqual(sum(r['model']=='70b' for r in rows),11)
  self.assertEqual(len({r['id'] for r in rows}),27)
  for r in rows:
   self.assertEqual(r['tp']*r['pp']*r['replicas'],4)
   if r['family']=='muxwise':self.assertEqual(r['pp'],1)
 def test_conditional70b_are_gated_and_unmeasured(self):
  m=create_manifest()
  self.assertEqual(sum(r['conditional_two_gpu'] for r in m['candidates']),5)
  for r in m['candidates']:
   self.assertEqual(r['status'],'not_executed')
   self.assertIsNone(r['selected_parameters'])
  self.assertFalse(m['formal_ready'])
if __name__=='__main__':unittest.main()
