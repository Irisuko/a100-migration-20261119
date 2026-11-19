import hashlib,os,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from migration_tools import is_70b,filter_json,DROP,verify_tree,structured_format
class MigrationTest(unittest.TestCase):
 def test_model_boundary_not_hash(self):
  self.assertTrue(is_70b("Llama-3.3-70B-Instruct"));self.assertTrue(is_70b("llama33_70b_formal"));self.assertFalse(is_70b("aabb70b1234"))
 def test_compound_aliases_are_excluded(self):
  for v in ["chunked70b-latency256-v1","main70b","audit70b","first70b"]:self.assertTrue(is_70b(v),v)
  for v in ["e270ba8e-a761-44fa","aabb70b1234","abcd70b"]:self.assertFalse(is_70b(v),v)
 def test_nested_run_identity_drops_entire_record(self):
  self.assertIs(filter_json({"args":{"model":"70b"},"measured_tps":123}),DROP)
  self.assertIs(filter_json({"command":["python","run.py","--model","70b"],"status":"failed"}),DROP)
 def test_compound_json_backups(self):
  self.assertEqual(structured_format("campaign.json.base"),"json")
  self.assertEqual(structured_format("requests.jsonl.bak"),"jsonl")
 def test_id_case_identity_and_shared_8b_parameter_projection(self):
  for k in ["id","case"]:self.assertIs(filter_json({k:"70B two-card instance","memory_fraction":0.91}),DROP)
  self.assertEqual(filter_json({"normal_8b_or_70b_four_gpu_instance":{"mem":0.85}}),{"normal_8b_four_gpu_instance":{"mem":0.85}})
 def test_drop_entire_70b_record_keep_8b_exact(self):
  good={"model":"8b","parameters":{"tp":4},"tokens":[1,2]};bad={"model":"70b","parameters":{"tp":8},"tokens":[3]}
  self.assertEqual(filter_json([good,bad]),[good]);self.assertIs(filter_json(bad),DROP)
 def test_nested_mixed_config(self):
  self.assertEqual(filter_json({"8b":{"tp":4},"70b":{"tp":8},"models":[{"path":"models/Llama-3.3-70B-Instruct","layers":80},{"path":"models/Llama-3.1-8B-Instruct","layers":32}]}),{"8b":{"tp":4},"models":[{"path":"models/Llama-3.1-8B-Instruct","layers":32}]})
 def test_integrity_and_link_and_missing_detection(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"file").write_bytes(b"data");os.chmod(p/"file",0o640);(p/"link").symlink_to("file")
   rows=[{"path":"file","type":"file","size":4,"sha256":hashlib.sha256(b"data").hexdigest(),"mode":0o640},{"path":"link","type":"symlink","target":"file"}]
   self.assertEqual(verify_tree(p,rows),[])
   (p/"file").write_bytes(b"xxxx");self.assertTrue(verify_tree(p,rows))
   (p/"file").write_bytes(b"data");(p/"link").unlink();(p/"link").symlink_to("missing");self.assertTrue(verify_tree(p,rows))
   (p/"file").unlink();self.assertTrue(verify_tree(p,rows))
 def test_reject_traversal_manifest(self):
  with tempfile.TemporaryDirectory() as d:self.assertTrue(verify_tree(Path(d),[{"path":"../escape","type":"file"}]))
if __name__=="__main__":unittest.main()
