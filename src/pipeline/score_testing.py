# tools/test_end_to_end.py
import json
from src.api.services.resume_service import process_pdf, STORE
from src.phases.scoring.helpers.scorer import score_candidate

pdf_path = "data/raw_resumes/my_resumes/Jay_Kakadiya_Fresher_Resume_for_placement.pdf"   # replace with an actual saved test PDF path you previously uploaded
processed = process_pdf(pdf_path, persist=True)
print("Processed doc keys:", list(processed.keys()))
print("Has candidate_json:", "candidate_json" in processed and bool(processed["candidate_json"]))
print("candidate_json keys counts:", {k: len(v) if isinstance(v, list) else 0 for k,v in processed.get("candidate_json", {}).items()})
# now score
jd = "Looking for Python developer who is able to write code for machine learning and take responsibility as a data scientist."
res = score_candidate(processed, jd)
print(json.dumps(res, indent=2))
