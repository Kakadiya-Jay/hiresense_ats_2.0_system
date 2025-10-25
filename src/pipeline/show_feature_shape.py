# tools/show_features_shape.py
import json
from src.api.services.resume_service import process_pdf

pdf_path = "/path/to/your/test/resume.pdf"  # same file that gave 29% before
processed = process_pdf(pdf_path, persist=False)

print("=== KEYS IN processed ===", list(processed.keys()))
print("\n=== FEATURE STRUCTURE PREVIEW ===")
features = processed.get("features", {})
print(json.dumps(features, indent=2)[:1500])

print("\n=== SECTION HEADERS ===")
sections = processed.get("sections", {})
print(list(sections.keys())[:20])
