
import re

file_path = "match_data.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Search for "events": [ and look backwards for the start of the object
match = re.search(r'"events":\s*\[', content)
if match:
    pos = match.start()
    # Go back 10000 characters and look for common starts like "var matchCentreData =" or just "{"
    snippet_before = content[max(0, pos-5000):pos]
    print(f"Snippet before 'events':\n{snippet_before[-500:]}")
    
    # Try to find the start of the JSON block
    # Often it's in a script tag
    script_start = content.rfind("<script", 0, pos)
    if script_start != -1:
        print(f"Script tag starts at {script_start}")
        print(f"Snippet from script start:\n{content[script_start:script_start+500]}")
