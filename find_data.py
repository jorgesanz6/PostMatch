
import re

file_path = "match_data.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Search for patterns like "matchCentreData" or "events": [
patterns = [
    r'matchCentreData\s*=\s*({.*?});',
    r'"events":\s*\[',
    r'require\.config\(.*?\)',
    r'Haaland'
]

for pattern in patterns:
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print(f"Found search pattern: {pattern}")
        # Print a snippet around the match
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 500)
        print(f"Snippet: {content[start:end]}")
    else:
        print(f"Pattern NOT found: {pattern}")
