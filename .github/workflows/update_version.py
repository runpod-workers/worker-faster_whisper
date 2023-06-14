"""
Checks for an update of the runpod package
"""

new_version = "0.0.1"

# Update requirements.txt
with open("./builder/requirements.txt", "r", encoding="UTF-8") as req_file:
    requirements = req_file.readlines()

for i, line in enumerate(requirements):
    if line.startswith("runpod=="):
        requirements[i] = f"runpod=={new_version}\n"
        break

with open("./builder/requirements.txt", "w", encoding="UTF-8") as req_file:
    req_file.writelines(requirements)
