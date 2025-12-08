# 02_files.py
# WEEK 1 – File handling

# 1. WRITE a study plan to a text file
study_plan = [
    "Day 1: Python basics (variables, lists, dicts, functions)",
    "Day 2: File handling + Jupyter",
    "Day 3: Practice exercises",
    "Day 4: Mini project (study tracker)",
    "Day 5: Review + HuggingFace preview"
]

with open("study_plan.txt", "w", encoding="utf-8") as f:
    for line in study_plan:
        f.write(line + "\n")

print("✅ study_plan.txt written.")

# 2. READ the file back and print it
print("\n📄 Reading study_plan.txt:\n")

with open("study_plan.txt", "r", encoding="utf-8") as f:
    content = f.read()

print(content)

# 3. APPEND a new line
with open("study_plan.txt", "a", encoding="utf-8") as f:
    f.write("Extra: Build 1 small LLM tool every week.\n")

print("✅ Extra line appended.")
