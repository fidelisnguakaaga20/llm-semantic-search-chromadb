# 01_basics.py
# WEEK 1 – Python basics

# 1. VARIABLES (numbers, strings, booleans)
app_name = "LLM Roadmap"
student_name = "Nguakaaga"
hours_per_day = 8
is_serious = True

print("App name:", app_name)
print("Student:", student_name)
print("Hours per day:", hours_per_day)
print("Serious about this?", is_serious)

# 2. STRINGS + F-STRINGS
welcome_message = f"Welcome, {student_name}! You study {hours_per_day} hours per day."
print(welcome_message)

# TODO: Change the message below to describe yourself in one sentence
my_bio = f"My name is {student_name} and I am learning LLM engineering."
print(my_bio)

# 3. LISTS
topics = ["Python", "Transformers", "Embeddings", "RAG", "Agents"]
print("All topics:", topics)
print("First topic:", topics[0])
print("Last topic:", topics[-1])

# TODO: Add 2 more topics to the list and print the new list
topics.append("FastAPI")
topics.append("Next.js Integration")
print("Updated topics:", topics)

# 4. DICTS (OBJECTS)
student = {
    "name": student_name,
    "daily_hours": hours_per_day,
    "goal": "LLM Engineer",
    "stack": ["Next.js", "TypeScript", "Python"]
}

print("Student dict:", student)
print("Student name from dict:", student["name"])

# TODO: Print the goal from the dict
print("Goal:", student["goal"])

# 5. FUNCTIONS
def study_message(name: str, hours: int) -> str:
    return f"{name} is studying LLM engineering for {hours} hours today."

msg = study_message(student_name, hours_per_day)
print(msg)

# TODO: Create another function that takes a topic and returns a sentence like:
# "Today I am focusing on <topic>."
def focus_on(topic: str) -> str:
    return f"Today I am focusing on {topic}."

print(focus_on("Python"))
print(focus_on("Transformers"))
