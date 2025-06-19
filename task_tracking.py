class Subtask:
    def __init__(self, name):
        self.name = name
        self.completed = False

    def complete(self):
        self.completed = True

class Task:
    def __init__(self, name, subtasks):
        self.name = name
        self.subtasks = [Subtask(s) for s in subtasks]

    def progress(self):
        total = len(self.subtasks)
        done = sum(s.completed for s in self.subtasks)
        return done / total if total else 0

    def mark_subtask_complete(self, subtask_name):
        for s in self.subtasks:
            if s.name == subtask_name:
                s.complete()
                return True
        return False
    
    def get_progress_text(self):
        lines = []
        for s in self.subtasks:
            status = "✓" if s.completed else "✗"
            lines.append(f"{status} {s.name}")
        return lines
    