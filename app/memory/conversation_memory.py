class ConversationMemory:
    def __init__(self):
        # Dictionary to store history per session_id
        self.sessions = {}

    def get_history(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # Format history as a string for the prompt
        history_str = ""
        for msg in self.sessions[session_id]:
            history_str += f"{msg['role']}: {msg['content']}\n"
        return history_str

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

        # Keep only the last 10 messages to save context window space
        if len(self.sessions[session_id]) > 10:
            self.sessions[session_id] = self.sessions[session_id][-10:]