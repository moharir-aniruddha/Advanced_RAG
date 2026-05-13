class ConversationMemory:

    def __init__(self):

        self.chat_history = []

    def add_message(self, role, content):

        self.chat_history.append({
            "role": role,
            "content": content
        })

    def get_history(self):

        formatted_history = ""

        for message in self.chat_history:

            formatted_history += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        return formatted_history