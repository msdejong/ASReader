

class Data:
    def __init__(self, document_tokens, query_tokens, answer_tokens, entity_locations):
        self.document_tokens = document_tokens
        self.query_tokens = query_tokens
        self.answer_tokens = answer_tokens
        self.entity_locations = entity_locations