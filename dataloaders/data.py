

class Data:
    def __init__(self, document_tokens, query_tokens, answer_token, entity_to_index, index_to_entity):
        self.document_tokens = document_tokens
        self.query_tokens = query_tokens
        self.answer_token = answer_token
        self.entity_to_index = entity_to_index
        self.index_to_entity = index_to_entity
