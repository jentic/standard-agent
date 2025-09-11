import json

class JSONParseError(json.JSONDecodeError):  
    """JSONDecodeError that preserves the raw content that failed parsing."""  
    def __init__(self, msg: str, doc: str, pos: int, raw_content: str):  
        super().__init__(msg, doc, pos)  
        self.raw_content = raw_content 