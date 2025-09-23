import json

class ToolInputSchema:
    def __init__(self, schema_or_schemas: dict | list[dict]):
        if schema_or_schemas is None or (isinstance(schema_or_schemas, list) and len(schema_or_schemas) < 1):
            raise ValueError("One or more input schemas must be provided")

        self.schema_or_schemas: dict | list[dict] = schema_or_schemas
        self.multiple_schemas: bool = isinstance(self.schema_or_schemas, list)

    def get_allowed_keys(self) -> set:
        if not self.multiple_schemas:
            return set(self.schema_or_schemas.keys())
        else:
            keys = set()
            for schema in self.schema_or_schemas:
                keys.update(schema.keys())
            return keys
    
    def to_string(self) -> str:
        return json.dumps(self.schema_or_schemas, ensure_ascii=False)