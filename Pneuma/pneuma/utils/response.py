import json
from enum import Enum


class ResponseStatus(Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class Response:
    def __init__(self, status: ResponseStatus, message: str = None, data=None):
        self.status = status
        self.message = message
        self.data = data

    def to_dict(self):
        return {"status": self.status.name, "message": self.message, "data": self.data}

    def to_json(self):
        return json.dumps(self.to_dict())
