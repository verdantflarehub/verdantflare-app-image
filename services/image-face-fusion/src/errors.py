class FaceFusionPipelineError(Exception):
    def __init__(self, message, error_code="PIPELINE_ERROR", status_code=500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code

    def as_dict(self):
        return {"message": self.message, "error_code": self.error_code, "status_code": self.status_code}
