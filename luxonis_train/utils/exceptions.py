class IncompatibleException(Exception):
    """Raised when two parts of the model are incompatible with each
    other."""

    @classmethod
    def from_missing_task(
        cls, task: str, present_tasks: list[str], class_name: str
    ):
        return cls(
            f"{class_name} requires '{task}' label, but it was not found in "
            f"the label dictionary. Available labels: {present_tasks}."
        )
