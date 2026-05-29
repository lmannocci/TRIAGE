class PromptUtils:

    @staticmethod
    def describe_binary_feature(value: int, description: str, feature_name: str, tense: str = "present") -> str:
        if tense not in {"present", "past"}:
            raise ValueError("Tense must be either 'present' or 'past'.")

        verb = "has" if tense == "present" else "had"
        
        if int(value) == 1:
            return f"{verb} {description} ('{feature_name}')"
        elif int(value) == 0:
            return f"{verb} not {description} ('{feature_name}')"
        else:
            raise ValueError("Value must be 0 or 1.")

