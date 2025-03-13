from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import yaml


class InputModel(BaseModel):
    """Input configuration for the surf spot finder test case"""

    model_config = ConfigDict(extra="forbid")
    location: str
    date: str
    max_driving_hours: int
    model_id: str
    api_key_var: str
    json_tracer: bool
    api_base: Optional[str] = None
    agent_type: str


class CheckpointCriteria(BaseModel):
    """Represents a checkpoint criteria with a value and description"""

    model_config = ConfigDict(extra="forbid")
    value: int
    criteria: str


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: InputModel
    ground_truth: Dict[str, Any]
    checkpoints: List[CheckpointCriteria] = Field(default_factory=list)
    final_answer_criteria: List[CheckpointCriteria] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, case_path: str) -> "TestCase":
        """Load a test case from a YAML file and process it"""
        with open(case_path, "r") as f:
            test_case_dict = yaml.safe_load(f)

        # Generate final_answer_criteria if not explicitly provided
        if "final_answer_criteria" not in test_case_dict:
            final_answer_criteria = []

            def add_gt_final_answer_criteria(ground_truth_dict, prefix=""):
                """Recursively add checkpoints for each value in the ground_truth dictionary"""
                for key, value in ground_truth_dict.items():
                    path = f"{prefix}: {key}" if prefix else key
                    if isinstance(value, dict):
                        add_gt_final_answer_criteria(value, path)
                    else:
                        final_answer_criteria.append(
                            {
                                "value": 1,
                                "criteria": f"Check if {path} is approximately '{value}'.",
                            }
                        )

            add_gt_final_answer_criteria(test_case_dict["ground_truth"])
            test_case_dict["final_answer_criteria"] = final_answer_criteria

        return cls.model_validate(test_case_dict)
