from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import yaml


class InputModel(BaseModel):
    """Input configuration for the surf spot finder test case"""

    model_config = ConfigDict(extra="forbid")
    location: str
    date: str
    max_driving_hours: int
    json_tracer: bool


class AgentModel(BaseModel):
    model_id: str
    api_key_var: str = "OPENAI_API_KEY"
    api_base: Optional[str] = None
    agent_type: str
    tools: Optional[List[str]] = None


class CheckpointCriteria(BaseModel):
    """Represents a checkpoint criteria with a description"""

    model_config = ConfigDict(extra="forbid")
    criteria: str
    points: int


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: InputModel
    agent: AgentModel
    ground_truth: List[Dict[str, Any]] = Field(default_factory=list)
    checkpoints: List[CheckpointCriteria] = Field(default_factory=list)
    final_answer_criteria: List[CheckpointCriteria] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, test_case_path: str, agent_config_path: str) -> "TestCase":
        """Load a test case from a YAML file and process it"""
        with open(test_case_path, "r") as f:
            test_case_dict = yaml.safe_load(f)

        with open(agent_config_path, "r") as f:
            agent_config_dict = yaml.safe_load(f)
        test_case_dict["agent"] = agent_config_dict["agent"]
        final_answer_criteria = []

        def add_gt_final_answer_criteria(ground_truth_list):
            """Add checkpoints for each item in the ground_truth list"""
            for item in ground_truth_list:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    points = item.get(
                        "points", 1
                    )  # Default to 1 if points not specified
                    final_answer_criteria.append(
                        {
                            "points": points,
                            "criteria": f"Check if {item['name']} is approximately '{item['value']}'.",
                        }
                    )

        add_gt_final_answer_criteria(test_case_dict["ground_truth"])
        test_case_dict["final_answer_criteria"] = final_answer_criteria
        # remove the points from the ground_truth list but keep the name and value
        test_case_dict["ground_truth"] = [
            item for item in test_case_dict["ground_truth"] if isinstance(item, dict)
        ]

        return cls.model_validate(test_case_dict)
