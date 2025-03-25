from typing import Dict, List, Any

from surf_spot_finder.evaluation.evaluators.LLMEvaluator import LLMEvaluator
from surf_spot_finder.evaluation.evaluators.schemas import EvaluationResult
from surf_spot_finder.evaluation.telemetry import TelemetryProcessor
from surf_spot_finder.evaluation.test_case import CheckpointCriteria


class CheckpointEvaluator(LLMEvaluator):
    """Evaluates checkpoints against telemetry"""

    def evaluate(
        self,
        telemetry: List[Dict[str, Any]],
        checkpoints: List[CheckpointCriteria],
        processor: TelemetryProcessor,
    ) -> List[EvaluationResult]:
        """
        Verify each checkpoint against the telemetry data using LLM

        Args:
            telemetry: The telemetry data to evaluate
            checkpoints: List of checkpoint criteria to verify
            processor: Telemetry processor to extract evidence

        Returns:
            List of evaluation results
        """
        evidence = processor.extract_evidence(telemetry)
        results = []

        for checkpoint in checkpoints:
            evaluation = self.llm_evaluate_with_criterion(
                criteria=checkpoint.criteria,
                points=checkpoint.points,
                evidence=evidence,
            )
            results.append(evaluation)

        return results
