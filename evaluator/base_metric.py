import torch
from recbole.utils import EvaluatorType


class AbstractMetric(object):
    """:class:`AbstractMetric` is the base object of all metrics. If you want to
        implement a metric, you should inherit this class.

    Args:
        config (Config): the config of evaluator.
    """

    smaller = False

    def __init__(self, config):
        self.decimal_place = config["metric_decimal_place"]

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
        """
        raise NotImplementedError("Method [calculate_metric] should be implemented.")



class LossMetric(AbstractMetric):
    """:class:`LossMetric` is a base object of loss based metrics and AUC. If you want to
    implement an loss based metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """

    metric_type = EvaluatorType.VALUE
    metric_need = ["rec.score", "data.label"]

    def __init__(self, config):
        super().__init__(config)

    def used_info(self, dataobject):
        """Get scores that model predicted and the ground truth."""
        preds = dataobject.get("rec.score")
        trues = dataobject.get("data.label")

        return preds.squeeze(-1).numpy(), trues.squeeze(-1).numpy()

    def output_metric(self, metric, dataobject):
        preds, trues = self.used_info(dataobject)
        result = self.metric_info(preds, trues)
        return {metric: round(result, self.decimal_place)}

    def metric_info(self, preds, trues):
        """Calculate the value of the metric.

        Args:
            preds (numpy.ndarray): the scores predicted by model, a one-dimensional vector.
            trues (numpy.ndarray): the label of items, which has the same shape as ``preds``.

        Returns:
            float: The value of the metric.
        """
        raise NotImplementedError(
            "Method [metric_info] of loss-based metric should be implemented."
        )
