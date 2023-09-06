import copy

import torch
from recbole.evaluator.register import Register


class DataStruct(object):
    def __init__(self):
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value: torch.Tensor):
        if name not in self._data_dict:
            self._data_dict[name] = value.cpu().clone().detach()
        else:
            if not isinstance(self._data_dict[name], torch.Tensor):
                raise ValueError("{} is not a tensor.".format(name))
            self._data_dict[name] = torch.cat(
                (self._data_dict[name], value.cpu().clone().detach()), dim=0
            )

    def __str__(self):
        data_info = "\nContaining:\n"
        for data_key in self._data_dict.keys():
            data_info += data_key + "\n"
        return data_info


class Collector(object):
    """The collector is used to collect the resource for evaluator.
    As the evaluation metrics are various, the needed resource not only contain the recommended result
    but also other resource from data and model. They all can be collected by the collector during the training
    and evaluation process.

    This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.device = self.config["device"]

    def data_collect(self, train_data):
        """Collect the evaluation resource from training data.
        Args:
            train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need("data.num_items"):
            item_id = self.config["ITEM_ID_FIELD"]
            self.data_struct.set(
                "data.num_items", train_data.dataset.num(item_id))
        if self.register.need("data.num_users"):
            user_id = self.config["USER_ID_FIELD"]
            self.data_struct.set(
                "data.num_users", train_data.dataset.num(user_id))
        if self.register.need("data.count_items"):
            self.data_struct.set("data.count_items",
                                 train_data.dataset.item_counter)
        if self.register.need("data.count_users"):
            self.data_struct.set("data.count_items",
                                 train_data.dataset.user_counter)

    def eval_batch_collect(
        self,
        scores_tensor: torch.Tensor,
        interaction
    ):
        """Collect the evaluation resource from batched eval data and batched model output.
        Args:
            scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
            interaction(Interaction): batched eval data.
            positive_u(Torch.Tensor): the row index of positive items for each user.
            positive_i(Torch.Tensor): the positive item id for each user.
        """

        if self.register.need("rec.score"):

            self.data_struct.update_tensor("rec.score", scores_tensor)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor(
                "data.label", interaction[self.label_field].to(self.device)
            )

    def model_collect(self, model: torch.nn.Module):
        """Collect the evaluation resource from model.
        Args:
            model (nn.Module): the trained recommendation model.
        """
        pass

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """Collect the evaluation resource from total output and label.
        It was designed for those models that can not predict with batch.
        Args:
            eval_pred (torch.Tensor): the output score tensor of model.
            data_label (torch.Tensor): the label tensor.
        """
        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", eval_pred)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor(
                "data.label", data_label.to(self.device))

    def get_data_struct(self):
        """Get all the evaluation resource that been collected.
        And reset some of outdated resource.
        """
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ["rec.score", "rec.items", "data.label"]:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct
