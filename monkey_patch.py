import rl_games.common.experience as experience
from torch_geometric.data import Data, Batch
from typing import List

BaseExpBuffer = experience.ExperienceBuffer


class ExperienceBufferPyG(BaseExpBuffer):
    """This class overrides the rl_games.common.experience.ExperienceBuffer to handle PyG Batch object in GNN"""

    def _init_from_env_info(self, env_info):
        super()._init_from_env_info(env_info)

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.tensor_dict[name][k][index, :] = v
        elif type(val) is Batch:
            self.tensor_dict[name][index] = val.to_data_list()
        else:
            self.tensor_dict[name][index, :] = val

    def update_data_rnn(self, name, indices, play_mask, val):
        return NotImplementedError

    def _do_op(self, v, transform_op):
        if type(v) is dict:
            transformed_dict = {}
            for kd, vd in v.items():
                transformed_dict[kd] = transform_op(vd)
            return transformed_dict
        elif (type(v) is List) and (type(v[0]) is Batch):
            # assert transform_op == a2c_common.swap_and_flatten01
            tmp = [Batch.from_data_list([v[j][i] for j in range(self.horizon_length)]) for i in range(self.num_actors)]
            return Batch.from_data_list(tmp)
        else:
            return transform_op(v)

    def get_transformed(self, transform_op):
        res_dict = {}
        for k, v in self.tensor_dict.items():
            self._do_op(v, transform_op)
        return res_dict

    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            self._do_op(v, transform_op)
        return res_dict


experience.ExperienceBuffer = ExperienceBufferPyG
