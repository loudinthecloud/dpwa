import logging
import pickle
import torch
import numpy as np
from ..dpwa import DpwaConnection


LOGGER = logging.getLogger(__name__)


TYPE_CONVERSION = {
    'torch.cuda.FloatTensor': np.float32,
    'torch.FloatTensor': np.float32
}


def _tensor_to_buffer(t):
    return bytes(t.cpu().numpy())


def _tensor_from_buffer_like(buf, t):
    n = np.frombuffer(buf, dtype=TYPE_CONVERSION[t.type()])
    result = torch.from_numpy(n).view(t.size())
    if t.is_cuda:
        result = result.cuda()
    return result


def _serialize_bytes_dict(params):
    return pickle.dumps(params)


def _deserialize_bytes_dict(blob):
    return pickle.loads(blob)


class DpwaPyTorchAdapter:
    def __init__(self, net, name, config_file):
        self._net = net
        self._conn = DpwaConnection(name, config_file)

    def update_send(self, loss):
        """Initiate an update to the cluster.

        Performs 2 things:
        1. Updates the local server with the latest parameters, so other peers could fetch them
        2. Initiate a fetch parameters request to a random peer.
        """
        params = {}
        for name, param in self._net.named_parameters():
            params[name] = _tensor_to_buffer(param.data)
        blob = _serialize_bytes_dict(params)
        self._conn.update_send(blob, loss)

    def update_wait(self, loss):
        """Waits for the cluster update to finish.

        Waiting for the fetch parameters request to complete (blocking)
        """
        blob, factor = self._conn.update_wait(loss)
        if blob is None:
            return

        other_params = _deserialize_bytes_dict(blob)
        LOGGER.debug("Averaging. factor=%f", factor)
        for name, param in self._net.named_parameters():
            t = _tensor_from_buffer_like(other_params[name], param.data)
            param.data = factor * t + (1 - factor) * param.data
