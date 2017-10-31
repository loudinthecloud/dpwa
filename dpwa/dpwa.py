"""Dpwa P2P connection."""
import logging
import random
import yaml

from .conn import RxThread, TxThread
from .interpolation import ConstantInterpolation, \
                           ClockWeightedInterpolation, LossInterpolation


INTERPOLATION_METHODS = {
    'constant': ConstantInterpolation,
    'clock': ClockWeightedInterpolation,
    'loss': LossInterpolation,
}


LOGGER = logging.getLogger(__name__)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return 'Struct: ' + repr(self.__dict__)


class DpwaConfiguration:
    def __init__(self, config_file):
        self.yaml = yaml.load(open(config_file, 'rt'))
        self.config = {}
        for c in self.yaml:
            k = list(c.keys())[0]
            self.config[k] = c[k]

    def get_nodes(self):
        return self.config['nodes']

    def get_interpolation(self):
        interpolation = self.config['interpolation']
        return (interpolation, self.config[interpolation])

    def get_timeoutms(self):
        return self.config['timeout_ms']

    def get_fetch_probability(self):
        return self.config['fetch_probability']

    def get_divergence_threshold(self):
        return self.config['divergence_threshold']


class DpwaConnection:
    def __init__(self, name, config_file):
        self.name = name
        # The clock is used to keep track of the model's age in terms of
        # training samples trained so far (increase by 1 in update_send())
        self.clock = 0
        self.config = DpwaConfiguration(config_file)
        self.nodes = self.config.get_nodes()
        self.fetch_probability = self.config.get_fetch_probability()
        self.fetching = False

        # Initialize the list of peers
        self.peers = []
        for node in self.nodes:
            node = Struct(**node)
            if node.name == name:
                self.me = node
            else:
                self.peers += [node]

        # Initialize the interpolatoin method
        interpolation_method, interpolation_config = self.config.get_interpolation()
        LOGGER.debug("Using %s interpolation method", interpolation_method)
        if interpolation_config == 0:
            interpolation_config = {}
        self.interpolation = INTERPOLATION_METHODS[interpolation_method](**interpolation_config)
        self.divergence_threshold = self.config.get_divergence_threshold()

        # Create the client/server threads
        timeout_ms = self.config.get_timeoutms()
        self.rx = RxThread(self.me.host, self.me.port, timeout_ms)
        self.tx = TxThread(timeout_ms)

        # Add all the peers
        for peer in self.peers:
            self.add_peer(peer.name, peer.host, peer.port)

        # Start the threads
        self.rx.start()
        self.tx.start()

    def add_peer(self, name, host, port):
        self.tx.add_peer(name, host, port)

    def remove_peer(self, name):
        self.tx.remove_peer(name)

    def _bernouli_trial(self, probability):
        return random.random() < probability

    def update_send(self, parameters, loss):
        """Initiate an update to the cluster.

        Performs 2 things:
        1. Updates the local server with the latest parameters, so other peers could fetch them
        2. Initiate a fetch parameters request to a random peer.
        """
        # Increase the clock value
        self.clock += 1

        # Serve the new parameters
        state = {'clock': self.clock, 'loss': loss}
        self.rx.set_current_state(state, parameters)

        if self._bernouli_trial(self.fetch_probability):
            LOGGER.debug("update_send(): starting fetch parameters request")
            self.fetching = True
            self.tx.fetch_send()
        else:
            self.fetching = False

    def update_wait(self, loss):
        """Waits for the cluster update to finish.

        Waiting for the fetch parameters request to complete (blocking)
        """
        if not self.fetching:
            return None, 0

        peer_state, peer_parameters = self.tx.fetch_wait()
        self.fetching = False
        # There may be no peers listening
        if peer_parameters is None:
            return None, 0

        peer_clock = peer_state['clock']
        peer_loss = peer_state['loss']

        # Calculate the averaging factor using the interpolation method chosen
        factor = self.interpolation(self.clock, peer_clock, loss, peer_loss)

        # Diverge models as loss decrease below the threshold
        if loss < self.divergence_threshold:
            factor = factor * (loss / self.divergence_threshold)

        # Update local clock
        new_clock = factor * peer_clock + (1 - factor) * self.clock

        LOGGER.debug("update_wait(): (clock=%f, peer_clock=%f), (loss=%s, peer_loss=%f) => factor=%f, new_clock=%f",
                     self.clock, peer_clock, loss, peer_loss, factor, new_clock)

        self.clock = new_clock
        return peer_parameters, factor
