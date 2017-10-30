#
# All workers communicate with each other via essentially a single RPC:
# fetch_parameters() which essentially requests the latest parameters
# from another worker.
#
# A flow-control mechanism is used to select the target to fetch parameters
# from. A timeout value is used to limit the time spent waiting for a
# chunk of data to arrive. Whenever the timeout expires, we decrease the score
# of the target, and we increase it any time we succeed. The score is bounded,
# and used combined with a random term to select the target.
#
# Every worker have 2 threads:
# RxThread:
#       Holds the most up-to-date model parameters and state,
#       it serves requests from the other workers.
# TxThread:
#       Initiating a fetch request to a random peer each time
#
"""Async Client/Server implementation."""
from copy import deepcopy
import logging
from threading import Thread, Lock
from queue import Queue
import select
import socket
import random

from .messaging import send_message, recv_message, MessageError


LOGGER = logging.getLogger(__name__)


MESSAGE_TYPE_FETCH_PARAMETERS = 1

# This has the effect of holding 16MB per socket. Since each node holds
# 2 sockets per peer, each node will need (16MB * 2 * nodes) of available
# memory, on a 36 nodes cluster, that equals about 1.2GB

TCP_SOCKET_BUFFER_SIZE = 8 * 1024 * 1024

def _create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


class RxThread(Thread):
    def __init__(self, bind_host, bind_port, socket_timeout_ms):
        super(RxThread, self).__init__()

        LOGGER.info("Starting RxThread. listening on %s:%d...", bind_host, bind_port)

        self.bind_host = bind_host
        self.bind_port = bind_port
        self.socket_timeout_ms = socket_timeout_ms
        self.have_state = False
        self.lock = Lock()

        # Create server socket
        self.sock = _create_tcp_socket()
        self.sock.bind((bind_host, bind_port))
        self.sock.listen(10)

        # Epoll
        self.fds = {}
        self.efd = select.epoll()
        self._register_fd(self.sock.fileno(), select.EPOLLIN, self._handle_new_connection, self.sock)

    def set_current_state(self, state, payload):
        # We're using a lock because we cannot update the state and payload
        # While it is sent to a remote peer
        with self.lock:
            self.state = deepcopy(state)
            self.payload = deepcopy(payload)
            self.have_state = True

    def _register_fd(self, fd, events, cb, arg):
        LOGGER.debug("RxThread: registering fd=%d", fd)
        assert fd not in self.fds, "fd already registered %d" % fd
        self.fds[fd] = (cb, arg)
        self.efd.register(fd, events)

    def _rearm_fd(self, fd, events):
        LOGGER.debug("RxThread: rearming fd=%d", fd)
        assert fd in self.fds, "fd not registered %d" % fd
        self.efd.modify(fd, events)

    def _unregister_fd(self, fd):
        LOGGER.debug("RxThread: unregistering fd=%d", fd)
        assert fd in self.fds, "fd not registered %d" % fd
        self.efd.unregister(fd)
        del self.fds[fd]

    def _handle_request(self, client_sock):
        try:
            # The socket is blocking
            LOGGER.debug("RxThread: receiving message fd=%d", client_sock.fileno())
            message_type, _, _ = recv_message(client_sock)
            assert message_type == MESSAGE_TYPE_FETCH_PARAMETERS

            # send the result
            if not self.have_state:
                send_message(client_sock, MESSAGE_TYPE_FETCH_PARAMETERS)
            else:
                with self.lock:
                    send_message(client_sock, MESSAGE_TYPE_FETCH_PARAMETERS, self.state, self.payload)

        except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning("Other end had a timeout, socket closed")
            self._unregister_fd(client_sock.fileno())
            client_sock.close()

        except:
            LOGGER.exception("Error handling request (closing socket, client will retry)")
            self._unregister_fd(client_sock.fileno())
            client_sock.close()

    def _handle_client_event(self, events, conn):
        fd = conn.fileno()

        # Hang-up
        if (events & select.EPOLLHUP) or (events & select.EPOLLERR):
            LOGGER.info("closed connection. fd=%d", fd)
            self._unregister_fd(fd)
            conn.close()

        # Read
        elif events & select.EPOLLIN:
            self._handle_request(conn)
        else:
            raise Exception("Unrecognized event=%d, fd=%d" % (events, fd))

    def _handle_new_connection(self, events, _):
        assert events == select.EPOLLIN

        (client_sock, address) = self.sock.accept()
        LOGGER.info("%s connected. fd=%d", address, client_sock.fileno())
        client_sock.settimeout(self.socket_timeout_ms)
        self._register_fd(client_sock.fileno(),
                          select.EPOLLIN | select.EPOLLHUP | select.EPOLLERR,
                          self._handle_client_event, client_sock)

    def run(self):
        LOGGER.info("RxThread: run()")
        try:
            while True:
                # Blocking
                events = self.efd.poll()
                for fd, events in events:
                    cb, args = self.fds[fd]
                    cb(events, args)
        finally:
            sock_fd = self.sock.fileno()
            fds = list(self.fds.keys())
            for fd in fds:
                if fd != sock_fd:
                    _, sock = self.fds[fd]
                    self.efd.unregister(fd)
                    sock.close()
            self.efd.unregister(sock_fd)
            self.sock.close()
            self.efd.close()

        LOGGER.info("TxThread: Exiting...")

    def shutdown(self):
        # TODO(guyz): Implement using eventfd...
        raise NotImplementedError

#
# Client
#

FLOW_CONTROL_MIN_SCORE = 10
FLOW_CONTROL_MAX_SCORE = 1000
FLOW_CONTROL_INC_SCORE = 10
FLOW_CONTROL_DEC_SCORE = 100


class WorkerConn:
    """Describes a peer's state in the cluster."""
    def __init__(self, name, host, port):
        self.name = name
        self.host = host
        self.port = port
        self.flow_control_score = FLOW_CONTROL_MAX_SCORE

        # Lazy connection
        self.connected = False
        self.sock = None


class TxThread(Thread):
    def __init__(self, socket_timeout_ms):
        super(TxThread, self).__init__()
        self.socket_timeout_ms = socket_timeout_ms
        self._queue = Queue(1)
        self.peers = {}
        self.peer_payload = None
        self.peer_message = None
        self.error = False
        self.peers_lock = Lock()

    def add_peer(self, name, host, port):
        LOGGER.debug("Adding peer %s (%s:%d)...", name, host, port)
        conn = WorkerConn(name, host, port)
        with self.peers_lock:
            self.peers[name] = conn
        LOGGER.debug("peer %s added.", name)

    def remove_peer(self, name):
        LOGGER.debug("Removing peer %s...", name)
        with self.peers_lock:
            peer = self.peers[name]
            if peer.connected:
                peer.sock.close()
            del self.peers[name]
        LOGGER.debug("peer %s removed.", name)

    def _get_random_peer(self):
        # The score is a weighted sum (with equal weights)
        with self.peers_lock:
            scores = {k: v.flow_control_score + \
                         random.randint(FLOW_CONTROL_MIN_SCORE, FLOW_CONTROL_MAX_SCORE) \
                      for k, v in self.peers.items()}

        if len(scores) == 0:
            LOGGER.debug("No peers were added.")
            return None

        max_score = max(scores.values())

        # There may be multiple peers with the same max score (rare)
        max_scores_keys = [k for k, v in scores.items() if v == max_score]
        key = max_scores_keys[random.randint(0, len(max_scores_keys)-1)]
        peer = self.peers[key]

        LOGGER.debug("peer %s selected score=%d, flow_control_score=%d",
                     peer.name, max_score, peer.flow_control_score)

        # Make sure the client is connected
        try:
            if not peer.connected:
                peer.sock = _create_tcp_socket()
                peer.sock.settimeout(self.socket_timeout_ms/1000)
                peer.sock.connect((peer.host, peer.port))
                peer.connected = True
                LOGGER.debug("connected to peer %s successfully", peer.name)
        except ConnectionRefusedError:
            LOGGER.debug("peer %s not listening yet", peer.name)
            self._flow_control_dec(peer)
            return None
        except:
            LOGGER.exception("Couldn't connect to peer %s (unrecoverable)", peer.name)
            self.remove_peer(peer.name)
            return None

        return peer

    def _flow_control_inc(self, peer):
        """Increase the flow control score of peer."""
        peer.flow_control_score = min(peer.flow_control_score + FLOW_CONTROL_INC_SCORE,
                                      FLOW_CONTROL_MAX_SCORE)

    def _flow_control_dec(self, peer):
        """Decrease the flow control score of peer."""
        peer.flow_control_score = max(peer.flow_control_score - FLOW_CONTROL_DEC_SCORE,
                                      FLOW_CONTROL_MIN_SCORE)

    def run(self):
        LOGGER.info("TxThread: run()")
        while True:
            witem = self._queue.get(block=True)
            LOGGER.debug("TxThread: have work...")
            if not witem:
                LOGGER.info("Exiting TxThread...")
                break

            # Wait until we succefully fetch from a peer,
            # or until we don't have any peers to fetch from
            done = False
            while not done:
                peer = self._get_random_peer()
                if peer is None:
                    self.peer_payload = None
                    self.peer_message = None
                    done = True
                    continue

                try:
                    # Send a fetch parameters request
                    LOGGER.debug("TxThread: Sending message fd=%d", peer.sock.fileno())
                    send_message(peer.sock, MESSAGE_TYPE_FETCH_PARAMETERS)
                    message_type, self.peer_message, self.peer_payload = recv_message(peer.sock)
                    assert message_type == MESSAGE_TYPE_FETCH_PARAMETERS

                    self._flow_control_inc(peer)
                    done = self.peer_payload is not None

                except socket.timeout:
                    LOGGER.warning("TxThread: peer %s timeout, restarting connection...", peer.name)
                    self._flow_control_dec(peer)
                    peer.sock.close()
                    peer.sock = None
                    peer.connected = False

                except:
                    LOGGER.exception("Error connecting with peer %s.", peer.name)
                    self.remove_peer(peer.name)

            self._queue.task_done()

        LOGGER.info("TxThread: exiting...")

    def fetch_send(self):
        """Initiate an async fetch_parameters request.

        Selects a random peer and fetch its latest parameters.
        """
        self._queue.put(True)

    def fetch_wait(self):
        """Waits for the fetch_parameters request to complete."""
        self._queue.join()
        return self.peer_message, self.peer_payload

    def shutdown(self):
        self._queue.put(False)
        self._queue.join()
        self.join()
