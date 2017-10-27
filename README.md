# Distributed Learning by Gossip Averaging Implementation

## Introduction

Performs distributed learning using a peer-to-peer parameter averaging approach. The method is described in [How to scale distributed deep learning?](https://arxiv.org/abs/1611.04581) and [Gossip training for deep learning](https://arxiv.org/abs/1611.09726).

I named the library **distributed pair-wise averaging (dpwa)**, as it is generic enough to be used with multiple optimizers or training methods.

This implementation can be used as a starting point for people who want to experiment with async p2p training algorithms. I haven't thoroughly tested how it scales so I can't report anything interesting here. You may see the papers above to learn more.

## Implementation Notes

* Currently there's only a single adapter for [PyTorch](http://pytorch.org/), but it can be extended easily to other frameworks as well (see  `dpwa/adapters`)
* As can be seen in the [pytorch-cifar example](examples/pytorch-cifar), integration to existing project requires reading a config file and supplying the current node's name as argument.
* There are multiple interpolation methods to calculate the averaging/interpolation factor in each averaging step, the papers listed above use a constant interpolation factor but I tried several other interpolation methods as well, see [here](dpwa/interpolation.py) for details. Unfortunately, I don't have the capacity at the moment to properly test and compare the methods, so they are there for reference only purpose.

### Implemetation Details

Each node creates 2 threads, one that accepts connections from the rest of the cluster and one for sending a **pull request** to a random node (chosen uniformly w/ flow control mechanism). The `RxThread` is serving the current parameters + state to any other peer. The `TxThread` request a random peer's parameters. Both sending/recieving is done asynchrounousely to training the network. The state is composed of the current `loss` and `clock` and is sent with the parameters. The `clock` is a sequence number that represents the age of the model, in term of training samples trained so far. The `loss` is updated by the user after each training iteration. The `clock` and `loss` values may be used to calculate the value of the interpolation factor.

## Configuration

Configuration is supplied using a `.yaml` file. A sample configuration file can be copied from the `samples/` directory (see [here](samples/config.yaml)).

The configuration file holds the following information:

* `nodes`: a list of nodes participating in the cluster
* `pull_probability`: probability of initiating a pull request to a random node, this can be a replacement to the **tau** variable used by the papers
* `timeout_ms`: the timeout of each socket send/recv, a simple flow control mechanism is used for selecting the target peer, combined with a random term. In each timeout, we decrease the flow control score of the peer so it'll be less likely to get picked in the next iteration.
* `interpolation`: the name of the method to be used to set the interpolation factor.
* The rest of the configuration is for setting individual interpolation methods configuration.

If you have any question or comment, feel free to contact me via email (see one of the commits), twitter (@guyzana) or whatever ;)