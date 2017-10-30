# TCP System Tuning

The following command is for configuring the system. It will set the socket buffer sizes to 8MB for both read and write. It will improve performance.

> sudo sysctl -p ./sysctl.conf

Note that each node holds 2 open sockets to any other peers in the cluster. This has a memory requirement, for example, if you're running a node per core on a 36 cores machine, then just the socket buffers will require 36 * 36 * 8MB * 2 * 2 = 41GB of RAM.

The implementation can be modified to reduce the memory footprint, with the tradeoff of increasing the latency by openning a new connection with each pair-wise interaction, this actually may not be so important.
