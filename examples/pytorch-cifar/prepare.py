#!/usr/bin/env python3
import re
import yaml
import os
import stat


def get_cpu_cores_info():
    cores = open('/proc/cpuinfo').read()
    cpu_ids = re.findall("processor.*: ([0-9]+)", cores)
    cpu_ids = [int(i) for i in cpu_ids]
    sockets = re.findall("physical id.*: ([0-9]+)", cores)
    sockets = [int(i) for i in sockets]
    return list(zip(cpu_ids, sockets))


def make_dpwa_config_file(cores):
    nodes = []
    for core, _ in cores:
        desc = "  - {{ name: w{}, host: localhost, port: {} }}".format(core, 45000 + core)
        nodes += [desc]

    template = open("./dpwa.yaml.t", 'rt').read()
    cfg_data = template.replace("<<<nodes>>>", '\n'.join(nodes))
    open("./dpwa.yaml", 'wt').write(cfg_data)


def make_run_file(cores):
    commands = []
    for core, _ in cores:
        cmd = "OMP_NUM_THREADS=1 taskset -c {0} python3 main.py --lr=0.01 --batch-size 8 --name w{0} --config-file ./dpwa.yaml &".format(core)
        commands += [cmd]

    template = open("./run.sh.t", 'rt').read()
    run_data = template.replace("<<<commands>>>", '\n'.join(commands))
    open("./run.sh", 'wt').write(run_data)

    # Make executable
    stats = os.stat("./run.sh")
    os.chmod("./run.sh", stats.st_mode | stat.S_IEXEC)


def main():
    cores = get_cpu_cores_info()
    print("Detected %d cores" % len(cores))

    make_dpwa_config_file(cores)
    make_run_file(cores)


if __name__ == '__main__':
    main()

