import time
import psutil
import datetime
import socket
import multiprocessing
from threading import Thread, Event

import wandb.errors
from scapy.all import sniff, conf, ETH_P_ALL
from wandb.wandb_agent import logger

try:
    from modules.monitoring.power_reader import PowerDBHandler
except ModuleNotFoundError:
    from power_reader import PowerDBHandler


class HWMonitor(Thread):

    do_run = True

    def __init__(self, monitoring_freq: float = 1.0, proc_ip_addrs: list = None, use_scapy=True, logger: logger = None,
                 stop_event=None):
        """
        :param monitoring_freq: Monitoring frequency in samples per second (float)
        :param proc_ip_addrs: IP addresses used in process-specific network traffic monitoring (with scapy) (list)
        :param use_scapy: Indicates whether process-specific monitoring should be used
        :param logger: A logger instance used to store the monitoring results.
        """
        super(HWMonitor, self).__init__(daemon=True)
        self.monitoring_freq = monitoring_freq
        self.proc_ip_addrs = proc_ip_addrs
        self.use_scapy = use_scapy
        self.stop_monitor = stop_event

        self.psutil_monitor = SystemStats()
        self.power_db = PowerDBHandler()

        if use_scapy:
            self.scapy_monitor = ScapyNetworkMonitor()
            self.scapy_monitor.start()

        self.logger = logger

    def join(self, timeout: float = 1.0) -> None:
        if self.use_scapy is True:
            self.scapy_monitor.join(timeout=timeout)
        super().join(timeout=timeout)

    def run(self):
        while not self.stop_monitor.is_set():
            sys_stats = self.psutil_monitor.read()
            power_stats = {"power/wattage": self.power_db.read_latest_power_metric()}

            all_res = {**sys_stats, **power_stats}

            if self.use_scapy:
                net_proc_stats = self.scapy_monitor.read_stats()
                all_res = {**all_res, **net_proc_stats}

            self.logger.log_metrics(all_res)
            time.sleep(1 / self.monitoring_freq)
        return


class SystemStats(object):

    def __init__(self):
        super().__init__()

        self.disk_read_sys_mb, self.disk_write_sys_mb = 0, 0
        self.net_sent_sys_mb, self.net_recv_sys_mb = 0, 0
        self.bandwidth_snapshot_time_s = 0
        self.create_bandwidth_snapshot()
        self.scapy_state = {}

    @staticmethod
    def get_static_sys_info():
        return {
            "cpu/logical_core_count": psutil.cpu_count(logical=True),
            "memory/total_memory_sys_mb": psutil.virtual_memory().total / 1024 ** 2
        }

    def read(self):
        """
        Get the current system and process info of the Python runtime.
        Bandwidths are calculate over the last interval since this method was called
        """
        cpu_info = self.get_cpu_info()
        memory_info = self.get_memory_info()
        proc_info = self.get_process_info()
        disk_info = self.get_disk_info()
        net_info = self.get_network_info()
        bandwidth_info = self.get_bandwidths(disk_info=disk_info, net_info=net_info)

        return {**cpu_info, **memory_info, **proc_info, **disk_info, **net_info, **bandwidth_info}

    def create_bandwidth_snapshot(self, disk_info=None, net_info=None):
        """
        Sets the disk and network counters + time to calculate the bandwidth on the next call of `get_bandwidths`
        """
        if disk_info is None:
            disk_info = self.get_disk_info()
        self.disk_read_sys_mb = disk_info["disk/disk_read_sys_mb"]
        self.disk_write_sys_mb = disk_info["disk/disk_write_sys_mb"]

        if net_info is None:
            net_info = self.get_network_info()
        self.net_sent_sys_mb = net_info["network/net_sent_sys_mb"]
        self.net_recv_sys_mb = net_info["network/net_recv_sys_mb"]
        self.bandwidth_snapshot_s = time.time()

    def get_bandwidths(self, disk_info, net_info):
        """
        Calculate the difference between the disk and network read/written bytes since the last call
        Populates the member variables that cached the last state + time
        """
        # todo: use a deque with size 2
        old_disk_read_sys_mb = self.disk_read_sys_mb
        old_disk_write_sys_mb = self.disk_write_sys_mb
        old_net_sent_sys_mb = self.net_sent_sys_mb
        old_net_recv_sys_mb = self.net_recv_sys_mb
        old_bandwidth_snapshot_s = self.bandwidth_snapshot_s

        self.create_bandwidth_snapshot()

        disk_read_sys_timeframe_mb = self.disk_read_sys_mb - old_disk_read_sys_mb
        disk_write_sys_timeframe_mb = self.disk_write_sys_mb - old_disk_write_sys_mb
        net_sent_sys_timeframe_mb = self.net_sent_sys_mb - old_net_sent_sys_mb
        net_recv_sys_timeframe_mb = self.net_recv_sys_mb - old_net_recv_sys_mb
        time_diff_s = self.bandwidth_snapshot_s - old_bandwidth_snapshot_s

        disk_read_sys_bandwidth_mbs = disk_read_sys_timeframe_mb / time_diff_s
        disk_write_sys_bandwidth_mbs = disk_write_sys_timeframe_mb / time_diff_s
        net_sent_sys_bandwidth_mbs = net_sent_sys_timeframe_mb / time_diff_s
        net_recv_sys_bandwidth_mbs = net_recv_sys_timeframe_mb / time_diff_s

        return {
            "bandwidth/disk_read_sys_bandwidth_mbs": disk_read_sys_bandwidth_mbs,
            "bandwidth/disk_write_sys_bandwidth_mbs": disk_write_sys_bandwidth_mbs,
            "bandwidth/net_sent_sys_bandwidth_mbs": net_sent_sys_bandwidth_mbs,
            "bandwidth/net_recv_sys_bandwidth_mbs": net_recv_sys_bandwidth_mbs
        }

    @staticmethod
    def get_cpu_info():
        # hyperthreaded cores included
        # type: int
        logical_core_count = psutil.cpu_count(logical=True)

        # global cpu stats, ever-increasing from boot
        # type: (int, int, int, int)
        cpu_stats = psutil.cpu_stats()

        # average system load over 1, 5 and 15 minutes summarized over all cores in percent
        # type: (float, float, float)
        one_min, five_min, fifteen_min = psutil.getloadavg()
        avg_sys_load_one_min_percent = one_min / logical_core_count * 100
        avg_sys_load_five_min_percent = five_min / logical_core_count * 100
        avg_sys_load_fifteen_min_percent = fifteen_min / logical_core_count * 100

        return {
            "cpu/interrupts/global_ctx_switches_count": cpu_stats.ctx_switches,
            "cpu/interrupts/global_interrupts_count": cpu_stats.interrupts,
            "cpu/interrupts/global_soft_interrupts_count": cpu_stats.soft_interrupts,
            "cpu/load/avg_sys_load_one_min_percent": avg_sys_load_one_min_percent,
            "cpu/load/avg_sys_load_five_min_percent": avg_sys_load_five_min_percent,
            "cpu/load/avg_sys_load_fifteen_min_percent": avg_sys_load_fifteen_min_percent
        }

    @staticmethod
    def get_memory_info():

        # global memory information
        # type (int): total_b - total memory on the system in bytes
        # type (int): available_b - available memory on the system in bytes
        # type (float): used_percent - total / used_b
        # type (int): used_b - used memory on the system in bytes (may not match "total - available" or "total - free")
        mem_stats = psutil.virtual_memory()

        total_memory_sys_mb = mem_stats.total / 1024 ** 2
        available_memory_sys_mb = mem_stats.available / 1024 ** 2
        used_memory_sys_mb = mem_stats.used / 1024 ** 2

        return {
            "memory/total_memory_sys_mb": total_memory_sys_mb,
            "memory/available_memory_sys_mb": available_memory_sys_mb,
            "memory/used_memory_sys_mb": used_memory_sys_mb,
            "memory/used_memory_sys_percent": used_memory_sys_mb
        }

    @staticmethod
    def get_process_info():

        # gets its own pid by default
        proc = psutil.Process()

        # voluntary and involunatry context switches by the process (cumulative)
        # type: (int, int)
        voluntary_proc_ctx_switches, involuntary_proc_ctx_switches = proc.num_ctx_switches()

        # memory information
        # type (int): rrs_b - resident set size: non-swappable physical memory used in bytes
        # type (int): vms_b - virtual memory size: total amount of virtual memory used in bytes
        # type (int): shared_b - shared memory size in bytes
        # type (int): trs_b - text resident set: memory devoted to executable code in bytes
        # type (int): drs_b - data resident set: physical memory devoted to other than code in bytes
        # type (int): lib_b - library memory: memory used by shared libraries in bytes
        # type (int): dirty_pages_count - number of dirty pages
        mem_info = proc.memory_info()

        # Memory info available on all platforms. See: psutil.readthedocs.io/en/latest/index.html?highlight=memory_info
        resident_set_size_proc_mb = mem_info.rss / 1024 ** 2
        virtual_memory_size_proc_mb = mem_info.vms / 1024 ** 2

        memory_dict = {
            "process/voluntary_proc_ctx_switches": voluntary_proc_ctx_switches,
            "process/involuntary_proc_ctx_switches": involuntary_proc_ctx_switches,
            "process/memory/resident_set_size_proc_mb": resident_set_size_proc_mb,
            "process/memory/virtual_memory_size_proc_mb": virtual_memory_size_proc_mb
        }

        try:
            # Linux attributes
            memory_dict["process/memory/shared_memory_proc_mb"] = mem_info.shared / 1024 ** 2
            memory_dict["process/memory/text_resident_set_proc_mb"] = mem_info.text / 1024 ** 2
            memory_dict["process/memory/data_resident_set_proc_mb"] = mem_info.data / 1024 ** 2
            memory_dict["process/memory/lib_memory_proc_mb"] = mem_info.lib / 1024 ** 2
        except AttributeError:
            pass

        return memory_dict

    @staticmethod
    def get_disk_info():

        # system disk stats
        # type (int): disk_read_sys_count - how often were reads performed
        # type (int): disk_write_sys_count - how often were writes performed
        # type (int): disk_read_sys_bytes - how much was read in bytes
        # type (int): writen_sys_bytes - how much was written in bytes
        # type (int): disk_read_time_sys_ms - how much time was used to read in milliseconds
        # type (int): disk_write_time_sys_ms - how much time was used to write in milliseconds
        # type (int): busy_time_sys_ms - how much time was used for actual I/O
        disk_info = psutil.disk_io_counters()

        disk_read_sys_mb = disk_info.read_bytes / 1024 ** 2
        disk_write_sys_mb = disk_info.write_bytes / 1024 ** 2
        disk_read_time_sys_s = disk_info.read_time / 1000
        disk_write_time_sys_s = disk_info.write_time / 1000

        disk_info_dict = {
            "disk/counter/disk_read_sys_count": disk_info.read_count,
            "disk/counter/disk_write_sys_count": disk_info.write_count,
            "disk/disk_read_sys_mb": disk_read_sys_mb,
            "disk/disk_write_sys_mb": disk_write_sys_mb,
            "disk/time/disk_read_time_sys_s": disk_read_time_sys_s,
            "disk/time/disk_write_time_sys_s": disk_write_time_sys_s
            # , "disk_busy_time_sys_s": disk_busy_time_sys_s
        }

        try:
            disk_info_dict["disk/time/disk_busy_time_sys_s"] = disk_info.busy_time / 1000  # returns seconds
        except AttributeError:
            pass

        return disk_info_dict

    @staticmethod
    def get_network_info():

        # network system stats
        # type (int): net_sent_sys_bytes - sent bytes over all network interfaces
        # type (int): net_recv_sys_bytes - received bytes over all network interfaces
        net_info = psutil.net_io_counters(pernic=False)

        return {
            "network/net_sent_sys_mb": net_info.bytes_sent / 1024 ** 2,
            "network/net_recv_sys_mb": net_info.bytes_recv / 1024 ** 2
        }


class ScapyNetworkMonitor(Thread):

    def __init__(self, ip_range: list = None) -> None:
        super().__init__()
        self.ip_range = ip_range
        self.daemon = True
        self.stop_trigger = False
        self.stop_sniffer = Event()

        self.my_ip, self.eth_iface = self._get_iface_adapter().values()

        self.traffic_map = {
            "inbound_total": 0,
            "outbound_total": 0,
            "inbound_bw": 0.0,
            "outbound_bw": 0.0,
            "last_measurement_ts": 0.0
        }

        self.intra_second_traffic_map = {
            "inbound": 0,
            "outbound": 0
        }

        self.socket = None
        # We start the network monitoring socket right when we spin up the Scapy Service.
        self.start_socket()

    def read_stats(self):

        return {
            "network/net_proc_inbound": self.traffic_map["inbound_total"],
            "network/net_proc_outbound": self.traffic_map["outbound_total"],
            "network/net_proc_bw_inbound": self.traffic_map["inbound_bw"],
            "network/net_proc_bw_outbound": self.traffic_map["outbound_bw"],
        }

    def start_socket(self):
        """
        We start the Scapy socket outside the sniff function, so we can control its life cycle.
        """
        self.socket = conf.L2listen(
            type=ETH_P_ALL,
            iface=self.eth_iface,
            filter="ip"
        )

    def run(self):
        """
        Let's start sniffing packages. We use the "prn" callback to read package lengths
        (and with that the traffic being generated).
        """

        sniff(
            opened_socket=self.socket,
            prn=lambda pkg: self._query(pkg=pkg),
            filter="ip",
            stop_filter=self.should_stop_sniffer
        )

    def join(self, timeout=None):
        """
        This method stops the network monitoring service.
        """
        self.stop_sniffer.set()
        super().join(timeout)
        self.socket.close()

    def should_stop_sniffer(self, packet):
        """
        This method generates the stop signal for the sniffing thread and the scapy socket.
        """
        self.stop_trigger = True
        return self.stop_sniffer.isSet()

    def _query(self, pkg):
        """
        With this method we read the traffic between the node and the number of target IP addresses we are interested
        in.
        """
        ip_layer = pkg.getlayer("IP")

        if ip_layer:
            if self.ip_range is not None and ip_layer.dst not in self.ip_range and ip_layer.src not in self.ip_range:
                pass
            else:
                pkg_dict = self._process_package(pkg=pkg, my_ip=self.my_ip)
                self._generate_measurements(pkg_dict)

    @staticmethod
    def _process_package(pkg, my_ip):
        """
        We process a network package and extract the direction (send/recv), package size, and peer ip.
        Please note this works only on packages that have an IP layer.
        """
        if not pkg.getlayer("IP"):
            raise NotImplementedError("This method only works with IP packages.")

        src = pkg.getlayer("IP").src
        dst = pkg.getlayer("IP").dst

        if src == my_ip:
            pkg_dict = {"direction": "outbound", "size": len(pkg), "peer_ip": dst}
        elif dst == my_ip:
            pkg_dict = {"direction": "inbound", "size": len(pkg), "peer_ip": src}
        else:
            pkg_dict = None

        return pkg_dict

    def _generate_measurements(self, pkg_dict: dict) -> None:
        """
        We now calculate the bandwidth utilization and update traffic metrics.
        The total traffic is measured in bytes.
        We calculate bandwidth in bit/s.
        """
        if pkg_dict is None:
            # We ignore empty pkg_dicts.
            return

        direction = pkg_dict["direction"]
        self.intra_second_traffic_map[direction] += pkg_dict["size"]
        ts_now = datetime.datetime.now().timestamp()
        if ts_now >= self.traffic_map["last_measurement_ts"] + 1:
            self.traffic_map["inbound_total"] += self.intra_second_traffic_map["inbound"]
            self.traffic_map["outbound_total"] += self.intra_second_traffic_map["outbound"]
            self.traffic_map["inbound_bw"] = self.intra_second_traffic_map["inbound"] * 8  # conversion to bit
            self.traffic_map["outbound_bw"] = self.intra_second_traffic_map["outbound"] * 8  # conversion to bit

            self.intra_second_traffic_map["inbound"] = 0
            self.intra_second_traffic_map["outbound"] = 0

            self.traffic_map["last_measurement_ts"] = datetime.datetime.now().timestamp()

    @staticmethod
    def _get_iface_adapter():
        """
        Helper function that returns the true IP address.
        :return: ip address (str)
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ext_ip_addr = s.getsockname()[0]

        adapter_name = None
        for adapter, info in psutil.net_if_addrs().items():
            if info[0].address == ext_ip_addr:
                adapter_name = adapter

        if adapter_name is None:
            adapter_name = "eth0"

        return {"ip_addr": ext_ip_addr, "iface": adapter_name}


if __name__ == "__main__":
    mon = HWMonitor(monitoring_freq=1.0, stop_event=Event(), use_scapy=False)
    mon.start()

    try:
        time.sleep(10)
        mon.stop_monitor.set()
    except KeyboardInterrupt:
        mon.join()
