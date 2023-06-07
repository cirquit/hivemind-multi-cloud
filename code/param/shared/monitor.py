import os
import time
import psutil


class Monitor:
    def __init__(self):

        # setting up t-1 values for bandwidth calculation
        self.disk_read_sys_mb, self.disk_write_sys_mb = 0, 0
        self.net_sent_sys_mbit, self.net_recv_sys_mbit = 0, 0
        self.bandwidth_snapshot_time_s = 0

        # setting up t=0 values for global system increasing values
        (
            self.global_ctx_switches_count_init,
            self.global_interrupts_count_init,
            self.global_soft_interrupts_count_init,
        ) = (0, 0, 0)
        self.disk_read_sys_count_init, self.disk_write_sys_count_init = 0, 0
        self.disk_read_time_sys_s_init, self.disk_write_time_sys_s_init = 0, 0
        self.disk_busy_time_sys_s_init = 0

        # populating the snapshot values
        self.create_bandwidth_snapshot()
        self.create_interrupt_snapshot()
        self.create_disk_access_snapshot()

    def get_static_info(self):
        logical_core_count = psutil.cpu_count(logical=True)
        total_b, _, _, _, _, _, _, _, _, _, _ = psutil.virtual_memory()
        total_memory_sys_MB = total_b / 1000**2

        return {
            "cpu/logical_core_count": logical_core_count,
            "memory/total_memory_sys_MB": total_memory_sys_MB,
        }

    def get_sys_info(self):
        """Get the current system and process info of the Python runtime.
        Bandwidths are calculated over the last interval since this method was called
        """
        cpu_info = self.get_cpu_info()
        memory_info = self.get_memory_info()
        proc_info = self.get_process_info()
        disk_info = self.get_disk_info()
        net_info = self.get_network_info()
        bandwidth_info = self.get_bandwidths(disk_info=disk_info, net_info=net_info)

        # remove the global counters
        del disk_info["disk/disk_read_sys_MB"]
        del disk_info["disk/disk_write_sys_MB"]
        del net_info["network/net_sent_sys_mbit"]
        del net_info["network/net_recv_sys_mbit"]

        return {
            **cpu_info,
            **memory_info,
            **proc_info,
            **disk_info,
            **net_info,
            **bandwidth_info,
        }

    def create_disk_access_snapshot(self):
        """Sets the disk counters to the initial value which is subtracted in `get_disk_info` to get a "per-run" count"""
        disk_info = self.get_disk_info()
        self.disk_read_sys_count_init = disk_info["disk/counter/disk_read_sys_count"]
        self.disk_write_sys_count_init = disk_info["disk/counter/disk_write_sys_count"]
        self.disk_read_time_sys_s_init = disk_info["disk/time/disk_read_time_sys_s"]
        self.disk_write_time_sys_s_init = disk_info["disk/time/disk_write_time_sys_s"]
        self.disk_busy_time_sys_s_init = disk_info["disk/time/disk_busy_time_sys_s"]

    def create_interrupt_snapshot(self):
        """Sets the interrupt counters to the initial value which is subtracted in `get_cpu_info` to get a "per-run" count"""
        cpu_info = self.get_cpu_info()
        self.global_ctx_switches_count_init = cpu_info["cpu/interrupts/ctx_switches_count"]
        self.global_interrupts_count_init = cpu_info["cpu/interrupts/interrupts_count"]
        self.global_soft_interrupts_count_init = cpu_info["cpu/interrupts/soft_interrupts_count"]

    def create_bandwidth_snapshot(self, disk_info=None, net_info=None):
        """Sets the disk and network counters + time to calculate the bandwidth on the next call of `get_bandwidths`"""
        if disk_info == None:
            disk_info = self.get_disk_info()
        self.disk_read_sys_mb = disk_info["disk/disk_read_sys_MB"]
        self.disk_write_sys_mb = disk_info["disk/disk_write_sys_MB"]

        if net_info == None:
            net_info = self.get_network_info()
        self.net_sent_sys_mbit = net_info["network/net_sent_sys_mbit"]
        self.net_recv_sys_mbit = net_info["network/net_recv_sys_mbit"]

        self.bandwidth_snapshot_s = time.time()

    def get_bandwidths(self, disk_info, net_info):
        """Calculate the difference between the disk and network read/written bytes since the last call
        Populates the member variables that cached the last state + time
        """
        # todo: use a deque with size 2
        old_disk_read_sys_mb = self.disk_read_sys_mb
        old_disk_write_sys_mb = self.disk_write_sys_mb
        old_net_sent_sys_mbit = self.net_sent_sys_mbit
        old_net_recv_sys_mbit = self.net_recv_sys_mbit
        old_bandwidth_snapshot_s = self.bandwidth_snapshot_s

        self.create_bandwidth_snapshot()

        disk_read_sys_timeframe_mb = self.disk_read_sys_mb - old_disk_read_sys_mb
        disk_write_sys_timeframe_mb = self.disk_write_sys_mb - old_disk_write_sys_mb
        net_sent_sys_timeframe_mbit = self.net_sent_sys_mbit - old_net_sent_sys_mbit
        net_recv_sys_timeframe_mbit = self.net_recv_sys_mbit - old_net_recv_sys_mbit
        time_diff_s = self.bandwidth_snapshot_s - old_bandwidth_snapshot_s

        disk_read_sys_bandwidth_mbs = disk_read_sys_timeframe_mb / time_diff_s
        disk_write_sys_bandwidth_mbs = disk_write_sys_timeframe_mb / time_diff_s
        net_sent_sys_bandwidth_mbs = net_sent_sys_timeframe_mbit / time_diff_s
        net_recv_sys_bandwidth_mbs = net_recv_sys_timeframe_mbit / time_diff_s

        return {
            "bandwidth/disk_read_sys_bandwidth_MBs": disk_read_sys_bandwidth_mbs,
            "bandwidth/disk_write_sys_bandwidth_MBs": disk_write_sys_bandwidth_mbs,
            "bandwidth/net_sent_sys_bandwidth_Mbits": net_sent_sys_bandwidth_mbs,
            "bandwidth/net_recv_sys_bandwidth_Mbits": net_recv_sys_bandwidth_mbs,
        }

    def get_cpu_info(self):
        # hyperthreaded cores included
        # type: int
        logical_core_count = psutil.cpu_count(logical=True)

        # global cpu stats, ever increasing from boot, we null them for easier comparison from each init of this class
        # type: (int, int, int, int)
        (
            global_ctx_switches_count,
            global_interrupts_count,
            global_soft_interrupts_count,
            _,
        ) = psutil.cpu_stats()
        ctx_switches_count = global_ctx_switches_count - self.global_ctx_switches_count_init
        interrupts_count = global_interrupts_count - self.global_interrupts_count_init
        soft_interrupts_count = global_soft_interrupts_count - self.global_soft_interrupts_count_init

        # average system load over 1, 5 and 15 minutes summarized over all cores in percent
        # type: (float, float, float)
        one_min, five_min, fifteen_min = psutil.getloadavg()
        avg_sys_load_one_min_percent = one_min / logical_core_count * 100
        avg_sys_load_five_min_percent = five_min / logical_core_count * 100
        avg_sys_load_fifteen_min_percent = fifteen_min / logical_core_count * 100

        return {
            "cpu/interrupts/ctx_switches_count": ctx_switches_count,
            "cpu/interrupts/interrupts_count": interrupts_count,
            "cpu/interrupts/soft_interrupts_count": soft_interrupts_count,
            "cpu/load/avg_sys_load_one_min_percent": avg_sys_load_one_min_percent,
            "cpu/load/avg_sys_load_five_min_percent": avg_sys_load_five_min_percent,
            "cpu/load/avg_sys_load_fifteen_min_percent": avg_sys_load_fifteen_min_percent,
        }

    @staticmethod
    def get_memory_info():
        # global memory information
        # type (int): total_b - total memory on the system in bytes
        # type (int): available_b - available memory on the system in bytes
        # type (float): used_percent - total / used_b
        # type (int): used_b - used memory on the system in bytes (may not match "total - available" or "total - free")
        (
            total_b,
            available_b,
            used_memory_sys_percent,
            used_b,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = psutil.virtual_memory()
        available_memory_sys_mb = available_b / 1000**2
        used_memory_sys_mb = used_b / 1000**2

        return {
            "memory/available_memory_sys_MB": available_memory_sys_mb,
            "memory/used_memory_sys_MB": used_memory_sys_mb,
            "memory/used_memory_sys_percent": used_memory_sys_percent,
        }

    @staticmethod
    def get_process_info():

        # gets its own pid by default
        proc = psutil.Process()

        # voluntary and involunatry context switches by the process (cumulative)
        # type: (int, int)
        (
            voluntary_proc_ctx_switches,
            involuntary_proc_ctx_switches,
        ) = proc.num_ctx_switches()

        # memory information
        # type (int): rrs_b - resident set size: non-swappable physical memory used in bytes
        # type (int): vms_b - virtual memory size: total amount of virtual memory used in bytes
        # type (int): shared_b - shared memory size in bytes
        # type (int): trs_b - text resident set: memory devoted to executable code in bytes
        # type (int): drs_b - data resident set: physical memory devoted to other than code in bytes
        # type (int): lib_b - library memory: memory used by shared libraries in bytes
        # type (int): dirty_pages_count - number of dirty pages
        (
            rss_b,
            vms_b,
            shared_b,
            trs_b,
            drs_b,
            lib_b,
            dirty_pages_proc_count,
        ) = proc.memory_info()
        resident_set_size_proc_mb = rss_b / 1000**2
        virtual_memory_size_proc_mb = vms_b / 1000**2
        shared_memory_proc_mb = shared_b / 1000**2
        text_resident_set_proc_mb = trs_b / 1000**2
        data_resident_set_proc_mb = drs_b / 1000**2
        lib_memory_proc_mb = lib_b / 1000**2

        return {
            "process/voluntary_proc_ctx_switches": voluntary_proc_ctx_switches,
            "process/involuntary_proc_ctx_switches": involuntary_proc_ctx_switches,
            "process/memory/resident_set_size_proc_MB": resident_set_size_proc_mb,
            "process/memory/virtual_memory_size_proc_MB": virtual_memory_size_proc_mb,
            "process/memory/shared_memory_proc_MB": shared_memory_proc_mb,
            "process/memory/text_resident_set_proc_MB": text_resident_set_proc_mb,
            "process/memory/data_resident_set_proc_MB": data_resident_set_proc_mb,
            "process/memory/lib_memory_proc_MB": lib_memory_proc_mb,
            "process/memory/dirty_pages_proc_count": dirty_pages_proc_count,
        }

    def get_disk_info(self):

        # system disk stats
        # type (int): disk_read_sys_count - how often were reads performed
        # type (int): disk_write_sys_count - how often were writes performed
        # type (int): disk_read_sys_bytes - how much was read in bytes
        # type (int): writen_sys_bytes - how much was written in bytes
        # type (int): disk_read_time_sys_ms - how much time was used to read in milliseconds
        # type (int): disk_write_time_sys_ms - how much time was used to write in milliseconds
        # type (int): busy_time_sys_ms - how much time was used for actual I/O

        (
            global_disk_read_sys_count,
            global_disk_write_sys_count,
            global_disk_read_sys_bytes,
            global_disk_write_sys_bytes,
            global_disk_read_time_sys_ms,
            global_disk_write_time_sys_ms,
            _,
            _,
            global_busy_time_sys_ms,
        ) = psutil.disk_io_counters()

        disk_read_sys_mb = global_disk_read_sys_bytes / 1000**2
        disk_write_sys_mb = global_disk_write_sys_bytes / 1000**2
        # subtracting global system start to get process values
        disk_read_sys_count = global_disk_read_sys_count - self.disk_read_sys_count_init
        disk_write_sys_count = global_disk_write_sys_count - self.disk_write_sys_count_init
        disk_read_time_sys_s = global_disk_read_time_sys_ms / 1000 - self.disk_read_time_sys_s_init
        disk_write_time_sys_s = global_disk_write_time_sys_ms / 1000 - self.disk_write_time_sys_s_init
        disk_busy_time_sys_s = global_busy_time_sys_ms / 1000 - self.disk_busy_time_sys_s_init

        return {
            "disk/counter/disk_read_sys_count": disk_read_sys_count,
            "disk/counter/disk_write_sys_count": disk_write_sys_count,
            "disk/disk_read_sys_MB": disk_read_sys_mb,
            "disk/disk_write_sys_MB": disk_write_sys_mb,
            "disk/time/disk_read_time_sys_s": disk_read_time_sys_s,
            "disk/time/disk_write_time_sys_s": disk_write_time_sys_s,
            "disk/time/disk_busy_time_sys_s": disk_busy_time_sys_s,
        }

    @staticmethod
    def get_network_info():

        # network system stats
        # type (int): net_sent_sys_bytes - sent bytes over all network interfaces
        # type (int): net_recv_sys_bytes - received bytes over all network interfaces
        (
            net_sent_sys_bytes,
            net_recv_sys_bytes,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = psutil.net_io_counters(pernic=False)
        net_sent_sys_mbit = (net_sent_sys_bytes / 1000**2) * 8
        net_recv_sys_mbit = (net_recv_sys_bytes / 1000**2) * 8

        return {
            "network/net_sent_sys_mbit": net_sent_sys_mbit,
            "network/net_recv_sys_mbit": net_recv_sys_mbit,
        }
