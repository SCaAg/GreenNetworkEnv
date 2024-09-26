from uuid import UUID, uuid1
from abc import ABC, abstractmethod
from random import uniform
from typing import Tuple, List, Set
from gymnasium import spaces
from shapely.geometry import Point
from numpy.random import normal
import numpy as np
from pycraf.pathprof.imt import imt_urban_macro_losses, imt_urban_micro_losses
from gymnasium import Env
import gymnasium as gym
from astropy import units as u
import astropy.units as apu
TIME_SLOT = 10  # in second
SOLAR = [0, 0, 0, 0, 0, 0, 0.0582, 0.8613, 2.308, 3.2245, 3.8713, 4.3433, 4.4567, 4.0858,
         3.3962, 2.4, 0.9011, 0.7908, 0.0613, 0, 0, 0, 0, 0]  # 24小时中,每个小时太阳能功率的数据 in kW
CHOSEN_HOUR = 1
MAP_WIDTH = 2000  # in m
MAP_HEIGHT = 2000  # in m
GRID_PRICE = 1  # in $/kWh


gym.register(
            id='CommunicationEnv-v0',
            entry_point='test:CommunicationEnv',
        )


class BaseStation(ABC):
    """
    基站抽象基类,定义了基站的基本属性和方法。

    属性:
    - bs_uuid: 基站的唯一标识符。
    - position: 基站的位置,单位为米 (m)。
    - bandwidth: 基站的带宽,单位为赫兹 (Hz)。
    - frequency: 基站的频率,单位为吉赫 (GHz)。
    - trans_power: 基站的传输功率,单位为分贝毫瓦 (dBm)。
    - height: 基站的高度,单位为米 (m)。
    - const_power: 基站的静态功率,单位为千瓦 (kW)。
    - connected_ues: 当前连接到基站的用户设备集合。

    方法:
    - point: 返回基站位置的点表示。
    - consume_energy: 计算基站在资源分享分配前消耗的总能量。
    - reset: 重置基站的状态,需要在子类中实现。
    - energy_info: 获取基站的能量信息,需要在子类中实现。
    - get_energy: 基站从外部获得能量,需要在子类中实现。
    """

    def __init__(self,
                 position: Tuple[float, float],
                 bandwidth: float,
                 frequency: float,
                 trans_power: float,
                 height: float,
                 const_power: float
                 ):
        """
        初始化基站对象。

        参数:
        - position: 基站的位置,单位为米 (m)。
        - bandwidth: 基站的带宽,单位为赫兹 (Hz)。
        - frequency: 基站的频率,单位为吉赫 (GHz)。
        - trans_power: 基站的传输功率,单位为分贝毫瓦 (dBm)。
        - height: 基站的高度,单位为米 (m)。
        - const_power: 基站的静态功率,单位为千瓦 (kW)。
        """
        self.bs_uuid: UUID = uuid1()
        self.position: Tuple[float, float] = position
        """in (m, m)"""
        self.bandwidth: float = bandwidth
        """in Hz"""
        self.frequency: float = frequency
        """in GHz"""
        self.trans_power: float = trans_power
        """in dBm"""
        self.height: float = height
        """in m"""
        self.const_power: float = const_power
        """in kW"""
        self.connected_ues: set[UserEquipment] = set()
        """connected user equipments"""

    @property
    def point(self):
        """
        返回基站位置的点表示。

        返回:
        - 基站位置的点表示。
        """
        return Point(self.position)

    @abstractmethod
    def __str__(self):
        """
        返回基站的字符串表示,需要在子类中实现。
        """
        pass

    @property
    def consume_energy(self):
        """
        计算基站在资源分享分配前消耗的总能量(单位:kWh)。

        公式:
        总能量 = ((静态功率 + 每用户传输功率 * 连接用户数量) * 时间片) / 3600

        具体步骤:
        1. 将传输功率从 dBm 转换为瓦 (kW)。
        2. 计算基站在指定时间片内的总能量消耗(kWh)。

        参数:
        - const_power: 基站的静态功率,单位为千瓦 (kW)。
        - trans_power: 每用户的传输功率,单位为 dBm,将其转换为千瓦 (kW)。
        - TIME_SLOT: 时间片,单位为秒 (s)。
        - ue_nums: 当前连接的用户设备数量。

        返回:
        - 基站在指定时间片内的总能量消耗,单位为瓦时 (kWh)。
        """
        ue_nums = len(self.connected_ues)
        trans_power_kw = 10 ** ((self.trans_power - 30) / 10) / 1000
        consumption = (self.const_power + trans_power_kw * ue_nums) * TIME_SLOT / 3600
        return consumption

    @abstractmethod
    def reset(self):
        """
        重置基站的状态,需要在子类中实现。
        """
        pass

    @abstractmethod
    def energy_info(self):
        """
        获取基站的能量信息,需要在子类中实现。
        """
        pass

    @abstractmethod
    def get_energy(self, energy: float):
        """
        基站获得额外能量,需要在子类中实现。

        参数:
        - energy: 基站声明的能量,单位为千瓦时 (kWh)。
        """
        pass


class MacroBaseStation(BaseStation):
    """
    宏基站类,继承自基站抽象基类 BaseStation。

    方法:
    - __init__: 初始化宏基站对象。
    - energy_info: 获取宏基站的能量信息,此方法在当前类中未实现。
    - __str__: 返回宏基站的字符串表示,当前返回的是宏基站消耗的能量。
    - get_energy: 宏基站额外获得能量。
    - reset: 重置宏基站的状态。
    """

    def __init__(self,
                 position: Tuple[float, float],
                 bandwidth: float,
                 frequency: float,
                 trans_power: float,
                 height: float,
                 const_power: float):
        """
        初始化宏基站对象。

        参数:
        - position: 宏基站的位置,单位为米 (m)。
        - bandwidth: 宏基站的带宽,单位为赫兹 (Hz)。
        - frequency: 宏基站的频率,单位为吉赫 (GHz)。
        - trans_power: 宏基站的传输功率,单位为分贝毫瓦 (dBm)。
        - height: 宏基站的高度,单位为米 (m)。
        - const_power: 宏基站的静态功率,单位为千瓦 (kW)。
        """
        super().__init__(
            position,
            bandwidth,
            frequency,
            trans_power,
            height,
            const_power)

    def energy_info(self):
        """
        获取宏基站的能量信息,此方法在当前类中未实现。
        """
        pass

    def __str__(self):
        """
        返回宏基站的字符串表示,当前返回的是宏基站消耗的能量。
        """
        return f"MacroBaseStation(position={self.position}, bandwidth={self.bandwidth}, frequency={self.frequency}, trans_power={self.trans_power}, height={self.height}, const_power={self.const_power})"

    def get_energy(self, energy: float)->float:
        """
        宏基站获得额外能量。

        参数:
        - energy: 微基站额外的能量，可能为负数，代表新能源不够用，需要从电网购买,单位为千瓦时 (kWh)。
        """
        return (self.consume_energy - energy)

    def reset(self):
        """
        重置宏基站的状态,当前的实现是将声明的能量重置为0。
        """
        pass


class MicroBaseStation(BaseStation):
    """
    微型基站类,继承自基站抽象基类 BaseStation。

    属性:
    - active: 表示微型基站是否处于活动状态。
    - battery_level: 微型基站的电池电量,单位为百分比。
    - needed_energy: 表示微型基站是否需要能量。

    方法:
    - __init__: 初始化微型基站对象。
    - energy_info: 获取微型基站的能量信息。
    - __str__: 返回微型基站的字符串表示,此方法在当前类中未实现。
    - get_energy: 微型基站获得额外能量。
    - reset: 重置微型基站的状态。
    - consume_energy: 计算微型基站在资源分享分配前消耗的总能量。
    """

    def __init__(self,
                 position: Tuple[float, float],
                 bandwidth: float,
                 frequency: float,
                 trans_power: float,
                 height: float,
                 const_power: float,
                 battery_capacity: float=100):
        """
        初始化微型基站对象。

        参数:
        - position: 微型基站的位置,单位为米 (m)。
        - bandwidth: 微型基站的带宽,单位为赫兹 (Hz)。
        - frequency: 微型基站的频率,单位为吉赫 (GHz)。
        - trans_power: 微型基站的传输功率,单位为分贝毫瓦 (dBm)。
        - height: 微型基站的高度,单位为米 (m)。
        - const_power: 微型基站的静态功率,单位为千瓦 (kW)。
        - battery_capacity: 微型基站的电池容量,单位为千瓦时 (kWh)。
        """
        super().__init__(position,
                         bandwidth,
                         frequency,
                         trans_power,
                         height,
                         const_power)
        self.active: bool = True
        self.battery_level: float = 0
        """in percentage"""
        self.needed_energy: float = 0
        self.battery_capacity: float = battery_capacity

    def energy_info(self) -> Tuple[float, float, float]:
        """
        获取微型基站的能量信息。

        返回:
        - extra_energy: 额外的能量,单位为瓦时 (kWh)。
        - battery_level: 微型基站的电池电量,单位为百分比。
        - needed_energy: 表示微型基站是否需要能量。
        """
        mu = SOLAR[CHOSEN_HOUR]
        sigma = mu * 1.5
        renewable_energy = max(normal(mu, sigma), 0) * TIME_SLOT / 3600
        battery_energy = self.battery_level * self.battery_capacity
        left_energy = battery_energy + renewable_energy - self.consume_energy
        extra_energy = left_energy - self.battery_capacity
        if extra_energy >= 0:
            self.battery_level = 1
            self.needed_energy = 0
        else:
            extra_energy = 0
            if left_energy >= 0:
                self.battery_level = left_energy / self.battery_capacity
                self.needed_energy = 0
            else:
                self.battery_level = 0
                self.needed_energy = left_energy
        return extra_energy, self.battery_level, self.needed_energy

    def __str__(self):
        """
        返回微型基站的字符串表示,此方法在当前类中未实现。
        """
        return f"MicroBaseStation(position={self.position}, bandwidth={self.bandwidth}, frequency={self.frequency}, trans_power={self.trans_power}, height={self.height}, const_power={self.const_power}, battery_level={self.battery_level})"

    def get_energy(self, energy: float):
        """
        微型基站获得额外能量。

        参数:
        - energy: 微型基站获得的能量,单位为千瓦时 (kWh)。

        返回:
        - extra_energy: 如果电池电量超过100%,返回超出的能量,单位为千瓦时 (kWh)。
        """
        self.needed_energy += energy
        if self.needed_energy > 0:
            self.battery_level = self.battery_level + self.needed_energy / self.battery_capacity
            if self.battery_level > 1:
                extra_energy = (self.battery_level - 1) * self.battery_capacity
                self.battery_level = 1
            else:
                extra_energy = 0
        else:
            extra_energy = self.needed_energy
        return extra_energy

    def reset(self):
        """
        重置微型基站的状态,当前的实现是将电池电量随机设置为30%到100%之间,将需要能量的标志设置为False,将获得的能量设置为0,将活动状态设置为True。
        """
        self.battery_level = np.random.uniform(0.3, 1.0)
        self.needed_energy = 0
        self.active = True

    @property
    def consume_energy(self):
        """
        计算微型基站在资源分享分配前消耗的总能量(单位:kWh)。

        公式:
        总能量 = ((静态功率 + 每用户传输功率 * 连接用户数量) * 时间片) / 3600

        具体步骤:
        1. 将传输功率从 dBm 转换为瓦 (W)。
        2. 计算基站在指定时间片内的总能量消耗(Wh)。

        参数:
        - const_power: 基站的静态功率,单位为千瓦 (kW)。
        - trans_power: 每用户的传输功率,单位为 dBm,将其转换为千瓦 (kW)。
        - TIME_SLOT: 时间片,单位为秒 (s)。
        - ue_nums: 当前连接的用户设备数量。

        返回:
        - 基站在指定时间片内的总能量消耗,单位为瓦时 (Wh)。
        """
        if self.active:
            return super().consume_energy
        return 0


class CommunicationEnv(Env):
    metadata = {'render_modes': ['None']}

    def __init__(self, render_mode=None):
        self.current_step = 0
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        stations = [
            MacroBaseStation((1000, 1000), bandwidth=100, frequency=3.5, trans_power=46, height=30, const_power=500),
        ]
        # Place micro base stations at random positions
        for _ in range(6):
            position = (uniform(0, MAP_WIDTH), uniform(0, MAP_HEIGHT))
            stations.append(
                MicroBaseStation(position, bandwidth=100, frequency=3.5, trans_power=30, height=10, const_power=50)
            )
        ues = [
            UserEquipment(speed=1, snr_threshold=0, noise=-90, height=1.5)
            for _ in range(100)
        ]
        self.micro_stations = [station for station in stations if isinstance(station, MicroBaseStation)]
        self.macro_station = next(station for station in stations if isinstance(station, MacroBaseStation))
        self.observation_space = spaces.Dict(
            {
                "consumed_energy": spaces.Box(low=0, high=np.inf, shape=(len(self.micro_stations),), dtype=np.float32),
                "needed_energy": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.micro_stations),), dtype=np.float32),
                "battery_level": spaces.Box(low=0.0, high=1.0, shape=(len(self.micro_stations),), dtype=np.float32),
                "sbs_status": spaces.MultiBinary(len(self.micro_stations)),
            }
        )
        self.action_space = spaces.Dict(
            {
                "energy_allocation": spaces.MultiBinary(len(self.micro_stations)),
                "sbs_status": spaces.MultiBinary(len(self.micro_stations)),
            }
        )

        self.stations: dict[UUID, BaseStation] = {station.bs_uuid: station for station in stations}
        self.ues: dict[UUID, UserEquipment] = {ue.ue_uuid: ue for ue in ues}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # np.random.seed(seed)
        # Reset the positions of UEs
        self.current_step = 0

        for ue in self.ues.values():
            ue.reset()

        # Reset the micro base stations
        for station in self.stations.values():
            station.reset()

        # Return the initial observation
        return self._get_obs()

    def step(self, action):
        # 1. 处理 action
        energy_allocation = action['energy_allocation']
        new_sbs_status = action['sbs_status']

        last_sbs_status = [
            station.active
            for station in self.micro_stations
        ]

        # 2. 计算切换惩罚
        switch_penalty = sum(old != new for old, new in zip(last_sbs_status, new_sbs_status))

        # 3. 更新基站状态
        for i, station in enumerate(self.micro_stations):
            station.active = bool(new_sbs_status[i])

        # 4. 分配能源和计算额外能源
        total_extra_energy = 0
        needed_energy_stations: list[MicroBaseStation] = []
        for i, station in enumerate(self.micro_stations):
                extra_energy, _, needed_energy = station.energy_info()
                total_extra_energy += extra_energy
                if (needed_energy < 0):
                    needed_energy_stations.append(station)

        # 5. 将多余的能源分配给需要能源的基站
        if needed_energy_stations:
            # 这段代码处理多余能源的分配逻辑
            total_extra_energy_original = total_extra_energy
            # 计算所有需要能源的基站的总需求量
            total_needed_energy = -sum(station.needed_energy for station in needed_energy_stations)
            # 遍历每个需要能源的基站
            for station in needed_energy_stations:
                # 按照基站需求的比例分配能源
                # 分配的能源 = min(按比例分配的能源, 基站实际需要的能源)
                energy_to_allocate = min(
                    (-station.needed_energy) / total_needed_energy * total_extra_energy_original,
                    -station.needed_energy
                )
                
                # 将分配的能源给到基站,并获取剩余未使用的能源
                remaining_extra_energy = station.get_energy(energy_to_allocate)
                
                # 更新总的多余能源,减去已分配的能源,加上未使用的能源
                total_extra_energy -= (energy_to_allocate - remaining_extra_energy)
                


        # 6. 如果还有剩余能源,分配给宏基站
        total_energy_consumption=self.macro_station.get_energy(total_extra_energy)

        # 7. 清除所有基站的连接用户
        for station in self.stations.values():
            station.connected_ues.clear()

        # 8. 移动用户
        for ue in self.ues.values():
            ue.move()
            # Check boundaries and reflect direction if necessary
            if ue.position[0] < 0 or ue.position[0] > MAP_WIDTH:
                ue.position[0] = np.clip(ue.position[0], 0, MAP_WIDTH)
                ue.direction = np.pi - ue.direction  # Reflect over y-axis
            if ue.position[1] < 0 or ue.position[1] > MAP_HEIGHT:
                ue.position[1] = np.clip(ue.position[1], 0, MAP_HEIGHT)
                ue.direction = -ue.direction  # Reflect over x-axis

        # 9. 重新计算用户连接
        # 这段代码用于重新计算用户连接
        for ue in self.ues.values():
            # 获取用户设备可用的基站列表
            available_stations = ue.available_connections(self)
            
            if available_stations:
                # 找到距离用户设备最近的基站
                # 使用lambda函数计算每个基站到用户设备的距离，然后选择距离最小的基站
                closest_station = min(available_stations, key=lambda bs: ue.point.distance(bs.point))
                
                # 将用户设备添加到最近基站的连接列表中
                closest_station.connected_ues.add(ue)
            else:
                # 如果没有可用的基站，可以记录日志或采取其他适当的措施
                print(f"警告：用户设备 {ue.ue_uuid} 没有可用的基站连接")




        # 10. 计算奖励
        reward = -total_energy_consumption - switch_penalty

        # 11. 更新观察
        obs = self._get_obs()

        # 12. 检查是否结束(这里假设模拟时间为 1 小时)
        done = self.current_step >= 3600 // TIME_SLOT

        # 13. 准备信息字典
        info = {
            'total_energy_consumption': total_energy_consumption,
            'switch_penalty': switch_penalty,
            'macro_energy_saved': total_extra_energy
        }

        # 14. 更新当前步数
        self.current_step += 1

        return obs, reward, done, info

    def _get_obs(self):
        consumed_energy = [
            station.consume_energy
            for station in self.micro_stations
        ]
        
        needed_energy = [
            station.needed_energy
            for station in self.micro_stations
        ]

        battery_levels = [
            station.battery_level
            for station in self.micro_stations
        ]

        sbs_status = [
            station.active
            for station in self.micro_stations
        ]

        return {
            "consumed_energy": np.array(consumed_energy, dtype=np.float32),
            "needed_energy": np.array(needed_energy, dtype=np.float32),
            "battery_level": np.array(battery_levels, dtype=np.float32),
            "sbs_status": np.array(sbs_status, dtype=np.int32)
        }


class UserEquipment:
    def __init__(self,
                 speed: float,
                 snr_threshold: float,
                 noise: float,
                 height: float):
        self.ue_uuid: UUID = uuid1()
        self.speed: float = speed
        self.snr_threshold: float = snr_threshold
        """in dB    """
        self.noise: float = noise
        """in dBm"""
        self.height: float = height
        """in m"""
        self.position: np.ndarray = np.array([uniform(0, MAP_WIDTH), uniform(0, MAP_HEIGHT)])
        self.direction: float = uniform(0, 2 * np.pi)  # random initial direction

    def __str__(self):
        return f"UserEquipment(position={self.position}, speed={self.speed}, snr_threshold={self.snr_threshold}, noise={self.noise}, height={self.height})"
    
    @property
    def point(self):
        return Point(self.position)

    def move(self):
        dx = np.cos(self.direction) * self.speed * TIME_SLOT
        dy = np.sin(self.direction) * self.speed * TIME_SLOT
        self.position += np.array([dx, dy])
        self.direction += np.random.normal(0, np.pi / 8)  # small random change in direction
        self.direction %= 2 * np.pi  # keep direction within [0, 2*pi]

    def snr(self, bs: BaseStation):
        """Calculate SNR for transmission between BS and UE."""
        distance = bs.point.distance(self.point)
        if distance > 5000:
            return -np.inf
        elif distance < 10:
            return np.inf
        elif isinstance(bs, MicroBaseStation):
            if bs.active is True:
                power_loss = imt_urban_micro_losses
            else:
                return -np.inf  # No signal from inactive BS
        else:
            power_loss = imt_urban_macro_losses
        loss,_,_ = power_loss(freq=bs.frequency*u.GHz, dist_2d=[distance]*u.m,h_bs=bs.height*u.m, h_ue=self.height*u.m)
        loss = float(loss.to('dB').value)
        trans_power_mW = 10 ** (bs.trans_power / 10)
        loss_linear = 10 ** (-loss / 10)
        signal_power_mW = trans_power_mW * loss_linear
        noise_power_mW = 10 ** (self.noise / 10)
        snr_linear = signal_power_mW / noise_power_mW
        snr_dB = 10 * np.log10(snr_linear)
        return snr_dB
    
    def check_connectivity(self, bs: BaseStation) -> bool:
        """Connection can be established if SNR exceeds threshold of UE."""
        return self.snr(bs) > self.snr_threshold

    def available_connections(self, env: CommunicationEnv) -> Set:
        """Returns set of what base stations users could connect to."""
        stations = env.stations.values()
        return {bs for bs in stations if self.check_connectivity(bs)}

    def reset(self):
        self.position = np.array([uniform(0, MAP_WIDTH), uniform(0, MAP_HEIGHT)])
        self.direction = uniform(0, 2 * np.pi)

