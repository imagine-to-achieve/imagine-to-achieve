# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import ipaddress
import warnings
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class UR5HWInfo(HardwareInfo):
    """Hardware information for a UR5 robotic system."""

    config: "UR5Config"


@Hardware.register()
class UR5Robot(Hardware):
    """Hardware policy for UR5 robotic systems."""

    HW_TYPE = "UR5"
    ROBOT_PING_COUNT: int = 2
    ROBOT_PING_TIMEOUT: int = 1

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["UR5Config"]] = None
    ) -> Optional[HardwareResource]:
        assert configs is not None, (
            "Robot hardware requires explicit configurations for robot IP and camera serials."
        )
        robot_configs: list["UR5Config"] = []
        for config in configs:
            if isinstance(config, UR5Config) and config.node_rank == node_rank:
                robot_configs.append(config)

        if not robot_configs:
            return None

        ur5_infos = []
        cameras = cls.enumerate_cameras()
        for config in robot_configs:
            if config.camera_serials is None:
                config.camera_serials = list(cameras)

            ur5_infos.append(
                UR5HWInfo(
                    type=cls.HW_TYPE,
                    model=cls.HW_TYPE,
                    config=config,
                )
            )

            if config.disable_validate:
                continue

            try:
                from icmplib import ping
            except ImportError as exc:
                raise ImportError(
                    "icmplib is required for UR5 robot IP connectivity check."
                ) from exc

            try:
                response = ping(
                    config.robot_ip,
                    count=cls.ROBOT_PING_COUNT,
                    timeout=cls.ROBOT_PING_TIMEOUT,
                )
                if not response.is_alive:
                    raise ConnectionError
            except ConnectionError as exc:
                raise ConnectionError(
                    f"Cannot reach UR5 robot at IP {config.robot_ip} from node rank {node_rank}."
                ) from exc
            except PermissionError as exc:
                warnings.warn(
                    f"Permission denied when pinging UR5 robot at IP {config.robot_ip}. "
                    f"Ignoring the ping test. Error: {exc}"
                )
            except Exception as exc:
                warnings.warn(
                    f"Unexpected error while pinging UR5 robot at IP {config.robot_ip}. "
                    f"Ignoring the ping test. Error: {exc}"
                )

            if config.camera_serials:
                try:
                    importlib.import_module("pyrealsense2")
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "pyrealsense2 is required for UR5 camera serial checks."
                    ) from exc

                if not cameras:
                    raise ValueError(
                        f"No cameras are connected to node rank {node_rank} while UR5 requires at least one camera."
                    )
                for serial in config.camera_serials:
                    if serial not in cameras:
                        raise ValueError(
                            f"Camera with serial {serial} is not connected to node rank {node_rank}. "
                            f"Available cameras are: {cameras}."
                        )

        return HardwareResource(type=cls.HW_TYPE, infos=ur5_infos)

    @classmethod
    def enumerate_cameras(cls):
        cameras: set[str] = set()
        try:
            import pyrealsense2 as rs
        except ImportError:
            return cameras

        for device in rs.context().devices:
            cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras


@NodeHardwareConfig.register_hardware_config(UR5Robot.HW_TYPE)
@dataclass
class UR5Config(HardwareConfig):
    """Configuration for a UR5 robotic system."""

    robot_ip: str
    camera_serials: Optional[list[str]] = None
    disable_validate: bool = False

    def __post_init__(self):
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in UR5 config must be an integer. But got {type(self.node_rank)}."
        )
        try:
            ipaddress.ip_address(self.robot_ip)
        except ValueError as exc:
            raise ValueError(
                f"'robot_ip' in UR5 config must be a valid IP address. But got {self.robot_ip}."
            ) from exc

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
