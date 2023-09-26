from typing import List
import sys
import torch

sys.path.append("src")
from rendering.z_buffer import ZBuffer
from abc import ABC, abstractmethod

class DepthRendering(ABC):

    @abstractmethod
    def get_images(self):
        pass


class ExteriorDepthRendering(DepthRendering):
    def __init__(self, zbufs: List[ZBuffer]):
        self._zbufs = zbufs

    @property
    def zbufs(self):
        return torch.stack([zbuf.zbuf for zbuf in self._zbufs])
    
    def get_images(self):
        zbufs = self.zbufs
        LARGE_POSITIVE_NUMBER = 619.0
        zbufs[zbufs == -1.0] = LARGE_POSITIVE_NUMBER
        images = zbufs.amin(dim=0)
        images[images == LARGE_POSITIVE_NUMBER] = -1.0
        images = [image for image in images]
        return images
    


class InteriorDepthRendering(DepthRendering):
    def __init__(self, robot_zbuf: ZBuffer, other_zbuf: ZBuffer):
        self._robot_zbuf = robot_zbuf
        self._other_zbuf = other_zbuf

    @property
    def robot_zbuf(self):
        return self._robot_zbuf.zbuf
    
    @property
    def other_zubf(self):
        return self._other_zbuf.zbuf


class InteriorGapRendering(InteriorDepthRendering):
    def get_images(self):
        gaps = []
        robot_zbuf = self.robot_zbuf
        other_zbuf = self.other_zubf
        for rz, oz in zip(robot_zbuf, other_zbuf): #looping over views
            distance = rz - oz
            distance[distance < 0] = 0
            gaps.append(distance)
        return gaps


class InteriorContactRendering(InteriorDepthRendering):
    def get_images(self):
        contacts = []
        robot_zbuf = self.robot_zbuf
        other_zbuf = self.other_zubf
        for rz, oz in zip(robot_zbuf, other_zbuf): # looping over views
            distance = oz - rz
            mask_contact = distance >= 0
            mask_other = oz > -1.0
            mask_robot = rz > -1.0
            mask = mask_contact * mask_robot * mask_other * 1.0
            contacts.append(mask)
        return contacts
