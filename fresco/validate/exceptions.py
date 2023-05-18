"""
    Simple module to catch exception errors and print helpful messages.
"""
import os
import sys


class ParamError(Exception):
    """
        Simple class to catch exception errors and print helpful debugging messages.
    """
    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return self.msg
