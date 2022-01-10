#!/usr/bin/python3
"""
This file holds the code for a class that can be used to handle early-stopping in the step*.py files.
Can be used similar to the lr_scheduler-classes from pytorch.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

class EarlyStopping(object):
    """
    Use the .step attribute which can be used as a methode. E.g.: 'if early_stop.step(1.234) is True'.
    """
    def __init__(self, patience, delta=0.1, mode="rel"):
        """
        patience: Number of epochs without change to wait befor stop
        delta: change-threshold
        mode: type of threshold usage
        """
        self._patience = patience
        self._delta = delta
        #
        self._prev_values = []
        #
        if mode == "rel":
            self.step = self._step_rel
        elif mode == "abs":
            self.step = self._step_abs
        else:
            self.step = None
    
    def _step_rel(self, new_val):
        #
        if len(self._prev_values) < self._patience:
            self._prev_values.append(new_val)
            return False
        #
        prev_best = max(self._prev_values[:-self._patience+1])
        stop_flag = prev_best*(1+self._delta) > new_val
        #
        self._prev_values.append(new_val)
        return stop_flag
    
    def _step_abs(self, new_val):
        #
        if len(self._prev_values) < self._patience:
            self._prev_values.append(new_val)
            return False
        #
        prev_best = max(self._prev_values[:-self._patience+1])
        stop_flag = prev_best+self._delta > new_val
        #
        self._prev_values.append(new_val)
        return stop_flag
