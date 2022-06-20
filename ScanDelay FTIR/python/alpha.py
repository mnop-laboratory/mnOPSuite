# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:55:28 2021

Required packages:
    websocket_client

Try to run "pip install websocket_client" from terminal if package is not found
or install websocket_client in your package manager (e.g. anaconda, ...)

@author: tobias
"""

# !/usr/bin/env python

from websocket import create_connection
import time
import numpy as np
import sympy as sym

class WsHandler(object):

    latest_socket = 8000

    def __init__(self, uri,timeout=1):
        self.useJSON = True

        full_uri = ':'.join((uri, str(self.latest_socket)))
        try:
            self.ws = create_connection(full_uri, timeout)
            self.connected = True
            print("%s connected!" % full_uri)
            #WsHandler.latest_socket +=1 #increment the socket index, we don't want to reconnect to this one

        except:
            self.connected = False
            raise RuntimeError('Alpha websocket not connected! Could not connect to %s' % full_uri)

    def _toJSON(self, msg, cmd_id=100):
        return ('{"cmd_id":%d,"data":"%s"}' % (cmd_id, msg))

    def _fromJSON(self, msg):
        return msg.split("data")[1].replace('""', '').replace(":", "").replace("}", "")

    def send(self, msg):
        if not self.connected:
            raise RuntimeError('Alpha websocket not connected!')
        if self.useJSON:
            self.ws.send(self._toJSON(msg))
        else:
            self.ws.send(msg)

    def recv(self):
        if not self.connected:
            raise RuntimeError('Alpha websocket not connected!')
        if self.useJSON:
            return self._fromJSON(self.ws.recv())
        else:
            return self.recv()

    def close(self):
        if not self.connected:
            return
        self.ws.close()
        self.connected = False

alpha_uri = "ws://192.168.51.1"  # alpha address


def pump_generator(wns=(600,1000,1200,1400,1600,1800),
                   pump_vals=np.array((100,10,6,5,10,15))):
    """Generate a function to estimate desired mid-IR pump values;
    function output will be extrapolated from a table of pump values at fixed mid-IR energies."""

    xs = np.array(wns).astype(float)
    ys = np.array(pump_vals).astype(float)

    N = len(ys) + 1
    coeffs = sym.symbols(' '.join(['c%i' % i for i in range(N)]))
    x = sym.symbols('x')
    xmin = xs[np.argmin(ys)]

    p = sum([coeffs[i] * x ** i for i in range(N)])
    ps = [p.subs(x, xs[i]) - ys[i] for i in range(N - 1)]
    ps.append(sym.diff(p, x).subs(x, xmin))
    sol = sym.solve(ps, dict=True)[0]
    polyfunc = sym.lambdify(x, p.subs(sol))

    return polyfunc

def wn_to_wl(wn): return np.round(1e7/wn)

def set_replicator_table(wavelengths=None,wns=None,pumps=None):

    assert wavelengths is not None or wns is not None
    assert wavelengths is None or wns is None
    if wns is not None: wavelengths = wn_to_wl(wns)
    N = len(wavelengths)
    if pumps is not None: assert len(pumps)==N

    Ws = WsHandler(alpha_uri)
    try:
        for i in range(N):
            print('Programming alpha replication entry i=%i of %i...' % (i,N))

            cmd = "@mir setsignal %i" % wavelengths[i]
            print(cmd)
            Ws.send(cmd)
            time.sleep(20)  # make sure the delay is at least 0.5 s

            if pumps is not None:
                cmd = "@mir setpump %1.1f" % pumps[i]
                print(cmd)
                Ws.send(cmd)
                time.sleep(2)  # make sure the delay is at least 0.5 s

            cmd = "@rep save %i" % i
            print(cmd)
            Ws.send(cmd)
            time.sleep(1)  # make sure the delay is at least 0.5 s

        Ws.close()
    except:
        Ws.close()
        raise

def touchup_replicator_table(N,pumps=None,optimize=True):

    Ws = WsHandler(alpha_uri)

    global pumpmax
    if hasattr(pumps,'__len__') and len(pumps)==N:
        setpump=True
        for pump in pumps:
            if pump>pumpmax:
                raise ValueError('pump value %1.1f%% greater than the maximum, %1.1f%%!'%(pump,pumpmax))
    else: setpump=False

    try:
        for i in range(20):

            print('Touching up replication i=%i' % i)

            cmd = "@rep load %i" % i
            Ws.send(cmd)
            time.sleep(10)  # make sure the delay is at least 0.5 s

            if optimize:
                print('Optimizing...')
                cmd = "@mir optimize"
                Ws.send(cmd)
                time.sleep(7)  # make sure the delay is at least 0.5 s

            if setpump:
                print('Setting pump=%1.1f...'%pumps[i])
                cmd='@mir setpump %1.1f'%pumps[i]
                Ws.send(cmd)
                time.sleep(1)

            cmd = "@rep save %i" % i
            Ws.send(cmd)
            time.sleep(1)  # make sure the delay is at least 0.5 s

        Ws.close()
    except:
        Ws.close()
        raise

def replicate(idx,wait_time=10,optimize=False): #wait time is in seconds

    ws = WsHandler(alpha_uri)

    module = "rep"  # name of the module
    task = "load"  # wavelength tuning command

    try:
        cmd = "@%s %s %i" % (module, task, idx)
        print("sending cmd: %s" % cmd)
        ws.send(cmd)
        time.sleep(np.maximum(wait_time, 0.5))  # make sure the delay is at least 0.5 s

        ws.close()

    except:
        ws.close()
        raise

    return cmd

def replicate_sequence(accumulation_no,wait_time=10,optimize=False):

    """if accumulation_no<20:
        #We'll do 10 on `idx=0` and 10 on `idx=10`
        if accumulation_no<10: idx=0
        else: idx=10
    else:"""
    idx = accumulation_no % 20 #so if we reached 20, or 40, we restart at 0

    return replicate(idx,wait_time,optimize)