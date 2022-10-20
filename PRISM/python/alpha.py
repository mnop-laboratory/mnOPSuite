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

    # Keep these class-wide variables so that different socket instances by default do not overlap their cmd_id's
    latest_socket = 8000
    cmd_id = 100

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

    def _toJSON(self, msg, cmd_id=None):
        if cmd_id is None: cmd_id=self.cmd_id
        return ('{"cmd_id":%d,"data":"%s"}' % (cmd_id, msg))

    def _fromJSON(self, json_msg):
        msg = json_msg.split("data")[1].replace('"', '').replace(":", "").replace("}", "")
        cmd_id = json_msg.split("data")[0].replace('{"cmd_id":', '').replace(',"', "")
        return cmd_id, msg

    def send(self, msg, cmd_id=None):

        # Increment latest cmd_id if not using specified
        if cmd_id is None:
            self.cmd_id += 1
            cmd_id = self.cmd_id
        else: self.cmd_id = cmd_id

        if self.useJSON:
            self.ws.send(self._toJSON(msg, cmd_id))
        else:
            self.ws.send(msg)

    def recv(self, cmd_id=None, timeout=10):

        # Use last sent cmd_id if one is not specified
        if cmd_id is None: cmd_id=self.cmd_id

        if timeout <= 0:
            return None  # return None if counter is 0
        if self.useJSON:
            msg_cmd_id, msg = self._fromJSON(self.ws.recv())  # receive next message from websocket
            if (int(cmd_id) == int(msg_cmd_id)):  # check message for correct id
                return msg
            else:
                time.sleep(1) #2022.09.22 @ASM: added to prevent `None`
                return self.recv(cmd_id, timeout - 1)  # check next message
        else:
            return self.ws.recv()

    def close(self):
        if not self.connected:
            return
        self.ws.close()
        self.connected = False

alpha_uri = "ws://192.168.51.1"  # alpha address

def wn_to_wl(wn): return np.round(1e7/np.array(wn))

def wl_to_wn(wl): return np.round(1e7/np.array(wl))

class PiecewiseInterpolator(object):

    @staticmethod
    def _linear(wn_pair,val_pair):

        dwn = wn_pair[1]-wn_pair[0]
        dval = val_pair[1] - val_pair[0]

        x0 = wn_pair[0]
        y0 = val_pair[0]
        m = dval/dwn

        return lambda x: m*(x-x0) + y0

    def __init__(self,wns,vals,
                 max_val=None,min_val=None):

        assert len(wns)==len(vals)

        wns,vals = zip(*sorted(zip(wns,vals)))
        self.wns = np.array(wns)
        self.vals = np.array(vals)

        self.max_val = max_val
        self.min_val = min_val

        self.wn_pairs = [(wns[i],wns[i+1]) \
                    for i in range(len(wns)-1)]
        self.val_pairs = [(vals[i],vals[i+1]) \
                    for i in range(len(vals)-1)]
        self.functions = [self._linear(wn_pair,val_pair) \
                          for wn_pair,val_pair in \
                          zip(self.wn_pairs,self.val_pairs)]
        self.functions = [lambda x: vals[0] + 0*x] \
                         + self.functions \
                        + [lambda x: vals[-1] + 0*x]

    def __call__(self,wn):

        conditions = [wn<self.wns[0]] +\
                     [(wn>=wn_pair[0]) * (wn<wn_pair[1]) \
                      for wn_pair in self.wn_pairs] +\
                     [wn>=self.wns[-1]]

        result = np.piecewise(wn, conditions, self.functions)

        if self.max_val is not None:
            result[result>self.max_val] = self.max_val
        if self.min_val is not None:
            result[result<self.min_val] = self.min_val

        return result




def pump_generator(wns=(600,1000,1200,1400,1600,1800),
                   pump_vals=np.array((100,10,6,5,10,15)),
                   min_val=0,max_val=100):
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

    def wrapped(wn):

        result = polyfunc(wn)

        #Make sure output is array
        if hasattr(wn,'__len__'):
            if not hasattr(result,'__len__'):
                result = np.array([result]*len(wn))
        else: result=np.array(result)

        #Constrain output
        result[result>max_val]=max_val
        result[result<min_val]=min_val

        return result

    return wrapped

def get_pump_at_replication_wavelength(wl,
                                        wns=(600, 1000, 1200, 1400, 1600, 1800),
                                        pump_vals=(100, 10, 6, 5, 10, 15)):

    wns=np.array(wns).flatten() #They may come in as a row vector
    pump_vals=np.array(pump_vals).flatten()

    wn_rep = wl_to_wn(wl)
    pumpgen = PiecewiseInterpolator(wns,pump_vals,
                                    min_val=np.min(pump_vals),
                                    max_val=100)

    return pumpgen(wn_rep)

pumpmax=100

def set_pump(pump,pump_max=pumpmax):

    Ws = WsHandler(alpha_uri)
    try:
        print('Setting pump to %1.1f...' % pump )

        if pump > pump_max: pump = pump_max
        cmd = "@mir setpump %1.1f" % pump
        print(cmd)
        Ws.send(cmd)
        time.sleep(1)  # make sure the delay is at least 0.5 s

        Ws.close()
        return cmd
    except:
        Ws.close()
        raise

def set_pump_factor(factor=0.2):

    Ws = WsHandler(alpha_uri)
    try:
        print('Modifying MIR pump by factor of %1.1f...' % factor)

        # command id
        my_cmd_id = 100

        # request axis 7 position
        Ws.send("@ax 7 getpos", my_cmd_id)

        # receive message with correct id
        reply = Ws.recv(my_cmd_id)

        # print reply (float/int conversion might be useful for further processing)
        theta_deg = float(reply)

        power = np.sin(2 * theta_deg / 180 * np.pi) ** 2 * 100 # pump power as a percentage
        new_power = power * factor

        if new_power > 100:
            print('New power of %1.1f%% must be less than 100%%' % new_power)
            new_power = 100

        print('Changing pump power from %1.1f to %1.1f...' % (power, new_power) )

        cmd = "@mir setpump %1.1f" % new_power
        print(cmd)
        Ws.send(cmd)
        time.sleep(0.5)  # make sure the delay is at least 0.5 s

        Ws.close()
        return cmd
    except:
        Ws.close()
        raise

def set_replicator_table(wavelengths=None,wns=None,pumps=None,
                         pump_max=100):

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
                pump=pumps[i]
                if pump>pump_max: pump=pump_max
                cmd = "@mir setpump %1.1f" % pump
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
        for i in range(N):

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