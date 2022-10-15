import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
import math
import pandas as pd
from multiprocessing import Process
import multiprocessing
import time
import gc

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

B = 1
N = 50
I = 18
V = 52
mu = 2.08
S = 0.015

pi = tf.constant(np.pi, dtype="float32")
t = np.linspace(0, 120, 1000)


def calculate_center_of_mass(parameters, show_plots=False):
    vertical_bar = tf.constant(
        [
            [0] * 6,
            [i*0.107/5 for i in range(6)],
        ],
        dtype="float32",
    )
    horizontal_bar = tf.constant(
        [
            [0.107*tf.cos(tf.constant(pi/2 + (pi/parameters[i + 16])*i)).numpy() for i in range(6)] + [0.107*tf.cos(tf.constant(pi/2 - (pi/parameters[i + 22])*i)).numpy() for i in range(6)],
            [0.107*tf.sin(tf.constant(pi/2 + (pi/parameters[i + 16])*i)).numpy() for i in range(6)] + [0.107*tf.sin(tf.constant(pi/2 - (pi/parameters[i + 22])*i)).numpy() for i in range(6)],
        ],
        dtype="float32"
    )
    v_centers = (vertical_bar[:, 1:] + vertical_bar[:, :-1])/2
    v_masses = tf.sqrt(tf.reduce_sum((vertical_bar[:, 1:] - vertical_bar[:, :-1])**2, axis=0)) * parameters[:5]
    
    h_centers = (horizontal_bar[:, 1:] + horizontal_bar[:, :-1])/2
    h_centers = tf.concat([h_centers[:, :5], h_centers[:, 6:]], axis=1)
    
    h_masses = tf.sqrt(tf.reduce_sum((horizontal_bar[:, 1:] - horizontal_bar[:, :-1])**2, axis=0)) * parameters[5:16]
    h_masses = tf.concat([h_masses[:5], h_masses[6:]], axis=0)
    
    total_mass = tf.reduce_sum(tf.concat([v_masses, h_masses], axis=0))
    center_of_mass = tf.reshape(tf.reduce_sum(tf.concat([v_centers * v_masses, h_centers * h_masses], axis=1), axis=1)/total_mass, shape=(2, 1))
        
    return center_of_mass, total_mass


def rotation_matrix(t, omega, i):
    return np.array([
        [np.cos(omega*t + (np.pi/2) + (2*np.pi*(i-1))/(3)), np.sin(omega*t + (np.pi/2) + (2*np.pi*(i-1))/(3))],
        [-np.sin(omega*t + (np.pi/2) + (2*np.pi*(i-1))/(3)), np.cos(omega*t + (np.pi/2) + (2*np.pi*(i-1))/(3))],
    ])


def delta(omega, t, i):
    value = omega*t + np.pi/2 + ((i - 1)*4*np.pi/6)
    return ((-np.pi*6 <= value) & (value <= 7*np.pi/6)).astype(int)


def torque(t, omega, r, m):
    return (-B*N*I*mu*S*(
        delta(omega, t, 1)*np.sin(omega*t + np.pi/2)
        + delta(omega, t, 2)*np.sin(omega*t + 7*np.pi/6)
        + delta(omega, t, 3)*np.sin(omega*t + 11*np.pi/6)
    # ) - 10*r[0]*m*(np.cos(omega*t + np.pi/2) + np.cos(omega*t + 7*np.pi/6) + np.cos(omega*t + 11*np.pi/6)))
    ) - 10*m*(
        (r[0]*np.cos(omega*t + np.pi/2) + r[1]*np.sin(omega*t + np.pi/2))
        + (r[0]*np.cos(omega*t + 7*np.pi/6) + r[1]*np.sin(omega*t + 7*np.pi/6))
        + (r[0]*np.cos(omega*t + 11*np.pi/6) + r[1]*np.sin(omega*t + 11*np.pi/6))
    ))


def domega_dt(t, omega, r, m):
    r1 = tf.tensordot(tf.constant(rotation_matrix(t, omega, 1), dtype="float32"), r, axes=[[1], [0]])
    r2 = tf.tensordot(tf.constant(rotation_matrix(t, omega, 2), dtype="float32"), r, axes=[[1], [0]])
    r3 = tf.tensordot(tf.constant(rotation_matrix(t, omega, 3), dtype="float32"), r, axes=[[1], [0]])
    r1_dot = tf.reduce_sum(tf.square(r1))
    r2_dot = tf.reduce_sum(tf.square(r2))
    r3_dot = tf.reduce_sum(tf.square(r3))
    return torque(t, omega, r, m)/(m*(r1_dot + r2_dot + r3_dot))


def evaluate_efficiency(parameters, t, show_plots=False):
    r, m = calculate_center_of_mass(parameters)
    omega = solve_ivp(domega_dt, [0, 120], np.array([0]), args=tuple([r, m]), t_eval=t).y[0]    
    power = omega*torque(t, omega, r, m)
    efficiency = power / (I * V)
    
    return tf.reduce_max(efficiency)


def calc_efficiency(index, chunk):
    print(f"Process: {index} has started")
    local_result = []
    
    df = pd.read_csv(f"eff_chunk_{index}.csv", header=None)
    next_index = df.shape[0]
    processed = next_index
    
    start = time.monotonic()
    
    # for c in chunk[next_index:]:
    for c in chunk:
        eff = evaluate_efficiency(c, t)
        local_result.append(eff.numpy())
        if len(local_result) >= 100 and len(local_result) % 100 == 0:
            df = pd.DataFrame(local_result)
            df.to_csv(f"eff_chunk_{index}.csv", mode="a", index=False, header=False)
            processed += 100
            now = time.monotonic()
            print(f"Process: {index}, processed:{len(local_result)}, elapsed: {now - start}s")
            local_result = []
            gc.collect()
    
    df = pd.DataFrame(local_result)
    df.to_csv(f"eff_chunk_{index}.csv", mode="a", index=False, header=False)
    processed += 100
    now = time.monotonic()
    print(f"Process: {index}, processed:{len(local_result)}, elapsed: {now - start}s")
    local_result = []
    gc.collect()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    df = pd.read_csv("params.csv", header=None)
    good_params = df.to_numpy()
    good_params = [tf.constant(g, dtype="float32") for g in good_params]
    
    # chunks = [(i, good_params[i*21084:min(i*21084 + 21084, 168676)]) for i in range(8)]
    chunks = [(7, good_params[168672:168676])]
    processes = []
    
    for chunk in chunks:
        process = Process(target=calc_efficiency, args=chunk)
        process.start()
        processes.append(process)

    for p in processes:
        p.join()