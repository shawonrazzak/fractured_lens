import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drone_model import DroneSimulator 

# ----- FUNCTIONS FOR MOTION (AKA TRAJECTORIES) ----- 

# 1
def apply_sine_wave_path_forward(psi_array, tsim, 
                                  start_time, duration, 
                                  num_waves, amplitude_deg):
    amplitude = np.deg2rad(amplitude_deg)
    end_time = start_time + duration

    for i, t in enumerate(tsim):
        if start_time <= t < end_time:
            progress = (t - start_time) / duration
            psi_array[i] += amplitude * np.sin(2 * np.pi * num_waves * progress)

# 2
def apply_smooth_turn(psi_array, tsim, 
                      start_time, turn_duration, 
                      final_angle):
    i_start = np.argmin(np.abs(tsim - start_time))
    psi_start = psi_array[i_start]
    psi_end = np.deg2rad(final_angle)
    t_turn_end = start_time + turn_duration

    for i, t in enumerate(tsim):
        if start_time <= t < t_turn_end:
            turn_progress = (t - start_time) / turn_duration
            easing = 0.5 - 0.5 * np.cos(np.pi * turn_progress)
            psi_array[i] = psi_start + (psi_end - psi_start) * easing
        elif t >= t_turn_end:
            psi_array[i] = psi_end

# 3
def apply_constant_wind(tsim, wind_speed, wind_direction_deg):
    w_array = wind_speed * np.ones_like(tsim)
    zeta_array = np.deg2rad(wind_direction_deg) * np.ones_like(tsim)
    return w_array, zeta_array

# 4
def apply_move_forward_and_speed(g_array, tsim, start_time, duration, final_speed):
    end_time = start_time + duration
    i_start = np.argmin(np.abs(tsim - start_time))
    start_speed = g_array[i_start]

    for i, t in enumerate(tsim):
        if start_time <= t < end_time:
            progress = (t - start_time) / duration
            easing = 0.5 - 0.5 * np.cos(np.pi * progress)
            g_array[i] = start_speed + (final_speed - start_speed) * easing
        elif t >= end_time:
            g_array[i] = final_speed

# 5
def apply_sine_speed_variation(
    psi_array, g_array, tsim,
    start_time, duration,
    num_waves, amplitude_deg,
    max_increase, max_decrease):

    amplitude = np.deg2rad(amplitude_deg)
    end_time = start_time + duration
    wave_duration = duration / num_waves

    for wave_idx in range(num_waves):
        t0 = start_time + wave_idx * wave_duration
        t_mid = t0 + wave_duration / 2
        t1 = t0 + wave_duration

        left_indices = np.where((tsim >= t0) & (tsim < t_mid))[0]
        right_indices = np.where((tsim >= t_mid) & (tsim < t1))[0]

        if len(left_indices) > 0:
            psi_start = psi_array[left_indices[0]]
            g_start = g_array[left_indices[0]]
            g_target = g_start + max_increase

            for i, idx in enumerate(left_indices):
                s = i / len(left_indices)
                easing = 0.5 - 0.5 * np.cos(np.pi * s)
                psi_array[idx] += amplitude * np.sin(np.pi * s)
                g_array[idx] = g_start + (g_target - g_start) * easing

        if len(right_indices) > 0:
            psi_start = psi_array[right_indices[0]]
            g_start = g_array[right_indices[0]]
            g_target = max(g_start - max_decrease, 1.0)

            for i, idx in enumerate(right_indices):
                s = i / len(right_indices)
                easing = 0.5 - 0.5 * np.cos(np.pi * s)
                psi_array[idx] += amplitude * np.sin(np.pi * (s + 1))
                g_array[idx] = g_start + (g_target - g_start) * easing


def apply_sharp_turn(psi_array, g_array, tsim, turn_time, pre_turn_speed, post_turn_angle_deg, post_turn_speed):
    i_turn = np.argmin(np.abs(tsim - turn_time))
    g_array[:i_turn] = pre_turn_speed
    psi_array[i_turn:] = np.deg2rad(post_turn_angle_deg)
    g_array[i_turn:] = post_turn_speed


def execute_face_angle(psi_array, angle_deg):
    psi_array[:] = np.deg2rad(angle_deg)


def execute_tangent_path(psi_array, psi_global_array):
    psi_array[:] = psi_global_array[:]


def apply_spiral_path(psi_array, tsim, start_time, duration, revolutions, clockwise=True):
    end_time = start_time + duration
    direction = -1 if clockwise else 1
    total_angle = direction * 2 * np.pi/2 * revolutions

    for i, t in enumerate(tsim):
        if start_time <= t <= end_time:
            progress = (t - start_time) / duration
            psi_array[i] += total_angle * progress
        elif t > end_time:
            psi_array[i] += total_angle


# ----- MAIN SIMULATOR -----

def main(): 
    # TIME DEFINITIONS
    dt = 0.1
    T = 4
    tsim = np.arange(0, T + dt/2, step=dt)

    # BASE CONDITIONS
    psi_global = np.deg2rad(90) * np.ones_like(tsim)  # trajectory direction
    psi = np.deg2rad(0) * np.ones_like(tsim)         # facing direction
    g = 7 * np.ones_like(tsim)
    z = 2.0 * np.ones_like(tsim)

    #----------------APPLY MOTION FUNCTIONS HERE---------------
    time_control = 0.0 

    #apply_sharp_turn(psi_array=psi_global, g_array=g, tsim=tsim, turn_time=5.0, pre_turn_speed=2.0, post_turn_angle_deg=135.0, post_turn_speed=8.0)
    apply_sine_speed_variation(psi_global, g, tsim, time_control, 4, 3, 350.0, 11.0, 2.0)
    time_control += 10.0        # advance by the duration you just used
    execute_tangent_path(psi, psi_global)
    

    #---------------------------------------------------------

    #----------------APPLY WIND CONDITIONS HERE---------------
    w, zeta = apply_constant_wind(tsim, wind_speed=10.0, wind_direction_deg=-225)
    #---------------------------------------------------------

    #----------------CODE I SHOULDN'T MODIFY WITHOUT TALKING TO MY PI OR POSTDOC---------------
    # Compute global x-y velocities using psi_global (trajectory)
    v_x_global = g * np.cos(psi_global)
    v_y_global = g * np.sin(psi_global)

    # Convert to body-frame using heading (psi)
    v_x = v_x_global * np.cos(psi) + v_y_global * np.sin(psi)
    v_y = -v_x_global * np.sin(psi) + v_y_global * np.cos(psi)

    mpc_horizon = 10
    mpc_control_penalty = 1e-2

    simulator = DroneSimulator(dt=dt, mpc_horizon=mpc_horizon, r_u=mpc_control_penalty)
    print(simulator.params.__dict__.keys())
    simulator.params.Mm = 2.0

    simulator.update_setpoint(v_x=v_x, v_y=v_y, psi=psi, z=z, w=w, zeta=zeta)

    # Run simulation
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=None, mpc=True, return_full_output=True)
    sim_data = pd.DataFrame(y_sim)
    sim_data.insert(0, 'time', t_sim)
    print('Done.')

    simulator.plot_trajectory(size_radius=0.7)
    plt.show()

    save_prompt = input("Do you want to save the simulation data to a CSV file? (y/n): ").strip().lower()
    if save_prompt == 'y':
        csv_filename = input("Enter a name for the CSV file (e.g., grad_school.csv): ").strip()
        if not csv_filename.endswith('.csv'):
            csv_filename += '.csv'
        sim_data.to_csv(csv_filename, index=False)
        print(f"Simulation data saved to '{csv_filename}'")
    else:
        print("Simulation data was not saved.")
#---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
