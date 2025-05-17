import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# Robot arm parameters
L1, L2, L3 = 1, 0.8, 0.6  # Lengths of arm segments
W1, W2, W3 = 0.2, 0.16, 0.12  # Widths of arm segments

# Quintic velocity profile
def quintic_trajectory(q0, qf, t, T):
    a0 = q0
    a1 = 0
    a2 = 0
    a3 = 10 * (qf - q0) / T**3
    a4 = -15 * (qf - q0) / T**4
    a5 = 6 * (qf - q0) / T**5
    
    q = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    dq = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    ddq = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    dddq = 6*a3 + 24*a4*t + 60*a5*t**2
    
    return q, dq, ddq, dddq

# Path planning
def generate_path(T, num_points):
    t = np.linspace(0, T, num_points)
    
    x = 0.8 * np.cos(t)
    y = 0.8 * np.sin(t)
    z = 1.2 + 0.2 * np.sin(2*t)
    
    dx = -0.8 * np.sin(t)
    dy = 0.8 * np.cos(t)
    dz = 0.4 * np.cos(2*t)
    
    ddx = -0.8 * np.cos(t)
    ddy = -0.8 * np.sin(t)
    ddz = -0.8 * np.sin(2*t)
    
    dddx = 0.8 * np.sin(t)
    dddy = -0.8 * np.cos(t)
    dddz = -1.6 * np.cos(2*t)
    
    return np.array([x, y, z]).T, np.array([dx, dy, dz]).T, np.array([ddx, ddy, ddz]).T, np.array([dddx, dddy, dddz]).T

# Total time and fps
T = 10  # Total time in seconds
fps = 30
num_points = T * fps

# Generate path
path, velocity, acceleration, jerk = generate_path(T, num_points)
x, y, z = path[:, 0], path[:, 1], path[:, 2]

# Inverse Kinematics
def inverse_kinematics(x, y, z):
    theta1 = np.arctan2(y, x)
    
    r = np.sqrt(x**2 + y**2)
    s = z - L1
    D = (r**2 + s**2 - L2**2 - L3**2) / (2 * L2 * L3)
    
    if D >= -1 and D <= 1:
        theta3 = np.arctan2(-np.sqrt(1 - D**2), D)
        theta2 = np.arctan2(s, r) - np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
        return theta1, theta2, theta3
    else:
        return None, None, None

# Calculate joint angles
theta1, theta2, theta3 = [], [], []
for xi, yi, zi in zip(x, y, z):
    t1, t2, t3 = inverse_kinematics(xi, yi, zi)
    theta1.append(t1)
    theta2.append(t2)
    theta3.append(t3)

theta1 = np.array(theta1)
theta2 = np.array(theta2)
theta3 = np.array(theta3)

# Set up the figure with subplots
fig = plt.figure(figsize=(12, 9))
ax_3d = fig.add_subplot(221, projection='3d')
ax_top = fig.add_subplot(222)
ax_front = fig.add_subplot(223)
ax_side = fig.add_subplot(224)

# Function to create arm segments
def create_segment(start, end, width):
    vec = end - start
    length = np.linalg.norm(vec)
    unit_vec = vec / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0], 0])
    
    corners = np.array([
        start + width/2 * perp_vec,
        start - width/2 * perp_vec,
        end - width/2 * perp_vec,
        end + width/2 * perp_vec
    ])
    
    return corners

# Initialize empty lists for the path
path_x, path_y, path_z = [], [], []

# Animation update function
def update(frame):
    ax_3d.clear()
    ax_3d.set_xlim(-2, 2)
    ax_3d.set_ylim(-2, 2)
    ax_3d.set_zlim(0, 2)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D View')
    ax_3d.view_init(elev=20, azim=45)

    ax_top.clear()
    ax_top.set_xlim(-2, 2)
    ax_top.set_ylim(-2, 2)
    ax_top.set_title('Top View')
    ax_top.set_aspect('equal')

    ax_front.clear()
    ax_front.set_xlim(-2, 2)
    ax_front.set_ylim(0, 2)
    ax_front.set_title('Front View')
    ax_front.set_aspect('equal')

    ax_side.clear()
    ax_side.set_xlim(-2, 2)
    ax_side.set_ylim(0, 2)
    ax_side.set_title('Side View')
    ax_side.set_aspect('equal')

    if theta1[frame] is not None and theta2[frame] is not None and theta3[frame] is not None:
        joint0 = np.array([0, 0, 0])
        joint1 = np.array([0, 0, L1])
        joint2 = np.array([L2 * np.cos(theta1[frame]) * np.cos(theta2[frame]),
                           L2 * np.sin(theta1[frame]) * np.cos(theta2[frame]),
                           L1 + L2 * np.sin(theta2[frame])])
        end_pos = np.array([x[frame], y[frame], z[frame]])

        # Append current position to path
        path_x.append(end_pos[0])
        path_y.append(end_pos[1])
        path_z.append(end_pos[2])

        # Create arm segments
        segment1 = create_segment(joint0, joint1, W1)
        segment2 = create_segment(joint1, joint2, W2)
        segment3 = create_segment(joint2, end_pos, W3)

        # Plot 3D view
        ax_3d.add_collection3d(Poly3DCollection([segment1], facecolors='red', alpha=0.7))
        ax_3d.add_collection3d(Poly3DCollection([segment2], facecolors='green', alpha=0.7))
        ax_3d.add_collection3d(Poly3DCollection([segment3], facecolors='blue', alpha=0.7))
        
        ax_3d.plot([joint0[0], joint1[0], joint2[0], end_pos[0]],
                   [joint0[1], joint1[1], joint2[1], end_pos[1]],
                   [joint0[2], joint1[2], joint2[2], end_pos[2]], 'ko-', linewidth=2, markersize=4)
        
        # Plot the path
        ax_3d.plot(path_x, path_y, path_z, 'r-', linewidth=1, alpha=0.7)

        # Plot top view
        ax_top.add_patch(plt.Polygon(segment1[:, [0, 1]], facecolor='red', alpha=0.7))
        ax_top.add_patch(plt.Polygon(segment2[:, [0, 1]], facecolor='green', alpha=0.7))
        ax_top.add_patch(plt.Polygon(segment3[:, [0, 1]], facecolor='blue', alpha=0.7))
        ax_top.plot([joint0[0], joint1[0], joint2[0], end_pos[0]],
                    [joint0[1], joint1[1], joint2[1], end_pos[1]], 'ko-', linewidth=2, markersize=4)
        ax_top.plot(path_x, path_y, 'r-', linewidth=1, alpha=0.7)

        # Plot front view
        ax_front.add_patch(plt.Polygon(segment1[:, [0, 2]], facecolor='red', alpha=0.7))
        ax_front.add_patch(plt.Polygon(segment2[:, [0, 2]], facecolor='green', alpha=0.7))
        ax_front.add_patch(plt.Polygon(segment3[:, [0, 2]], facecolor='blue', alpha=0.7))
        ax_front.plot([joint0[0], joint1[0], joint2[0], end_pos[0]],
                      [joint0[2], joint1[2], joint2[2], end_pos[2]], 'ko-', linewidth=2, markersize=4)
        ax_front.plot(path_x, path_z, 'r-', linewidth=1, alpha=0.7)

        # Plot side view
        ax_side.add_patch(plt.Polygon(segment1[:, [1, 2]], facecolor='red', alpha=0.7))
        ax_side.add_patch(plt.Polygon(segment2[:, [1, 2]], facecolor='green', alpha=0.7))
        ax_side.add_patch(plt.Polygon(segment3[:, [1, 2]], facecolor='blue', alpha=0.7))
        ax_side.plot([joint0[1], joint1[1], joint2[1], end_pos[1]],
                     [joint0[2], joint1[2], joint2[2], end_pos[2]], 'ko-', linewidth=2, markersize=4)
        ax_side.plot(path_y, path_z, 'r-', linewidth=1, alpha=0.7)

    return ax_3d, ax_top, ax_front, ax_side

# Create the animation
anim = FuncAnimation(fig, update, frames=num_points, interval=1000/fps, blit=False)

# Set up the writer
#writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation
#anim.save('robot_arm_animation.mp4', writer=writer)

#print("Animation saved as 'robot_arm_animation.mp4'")


plt.tight_layout()
plt.show()

# After the animation is complete, show the profiles for end effector and joints
t = np.linspace(0, T, num_points)

# End effector profiles
fig_ee, axs_ee = plt.subplots(4, 3, figsize=(15, 20))
fig_ee.suptitle('End Effector Profiles')

for i, (coord, label) in enumerate(zip(['X', 'Y', 'Z'], ['x', 'y', 'z'])):
    axs_ee[0, i].plot(t, path[:, i])
    axs_ee[0, i].set_ylabel(f'{coord} Position')
    axs_ee[0, i].grid(True)
    
    axs_ee[1, i].plot(t, velocity[:, i])
    axs_ee[1, i].set_ylabel(f'{coord} Velocity')
    axs_ee[1, i].grid(True)
    
    axs_ee[2, i].plot(t, acceleration[:, i])
    axs_ee[2, i].set_ylabel(f'{coord} Acceleration')
    axs_ee[2, i].grid(True)
    
    axs_ee[3, i].plot(t, jerk[:, i])
    axs_ee[3, i].set_ylabel(f'{coord} Jerk')
    axs_ee[3, i].set_xlabel('Time (s)')
    axs_ee[3, i].grid(True)

plt.tight_layout()
plt.show()

# Joint profiles
for j, (theta, label) in enumerate(zip([theta1, theta2, theta3], ['θ1', 'θ2', 'θ3'])):
    fig_joint, axs_joint = plt.subplots(4, 1, figsize=(10, 20))
    fig_joint.suptitle(f'Joint {j+1} Profiles')
    
    valid_indices = ~np.isnan(theta)
    t_valid = t[valid_indices]
    theta_valid = theta[valid_indices]
    
    axs_joint[0].plot(t_valid, theta_valid)
    axs_joint[0].set_ylabel(f'{label} Position')
    axs_joint[0].grid(True)
    
    velocity = np.gradient(theta_valid, t_valid)
    axs_joint[1].plot(t_valid, velocity)
    axs_joint[1].set_ylabel(f'{label} Velocity')
    axs_joint[1].grid(True)
    
    acceleration = np.gradient(velocity, t_valid)
    axs_joint[2].plot(t_valid, acceleration)
    axs_joint[2].set_ylabel(f'{label} Acceleration')
    axs_joint[2].grid(True)
    
    jerk = np.gradient(acceleration, t_valid)
    axs_joint[3].plot(t_valid, jerk)
    axs_joint[3].set_ylabel(f'{label} Jerk')
    axs_joint[3].set_xlabel('Time (s)')
    axs_joint[3].grid(True)
    
    plt.tight_layout()
    plt.show()
