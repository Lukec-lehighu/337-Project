import mujoco as mj
from mujoco.glfw import glfw

from model import predict, replay_buffer, train_dqn

import matplotlib.pyplot as plt
import numpy as np
import math
import os

ENVIRONMENT_PATH = 'environment.xml'

NUM_EPISODES = 5000
MAX_STEPS = 700

EPSILON = 0.1

# for random spawning
MAX_X = [-0.7,0.7]
MAX_Y = [-0.7,0.7]

MAX_X = [-0.4,0.4]
MAX_Y = [-0.4,0.4]

MIN_DISTANCE_FOR_FINISH = 0.03
TARGET_SPEED_FOR_FINISH = 0.03

MAX_TILT = 15 # max number of degrees that the platform can tilt (prevent runaway values in target rotation)

#for model outputs
move_speed = 0.05
platform_rot = np.zeros((2,))

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

isDone = False
old_state = []
action_took = False
state = []
action = []
loop_num = 0

#for model inputs (where we want the ball to go)
target_x = 0
target_y = 0

#for debugging
def print_state(state):
    os.system('cls')
    print(f"Ball loc: {state[0]}, {state[1]}")
    print(f"Ball vel: {state[2]}, {state[3]}")

def get_ctrl_for_pred(action):
    global platform_rot
    if action == 0:
        platform_rot += np.array([move_speed, 0])
    elif action == 1:
        platform_rot += np.array([-move_speed, 0])
    elif action == 2:
        platform_rot += np.array([0, move_speed])
    elif action == 3:
        platform_rot += np.array([0, -move_speed])

    platform_rot = np.clip(platform_rot, -MAX_TILT, MAX_TILT)

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def reset(data):
    #set random start location for the ball
    global loop_num, platform_rot
    data.qpos[2] = np.random.uniform(MAX_X[0], MAX_X[1])
    data.qpos[3] = np.random.uniform(MAX_Y[0], MAX_Y[1])

    platform_rot = np.zeros((2,)) # reset platform target rotation

    loop_num = 0

def get_state(data):
    '''
        Get the current state and return it to the main code
    '''

    #model inputs
    ball_x = data.qpos[2] / 1.5 # "normalize" somewhat
    ball_y = data.qpos[3] / 1.5

    ball_xvel = data.qvel[2]
    ball_yvel = data.qvel[3]

    floor_rotx = data.sensor('plat_rx').data.copy()[0]
    floor_roty = data.sensor('plat_ry').data.copy()[0]

    return [ball_x, ball_y, ball_xvel, ball_yvel, floor_rotx, floor_roty]

def calcReward(state, last_state):
    global isDone

    pos = np.array([state[0], state[1]])
    lastpos = np.array([last_state[0], last_state[1]])
    target = np.array([target_x, target_y])

     #calculate reward
    distance_to_target = math.dist(target, pos)
    ball_speed = math.dist([state[2], state[3]], [0,0])

    reward = -(distance_to_target**2)
    reward -= ball_speed # slower ball is better ball
    reward += 10 * (math.dist(lastpos, target) - math.dist(pos, target)) # reward getting closer to the target

    #calculate if is done
    if data.qpos[4] < -0.5: # ball height, when it gets below a certain point, it fell off of the platform :(
        reward -= 300 #punish the model for misbehaving
        isDone = True
    else:
        reward += 2 # reward staying on the platform

    if distance_to_target < MIN_DISTANCE_FOR_FINISH and ball_speed <= TARGET_SPEED_FOR_FINISH:
        reward += 300
        isDone = True

    return reward

prediction = []
def controller(model, data):
    global prediction, loop_num, isDone, state, action, old_state, action_took

    if not action_took:
        state = get_state(data)
        old_state = state.copy()
        
        #predict
        action = predict(state, epsilon=EPSILON) # random prediction
        loop_num+=1

        get_ctrl_for_pred(action) # update platform_rot (target rotation)
        data.ctrl = [np.deg2rad(platform_rot[0]), np.deg2rad(platform_rot[1])]

        action_took = True


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                    dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, 0.05 *
                    yoffset, scene, cam)

def render():
    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                    mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

def train():
    global isDone, action_took, action

    #initialize the controller
    init_controller(model,data)

    #set the controller
    mj.set_mjcb_control(controller)
    rewards = []

    for episode in range(NUM_EPISODES):
        mj.mj_resetData(model, data)
        reset(data) # set up random start location and such

        total_reward = 0
        isDone = False

        if glfw.window_should_close(window):
            break

        for step in range(MAX_STEPS):
            time_prev = data.time
            while (data.time - time_prev < 2.0/60.0):
                mj.mj_step(model, data)

            next_state = get_state(data)
            reward = calcReward(state=next_state, last_state=state)
            
            #print_state(state)
            replay_buffer.append((old_state.copy(), action, reward, next_state.copy(), isDone)) #replay buffer is in model.py
            train_dqn()
            
            total_reward += reward # reward is set when in mj.mj_step (controller)

            if isDone:
                break

            # tell the model it can take another action
            action_took = False

            if episode % 10 == 0:
                render()

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        print(f'[+] Total reward for episode {episode}: {total_reward}')
        rewards.append(total_reward)

    glfw.terminate()

    plt.plot(rewards)
    plt.show()

# MuJoCo data structures
model = mj.MjModel.from_xml_path(ENVIRONMENT_PATH)   # MuJoCo model
data = mj.MjData(model)                              # MuJoCo data
cam = mj.MjvCamera()                                 # Abstract camera
opt = mj.MjvOption()                                 # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

if __name__=='__main__':
    train()