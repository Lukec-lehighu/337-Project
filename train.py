import mujoco as mj
from mujoco.glfw import glfw
from scipy.spatial.transform import Rotation

from model import predict

import numpy as np
import os

ENVIRONMENT_PATH = 'environment.xml'

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

loop_num = 0
prediction = []
def controller(model, data):
    global prediction, loop_num

    #put the controller here. This function is called inside the simulation.
    vel = data.sensor('speed').data
    rot = data.sensor('rotation').data

    #model inputs
    ball_x = data.qpos[2]
    ball_y = data.qpos[3]

    ball_xvel = vel[0]
    ball_yvel = vel[1]

    floor_xrot = rot[0]
    floor_yrot = rot[1]

    #predict
    if loop_num % 100 == 0:
        prediction = predict([ball_x, ball_y, ball_xvel, ball_yvel, floor_xrot, floor_yrot], epsilon=1)
    loop_num+=1

    data.ctrl = prediction

    #debugging
    # print('-'*50)
    # print(f'Ball Speed:')
    # print(f'   X -> {ball_xvel}')
    # print(f'   Y -> {ball_yvel}')
    # print()
    # print("Ball Pos:")
    # print(f'   X -> {ball_x}')
    # print(f'   Y -> {ball_y}')
    # print()
    # print("Platform Angle:")
    # print(f'   X -> {floor_xrot}')
    # print(f'   Y -> {floor_yrot}')
    # print('-'*50)

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
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
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

def run_simulation():
    #initialize the controller
    init_controller(model,data)

    #set the controller
    mj.set_mjcb_control(controller)

    while not glfw.window_should_close(window):
        time_prev = data.time

        while (data.time - time_prev < 1.0/60.0):
            mj.mj_step(model, data)

        # if (data.time>=simend):
        #     break

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

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    glfw.terminate()

# MuJoCo data structures
model = mj.MjModel.from_xml_path(ENVIRONMENT_PATH)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

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
    run_simulation()