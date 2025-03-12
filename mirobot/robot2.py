
from wlkata_mirobot import WlkataMirobot, WlkataMirobotTool
from time import sleep
import coin


hight_z = 35 #default height when picking up blocks
bx = 139 #defualt cooordinates for block calibration
by = 117 #default values for block calibration 
cx = 0 #coordiante offset
cy = 0 #coordiante offset 


#setting up the sorting area
color_roi = (bx-75, by-75, 150, 150)

arm = WlkataMirobot(portname="/dev/ttyUSB0") #global variable for arm

def start_process(cmd_queue):
    home()
    while True:
        if cmd_queue.empty() == False:
            print("Getting next command")
            command = cmd_queue.get()
            print(f"Working on command: {command}")
            manipulate_coin(command)
            print(f"Finished command: {command}")
            cmd_queue.task_done() # releases the lock on the queue

def manipulate_coin(coin_data):
    y_pretranslate = coin_data.x
    x_pretranslate = coin_data.y
    coin_type = coin_data.coinSize

    x_mm, y_mm = translate(y_pretranslate, x_pretranslate)

    arm.set_tool_type(WlkataMirobotTool.SUCTION_CUP)
    
    print(f"Moving to the coin x:{x_mm} y:{y_mm}")
    my_go_to_axis(x=(240 - x_mm), y=(10 + y_mm), z=75)
    sleep(2.0)
    my_go_to_axis(x=(240 - x_mm), y=(10 + y_mm), z=8)
    print("Suck the coin")
    arm.pump_suction()
    sleep(1.0)
    my_go_to_axis(x=(240 - x_mm), y=(10 + y_mm), z=75)
    sleep(1.0)
    print("Move to drop off zone")
    if coin_type == 0:
        my_go_to_axis(x = 175, y =-175, z = 30) #DIME
    elif coin_type == 1:
        my_go_to_axis(x = 120, y =-190, z = 30) # PENNY
    elif coin_type == 2:
        my_go_to_axis(x = 120, y =-150, z = 30) # NICKLE
    elif coin_type == 3:
        my_go_to_axis(x = 190, y =-140, z = 30) #QUARTER
    print("Drop the baby")
    arm.pump_off()
    sleep(1.0)
    print("Return to starting position")
    #home()

def translate(x, y):
    if x > 0 and y > 0 or x < 0 and y < 0:
        return y, x
    
    # switch the signs
    x_s, y_s = -x, -y

    # switch the values
    x_s_positive = 1 if x_s > 0 else -1
    y_s_positive = 1 if y_s > 0 else -1

    x_new = abs(y) * x_s_positive
    y_new = abs(x) * y_s_positive
    
    return x_new, y_new

def home():
    # arm controls
    #arm = WlkataMirobot(portname="/dev/ttyUSB0")
    arm.unlock_all_axis()
    print("Instantiate the Mirobot Arm instance")
    # arm = WlkataMirobot()
    # Mirobot Arm Multi-axis executing
    print("Homing start")
    # Note:
    # - In general, if there is no seventh axis, just execute arm.home(),
    # has_slider parameter is set to False by default
    # - If there is a slider (axis 7), set has_slider to True
    arm.home()
    my_go_to_axis(x=120, y=-190, z=200)
    # arm.home(has_slider=False)
    # arm.home(has_slider=True)
    print("Homing finish")
    # Status Update and Query
    print("update robotic arm status")
    arm.get_status()
    print(f"instance status after update: {arm.status}")
    print(f"instance status name after update: {arm.status.state}")
    pass

def my_go_to_axis(x=None, y=None, z=None):
	# instruction = 'M21 G90'  # X{x} Y{y} Z{z} A{a} B{b} C{c} F{speed}
    # instruction = 'M20 G90 G00'  # X{x} Y{y} Z{z} A{a} B{b} C{c} F{speed}
    instruction = 'M20 G90 G00 F2000'
    pairings = {'X': x, 'Y': y, 'Z': z}
    msg = arm.generate_args_string(instruction, pairings)
    return arm.send_msg(msg, wait_ok=True, wait_idle=True)


def move(x, y, z, a = 0, b = 0, c = 0):
    '''
    Move the robot by a the defined contrains above
    '''
    # arm controls
   
    arm.unlock_all_axis()
    arm.go_to_axis(x, y, z, a, b, c)
    print("Successfully moved the robot")
    


if __name__ == "__main__":
    print("directly running the robot.py script")
    #leftRight()
    #arm.pump_off()
    arm = WlkataMirobot(portname="/dev/ttyUSB0")
    home()
    print('after home, start to go to axis')
    #my_go_to_axis(x=80, y=-200, z=200) 
    # penny drop zone is 120,-190, 200
    # nickle drop zone is 220, -190, 75
    my_go_to_axis(x = 120, y =-190, z = 30) # PENNY
    my_go_to_axis(x = 120, y =-150, z = 30) # NICKLE
    my_go_to_axis(x = 175, y =-175, z = 30) #DIME
    my_go_to_axis(x = 190, y =-140, z = 30) #QUARTER
    # middle is x = 225, y = 10, and z = 9
    #my_go_to_axis(x = 210, y =-175, z = 75)
    print('before home, finish go to axis')

   
    
