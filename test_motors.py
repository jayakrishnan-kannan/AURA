#!/usr/bin/env python3

"""
Individual Motor Tester
-----------------------

Usage:
- Select one motor at a time
- Motor runs for 15 seconds
- Automatically stops
- Waits for next user input

Useful for:
- Finding faulty motor driver channels
- Checking motor wiring
- Verifying GPIO outputs
"""

import RPi.GPIO as GPIO
import time

# =====================================
# MOTOR GPIO DEFINITIONS
# =====================================
MOTOR_PINS = {
    "left_front": {"in1": 5, "in2": 6},
    "left_rear": {"in1": 12, "in2": 13},
    "right_front": {"in1": 19, "in2": 16},
    "right_rear": {"in1": 26, "in2": 20},
}

RUN_TIME = 5  # seconds

# =====================================
# GPIO SETUP
# =====================================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for motor in MOTOR_PINS.values():
    GPIO.setup(motor["in1"], GPIO.OUT)
    GPIO.setup(motor["in2"], GPIO.OUT)

    GPIO.output(motor["in1"], GPIO.LOW)
    GPIO.output(motor["in2"], GPIO.LOW)


# =====================================
# FUNCTIONS
# =====================================
def stop_all():
    for motor in MOTOR_PINS.values():
        GPIO.output(motor["in1"], GPIO.LOW)
        GPIO.output(motor["in2"], GPIO.LOW)


def run_motor(name):
    stop_all()

    pins = MOTOR_PINS[name]

    # Forward direction only
    GPIO.output(pins["in1"], GPIO.HIGH)
    GPIO.output(pins["in2"], GPIO.LOW)

    print(f"\n{name} RUNNING for {RUN_TIME} seconds...")

    time.sleep(RUN_TIME)

    stop_all()

    print(f"{name} STOPPED\n")


# =====================================
# MENU
# =====================================
menu = """
=============================
 INDIVIDUAL MOTOR TEST
=============================

1 -> left_front
2 -> left_rear
3 -> right_front
4 -> right_rear

q -> quit

Enter option:
"""

# =====================================
# MAIN LOOP
# =====================================
try:
    while True:

        choice = input(menu).strip().lower()

        if choice == "1":
            run_motor("left_front")

        elif choice == "2":
            run_motor("left_rear")

        elif choice == "3":
            run_motor("right_front")

        elif choice == "4":
            run_motor("right_rear")

        elif choice == "q":
            break

        else:
            print("Invalid option\n")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    stop_all()
    GPIO.cleanup()
    print("GPIO cleaned up")
