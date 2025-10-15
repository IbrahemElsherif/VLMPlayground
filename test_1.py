import serial
import time

# --- SETUP ---
# Find your Arduino's serial port.
# On Windows, it's usually 'COM3', 'COM4', etc.
# On Linux (like Jetson Nano) or Mac, it's '/dev/ttyACM0', '/dev/ttyUSB0', etc.
# You can find the correct port in the Arduino IDE under Tools > Port.
SERIAL_PORT = 'COM10'  # <--- CHANGE THIS TO YOUR PORT
BAUD_RATE = 9600

# --- MAIN ---
try:
    # Establish a connection to the serial port.
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    # Wait a moment for the connection to initialize.
    time.sleep(2) 
    print(f"Connected to Arduino on {SERIAL_PORT}")

    # --- MOVEMENT LOGIC ---
    # This loop will make the servos sweep back and forth, just like before,
    # but now the commands are coming from Python!
    user_input = input("Enter thec command: Right or Left!")
    if user_input == "Right":
        while True:
            print("Sweeping Out...")
            for angle in range(181):
                inverted_angle = 180 - angle
                
                # Create the command string in the format "angle1,angle2"
                command = f"{angle},{inverted_angle}"
                
                # Send the command to the Arduino.
                arduino.write(command.encode())
                
                # Wait for a response from the Arduino (optional, but good for debugging).
                response = arduino.readline().decode().strip()
                if response:
                    print(f"Arduino response: {response}")

                time.sleep(0.015) # Controls the speed of the sweep.

            print("Sweeping Back...")
            for angle in range(180, -1, -1):
                inverted_angle = 180 - angle
                
                command = f"{angle},{inverted_angle}"
                arduino.write(command.encode())
                
                response = arduino.readline().decode().strip()
                if response:
                    print(f"Arduino response: {response}")
                
                time.sleep(0.015)
                
            else:
                print("enter valid command")

except Exception as e:
    print(f"Error: {e}")
    print("Could not connect to the Arduino. Check the SERIAL_PORT variable.")

finally:
    # Close the connection when the script is stopped.
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Serial connection closed.")
