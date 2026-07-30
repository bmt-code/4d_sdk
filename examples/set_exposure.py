#!/usr/bin/env python3
"""
Remote Exposure Controller for 4D Stereo Camera.
Sends live exposure toggle commands to an already running camera over ZeroMQ
without interrupting active image streaming.
"""

import time
import json
import argparse
import zmq

def set_camera_exposure(camera_ip: str, enable: bool):
    """
    Connects to the 4D Camera's command socket (port 5556) and updates exposure control live.
    
    :param camera_ip: IP address of the camera (e.g. '172.31.1.77')
    :param enable: True to enable custom Highlight AEC, False for standard Normal AEC
    """
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.LINGER, 1000)
    
    endpoint = f"tcp://{camera_ip}:5556"
    print(f"Connecting to camera command socket at {endpoint}...")
    pub_socket.connect(endpoint)

    # ZMQ requires a brief pause for the TCP handshake to establish
    time.sleep(0.3)

    command_payload = {
        "action": "set_exposure",
        "enable": enable
    }

    # Send multipart ZMQ message (topic: 'command', body: JSON)
    pub_socket.send_multipart([
        b"command",
        json.dumps(command_payload).encode('utf-8')
    ])

    # Allow ZMQ socket buffer time to flush
    time.sleep(0.2)
    pub_socket.close(linger=1000)
    context.term()

    mode_name = "ON (Highlight AEC)" if enable else "OFF (Normal AEC)"
    print(f"[SUCCESS] Camera exposure control set to {mode_name} on {camera_ip}")

def main():
    parser = argparse.ArgumentParser(
        description="Toggle 4D Stereo Camera exposure remotely while camera is running."
    )
    parser.add_argument(
        "state", 
        choices=["on", "off"], 
        help="Turn custom exposure control 'on' (Highlight AEC) or 'off' (Normal AEC)"
    )
    parser.add_argument(
        "--ip", 
        default="172.31.1.77", 
        help="Camera IP address (default: 172.31.1.77)"
    )
    
    args = parser.parse_args()
    enable = (args.state.lower() == "on")
    
    set_camera_exposure(args.ip, enable)

if __name__ == "__main__":
    main()