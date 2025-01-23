import asyncio
from bleak import BleakClient, BleakScanner

# Replace this with your smartwatch's MAC address
WATCH_MAC_ADDRESS = "31:E5:D2:5E:3B:19"

async def discover_services():
    async with BleakClient(WATCH_MAC_ADDRESS) as client:
        if client.is_connected:
            print(f"Connected to device: {WATCH_MAC_ADDRESS}")
            services = await client.get_services()
            for service in services:
                print(f"Service: {service.uuid}")
                for characteristic in service.characteristics:
                    print(f"  Characteristic: {characteristic.uuid}")
                    print(f"    Properties: {characteristic.properties}")
        else:
            print("Failed to connect to the smartwatch.")

# Discover the services and characteristics
asyncio.run(discover_services())
