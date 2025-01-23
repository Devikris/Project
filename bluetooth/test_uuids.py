import asyncio
from bleak import BleakClient

WATCH_MAC_ADDRESS = "31:E5:D2:5E:3B:19"  # Replace with your MAC address
heart_rate_uuid = "00002a19-0000-1000-8000-00805f9b34fb"
blood_pressure_uuid = "00002a25-0000-1000-8000-00805f9b34fb"  # Updated UUID for BP
blood_oxygen_uuid = "6e400003-b5a3-f393-e0a9-e50e24dcca9f"

async def fetch_smartwatch_data():
    async with BleakClient(WATCH_MAC_ADDRESS) as client:
        if client.is_connected:
            print("Connected to Smartwatch")

            try:
                heart_rate_data = await client.read_gatt_char(heart_rate_uuid)
                blood_pressure_data = await client.read_gatt_char(blood_pressure_uuid)
                blood_oxygen_data = await client.read_gatt_char(blood_oxygen_uuid)

                # Handle Heart Rate
                heart_rate = int.from_bytes(heart_rate_data, byteorder="little")
                print(f"Heart Rate: {heart_rate} BPM")

                # Handle Blood Pressure (if data is available)
                if blood_pressure_data:
                    print(f"Blood Pressure Data: {blood_pressure_data.decode('utf-8')}")
                else:
                    print("No Blood Pressure Data available")

                # Handle Blood Oxygen (if data is available)
                if blood_oxygen_data:
                    blood_oxygen = int.from_bytes(blood_oxygen_data, byteorder="little")
                    print(f"Blood Oxygen: {blood_oxygen}%")
                else:
                    print("No Blood Oxygen Data available")

            except Exception as e:
                print(f"Error reading data: {e}")
        else:
            print("Failed to connect to the smartwatch.")

asyncio.run(fetch_smartwatch_data())
