from datetime import datetime


# Minimum increase in distance (cm) that indicates stock was removed.
REMOVAL_DISTANCE_CM = 1.5

# Valid ultrasonic sensor range.
MIN_VALID_DISTANCE_CM = 2.0
MAX_VALID_DISTANCE_CM = 400.0


class ShelfDetector:
    def __init__(self, removal_distance_cm=REMOVAL_DISTANCE_CM):
        self.removal_distance_cm = removal_distance_cm

        # Store the previous valid reading for each rack.
        self.previous_readings = {}

    def _is_valid_reading(self, distance_cm):
        """Check whether an ultrasonic reading is valid."""
        if distance_cm < 0:
            return False

        return MIN_VALID_DISTANCE_CM <= distance_cm <= MAX_VALID_DISTANCE_CM

    def update(self, telemetry):
        """
        Process the latest ultrasonic readings.

        Example input:
        {
            "rack_1": 14.2,
            "rack_2": 28.5,
            "rack_3": -1.0,
            "rack_4": 12.0
        }
        """

        stock_alert = False

        for rack_id, current_distance in telemetry.items():

            # Ignore invalid readings such as -1.0.
            if not self._is_valid_reading(current_distance):
                self.previous_readings[rack_id] = None
                continue

            previous_distance = self.previous_readings.get(rack_id)

            # Compare with the previous reading.
            if previous_distance is not None:

                distance_change = current_distance - previous_distance

                # Greater distance means the stock moved farther
                # away from the ultrasonic sensor.
                if distance_change >= self.removal_distance_cm:
                    stock_alert = True

            # Save current reading for the next measurement.
            self.previous_readings[rack_id] = current_distance

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "telemetry": telemetry,
            "stock_alert": stock_alert
        }
