from shelf_detector import ShelfDetector


detector = ShelfDetector(removal_distance_cm=1.5)


# First reading: baseline
reading_1 = {
    "rack_1": 14.2,
    "rack_2": 28.5,
    "rack_3": -1.0,
    "rack_4": 12.0,
}

print("READING 1")
print(detector.update(reading_1))


# Second reading: rack_1 distance increased by 2 cm.
# This should trigger the stock alert.
reading_2 = {
    "rack_1": 16.2,
    "rack_2": 28.5,
    "rack_3": -1.0,
    "rack_4": 12.0,
}

print("\nREADING 2")
print(detector.update(reading_2))
