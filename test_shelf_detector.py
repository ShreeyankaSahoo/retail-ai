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


# Second reading: no change
reading_2 = {
    "rack_1": 14.2,
    "rack_2": 28.5,
    "rack_3": -1.0,
    "rack_4": 12.0,
}

print("\nREADING 2 - NO CHANGE")
print(detector.update(reading_2))


# Third reading: rack_1 increased by 2 cm
# This should trigger the stock alert.
reading_3 = {
    "rack_1": 16.2,
    "rack_2": 28.5,
    "rack_3": -1.0,
    "rack_4": 12.0,
}

print("\nREADING 3 - STOCK REMOVED")
print(detector.update(reading_3))
