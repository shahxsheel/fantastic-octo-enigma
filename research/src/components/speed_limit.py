#!/usr/bin/env python3
"""
Speed Limit Checker
Gets the speed limit of a road given GPS coordinates using OpenStreetMap data
"""

import requests
from typing import Optional, Tuple


class SpeedLimitChecker:
    """
    Checks speed limits for roads at given coordinates using OpenStreetMap Overpass API
    """

    def __init__(self, search_radius: int = 50):
        """
        Initialize the speed limit checker

        Args:
            search_radius: Radius in meters to search for roads (default: 50m)
        """
        self.search_radius = search_radius
        self.overpass_url = "https://overpass-api.de/api/interpreter"

    def get_speed_limit(self, latitude: float, longitude: float) -> Optional[int]:
        """
        Get the speed limit at the given coordinates

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Speed limit in MPH, or None if not found
        """
        try:
            # Build Overpass query to find roads near the coordinates
            query = f"""
            [out:json];
            (
              way(around:{self.search_radius},{latitude},{longitude})["highway"];
            );
            out body;
            """

            response = requests.post(
                self.overpass_url,
                data={"data": query},
                timeout=10
            )

            if response.status_code != 200:
                print(f"API request failed with status {response.status_code}")
                return None

            data = response.json()

            if not data.get("elements"):
                return None

            # Get the closest road with a speed limit
            for element in data["elements"]:
                if "tags" in element and "maxspeed" in element["tags"]:
                    maxspeed = element["tags"]["maxspeed"]
                    return self._parse_speed_limit(maxspeed)

            return None

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return None
        except Exception as e:
            print(f"Error getting speed limit: {e}")
            return None

    def _parse_speed_limit(self, maxspeed: str) -> Optional[int]:
        """
        Parse speed limit string to MPH integer

        Args:
            maxspeed: Speed limit string (e.g., "30 mph", "50", "80 km/h")

        Returns:
            Speed limit in MPH
        """
        try:
            # Remove extra spaces
            maxspeed = maxspeed.strip().lower()

            # Handle "mph" explicitly marked
            if "mph" in maxspeed:
                return int(maxspeed.replace("mph", "").strip())

            # Handle "km/h" or "kmh" - convert to mph
            if "km" in maxspeed or "kmh" in maxspeed:
                kmh = int(maxspeed.replace("km/h", "").replace("kmh", "").strip())
                return int(kmh * 0.621371)  # Convert km/h to mph

            # Handle numeric only (assume mph in US, km/h in most other places)
            # Default to mph for this US-based project
            if maxspeed.isdigit():
                return int(maxspeed)

            # Handle special values
            if maxspeed in ["none", "unlimited", "signals"]:
                return None

            return None

        except (ValueError, AttributeError):
            return None

    def get_speed_limit_with_details(
        self,
        latitude: float,
        longitude: float
    ) -> Tuple[Optional[int], Optional[dict]]:
        """
        Get speed limit and additional road details

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Tuple of (speed_limit_mph, road_info_dict)
        """
        try:
            query = f"""
            [out:json];
            (
              way(around:{self.search_radius},{latitude},{longitude})["highway"];
            );
            out body;
            """

            response = requests.post(
                self.overpass_url,
                data={"data": query},
                timeout=10
            )

            if response.status_code != 200:
                return None, None

            data = response.json()

            if not data.get("elements"):
                return None, None

            # Find closest road with most information
            for element in data["elements"]:
                if "tags" in element:
                    tags = element["tags"]

                    road_info = {
                        "name": tags.get("name", "Unknown"),
                        "highway_type": tags.get("highway", "Unknown"),
                        "surface": tags.get("surface", "Unknown"),
                        "lanes": tags.get("lanes", "Unknown"),
                    }

                    speed_limit = None
                    if "maxspeed" in tags:
                        speed_limit = self._parse_speed_limit(tags["maxspeed"])

                    return speed_limit, road_info

            return None, None

        except Exception as e:
            print(f"Error getting road details: {e}")
            return None, None


def main():
    """Example usage"""
    checker = SpeedLimitChecker()

    # Example: San Francisco coordinates
    lat = 37.7749
    lon = -122.4194

    print(f"Checking speed limit at ({lat}, {lon})...")
    speed_limit = checker.get_speed_limit(lat, lon)

    if speed_limit:
        print(f"Speed limit: {speed_limit} MPH")
    else:
        print("Speed limit not found")

    # Get detailed information
    speed_limit, road_info = checker.get_speed_limit_with_details(lat, lon)
    if road_info:
        print(f"\nRoad details:")
        print(f"  Name: {road_info['name']}")
        print(f"  Type: {road_info['highway_type']}")
        print(f"  Surface: {road_info['surface']}")
        print(f"  Lanes: {road_info['lanes']}")
        if speed_limit:
            print(f"  Speed limit: {speed_limit} MPH")


if __name__ == "__main__":
    main()
